use super::{Scheduler, SchedulerStep, SeqFinishReason, Sequence, SequenceStatus};

#[derive(Default)]
struct AdmissionBatch {
    to_prefill: Vec<Sequence>,
    preempted: Vec<Sequence>,
}

impl Scheduler {
    fn admit_waiting_sequences(
        &mut self,
        available_slots: usize,
        mut prefill_token_budget: usize,
        mut total_token_budget: usize,
    ) -> AdmissionBatch {
        let mut admission = AdmissionBatch::default();
        let global_prefill_cap = self.config.max_prefill_tokens;

        while !self.waiting_queue.is_empty() && admission.to_prefill.len() < available_slots {
            let prompt_len = self
                .waiting_queue
                .front()
                .expect("queue was checked non-empty")
                .prefill_len();
            let estimated_total = {
                let sequence = self.waiting_queue.front().expect("queue was checked non-empty");
                sequence.total_len() + sequence.remaining_tokens() as usize
            };

            // Prompt exceeds the global per-step prefill cap — it can never fit,
            // so fail it immediately rather than silently blocking the queue.
            // This used to be a `break` that left oversized sequences stuck
            // forever in `waiting`, surfacing to clients as a request timeout.
            if prompt_len > global_prefill_cap {
                let mut sequence = self
                    .waiting_queue
                    .pop_front()
                    .expect("queue was checked non-empty");
                sequence.status = SequenceStatus::Finished;
                sequence.finish_reason = Some(SeqFinishReason::Abort(
                    format!(
                        "prompt length {prompt_len} exceeds max_prefill_tokens={global_prefill_cap}; \
                         raise --max-prefill-tokens or shorten the prompt"
                    ),
                ));
                self.finished.push(sequence);
                continue;
            }

            // Prompt fits the global cap but not this step's remaining budget.
            // Defer to the next scheduling step.
            if prompt_len > prefill_token_budget {
                break;
            }

            if estimated_total > total_token_budget {
                if let Some((freed, victim)) = self.preempt_one_from_running() {
                    total_token_budget += freed;
                    admission.preempted.push(victim);
                    if estimated_total > total_token_budget {
                        break;
                    }
                } else {
                    break;
                }
            }

            let mut sequence = self.waiting_queue.pop_front().expect("queue was checked non-empty");
            sequence.status = SequenceStatus::Prefilling;
            prefill_token_budget = prefill_token_budget.saturating_sub(prompt_len);
            total_token_budget = total_token_budget.saturating_sub(estimated_total);
            self.tokens_in_use += sequence.input_ids.len();
            admission.to_prefill.push(sequence);
        }

        admission
    }

    fn restore_preempted_to_waiting_queue(&mut self, preempted: Vec<Sequence>) {
        for victim in preempted.into_iter().rev() {
            self.waiting_queue.push_front(victim);
        }
    }

    fn promote_prefill_to_running(&mut self, mut to_prefill: Vec<Sequence>) -> Vec<String> {
        let request_ids = to_prefill
            .iter()
            .map(|sequence| sequence.request_id.clone())
            .collect();

        for sequence in &mut to_prefill {
            sequence.status = SequenceStatus::Decoding;
            sequence.kv_computed_len = sequence.input_ids.len();
        }

        self.running.extend(to_prefill);
        request_ids
    }

    pub(crate) fn get_new_prefill_batch(&mut self) -> Option<SchedulerStep> {
        if self.waiting_queue.is_empty() {
            return None;
        }

        self.sort_waiting_queue();

        let available_slots = self
            .config
            .max_running_requests
            .saturating_sub(self.running.len());
        if available_slots == 0 {
            return None;
        }

        let reserved_for_decode = (self.estimate_running_future_tokens() as f32
            * self.effective_new_token_ratio) as usize;
        let total_token_budget = self
            .config
            .max_total_tokens
            .saturating_sub(self.tokens_in_use)
            .saturating_sub(reserved_for_decode);

        let admission = self.admit_waiting_sequences(
            available_slots,
            self.config.max_prefill_tokens,
            total_token_budget,
        );
        self.restore_preempted_to_waiting_queue(admission.preempted);

        if admission.to_prefill.is_empty() {
            return None;
        }

        Some(SchedulerStep::prefill(
            self.promote_prefill_to_running(admission.to_prefill),
        ))
    }

    pub(crate) fn get_decode_batch(&mut self) -> Option<SchedulerStep> {
        if self.running.is_empty() {
            return None;
        }

        let needed = self.running.len();
        let available = self
            .config
            .max_total_tokens
            .saturating_sub(self.tokens_in_use);

        if needed > available {
            let mut deficit = needed - available;
            while deficit > 0 {
                if let Some((freed, victim)) = self.preempt_one_from_running() {
                    deficit = deficit.saturating_sub(freed);
                    self.waiting_queue.push_front(victim);
                } else {
                    break;
                }
            }
        }

        if self.running.is_empty() {
            return None;
        }

        self.effective_new_token_ratio = (self.effective_new_token_ratio
            - self.config.new_token_ratio_decay)
            .max(self.config.min_new_token_ratio);

        let request_ids = self
            .running
            .iter()
            .map(|sequence| sequence.request_id.clone())
            .collect();

        Some(SchedulerStep::decode(request_ids))
    }

    pub(crate) fn get_mixed_batch(&mut self) -> Option<SchedulerStep> {
        self.ensure_decode_capacity();
        self.sort_waiting_queue();

        let running_count = self.running.len();
        let available_slots = self
            .config
            .max_running_requests
            .saturating_sub(running_count);
        let reserved_for_current_decode = running_count;
        let reserved_for_future_decode = (self.estimate_running_future_tokens() as f32
            * self.effective_new_token_ratio) as usize;
        let total_token_budget = self
            .config
            .max_total_tokens
            .saturating_sub(self.tokens_in_use)
            .saturating_sub(reserved_for_current_decode)
            .saturating_sub(reserved_for_future_decode);

        let admission = self.admit_waiting_sequences(
            available_slots,
            self.config.max_prefill_tokens,
            total_token_budget,
        );
        self.restore_preempted_to_waiting_queue(admission.preempted);

        let decode_ids = self
            .running
            .iter()
            .map(|sequence| sequence.request_id.clone())
            .collect::<Vec<_>>();
        let prefill_ids = self.promote_prefill_to_running(admission.to_prefill);

        match (prefill_ids.is_empty(), decode_ids.is_empty()) {
            (false, false) => Some(SchedulerStep::mixed(prefill_ids, decode_ids)),
            (false, true) => Some(SchedulerStep::prefill(prefill_ids)),
            (true, false) => Some(SchedulerStep::decode(decode_ids)),
            (true, true) => None,
        }
    }
}
