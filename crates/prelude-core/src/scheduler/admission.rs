use super::{Scheduler, SchedulerStep, SequenceStatus};

impl Scheduler {
    /// Schedule one step using a unified per-step token budget.
    ///
    /// Like vLLM V1: running requests are scheduled first (decode = 1 token,
    /// partial prefill = chunk), then waiting requests fill remaining budget.
    /// Large prefills are chunked to fit within the budget.
    pub(crate) fn get_mixed_batch(&mut self) -> Option<SchedulerStep> {
        let had_preemption = self.ensure_decode_capacity_tracked();

        let mut token_budget = self.config.max_num_batched_tokens;
        let mut prefill_ids: Vec<String> = Vec::new();
        let mut prefill_chunk_lens: Vec<usize> = Vec::new();
        let mut decode_ids: Vec<String> = Vec::new();

        // ── 1. Schedule RUNNING requests first ────────────────────────
        for seq in &self.running {
            if token_budget == 0 {
                break;
            }
            let remaining_prefill = seq.prefill_len();
            if remaining_prefill > 0 {
                // Partially prefilled — needs more prefill tokens
                let mut chunk = remaining_prefill;
                if self.config.long_prefill_token_threshold > 0 {
                    chunk = chunk.min(self.config.long_prefill_token_threshold);
                }
                chunk = chunk.min(token_budget);
                if chunk > 0 {
                    prefill_ids.push(seq.request_id.clone());
                    prefill_chunk_lens.push(chunk);
                    token_budget -= chunk;
                }
            } else {
                // Fully prefilled — needs 1 decode token
                decode_ids.push(seq.request_id.clone());
                token_budget -= 1;
            }
        }

        // ── 2. Schedule WAITING requests with remaining budget ────────
        // Like vLLM V1: skip waiting admission entirely if preemption
        // happened (system is under memory pressure).
        if !had_preemption && token_budget > 0 {
            self.admit_waiting_into_mixed(
                token_budget,
                &mut prefill_ids,
                &mut prefill_chunk_lens,
                &mut decode_ids,
            );
        }

        // ── 3. Update kv_computed_len for prefill requests ────────────
        // This is updated eagerly so subsequent schedule_step calls know
        // prefill progress. build_step_batch recovers the pre-update offset
        // via kv_computed_len - chunk_len.
        for (id, &chunk_len) in prefill_ids.iter().zip(prefill_chunk_lens.iter()) {
            if let Some(seq) = self.running.iter_mut().find(|s| s.request_id == *id) {
                let new_computed = seq.kv_computed_len + chunk_len;
                seq.kv_computed_len = new_computed;
                if new_computed >= seq.input_ids.len() {
                    seq.status = SequenceStatus::Decoding;
                }
            }
        }

        // ── 4. Decay new_token_ratio if we have decode work ───────────
        if !decode_ids.is_empty() {
            self.effective_new_token_ratio = (self.effective_new_token_ratio
                - self.config.new_token_ratio_decay)
                .max(self.config.min_new_token_ratio);
        }

        // ── 5. Build step ─────────────────────────────────────────────
        let forward_mode = match (prefill_ids.is_empty(), decode_ids.is_empty()) {
            (false, false) => super::ForwardMode::Mixed,
            (false, true) => super::ForwardMode::Prefill,
            (true, false) => super::ForwardMode::Decode,
            (true, true) => return None,
        };

        Some(SchedulerStep {
            prefill_request_ids: prefill_ids,
            prefill_chunk_lens,
            decode_request_ids: decode_ids,
            forward_mode,
        })
    }

    /// Legacy non-chunked prefill: schedule a batch of new prefill requests.
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
            self.config.max_num_batched_tokens,
            total_token_budget,
        );
        self.restore_preempted_to_waiting_queue(admission.preempted);

        if admission.to_prefill.is_empty() {
            return None;
        }

        let chunk_lens: Vec<usize> = admission
            .to_prefill
            .iter()
            .map(|seq| seq.input_ids.len())
            .collect();
        Some(SchedulerStep::prefill(
            self.promote_prefill_to_running(admission.to_prefill),
            chunk_lens,
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

    /// Admit waiting requests into a mixed batch using remaining token budget.
    fn admit_waiting_into_mixed(
        &mut self,
        mut token_budget: usize,
        prefill_ids: &mut Vec<String>,
        prefill_chunk_lens: &mut Vec<usize>,
        decode_ids: &mut Vec<String>,
    ) {
        self.sort_waiting_queue();

        let available_slots = self
            .config
            .max_running_requests
            .saturating_sub(self.running.len());

        let block_size = self.config.block_size;

        let mut admitted = 0usize;
        while !self.waiting_queue.is_empty()
            && admitted < available_slots
            && token_budget > 0
        {
            let seq = self.waiting_queue.front().expect("queue checked non-empty");

            // Block-level KV cache capacity check (like vLLM's allocate_slots).
            // Reserve blocks for the full prompt + capped future decode tokens.
            let total_tokens = seq.total_len()
                + seq.remaining_tokens().min(self.config.decode_reservation_cap as u32) as usize;
            let blocks_needed = total_tokens.div_ceil(block_size);
            if blocks_needed > self.available_blocks {
                break;
            }

            // Per-request prefill cap
            let mut chunk = seq.prefill_len();
            if self.config.long_prefill_token_threshold > 0 {
                chunk = chunk.min(self.config.long_prefill_token_threshold);
            }
            // Per-step token budget
            chunk = chunk.min(token_budget);

            // Deadlock prevention: if nothing else is scheduled and chunk would
            // be 0, force at least 1 token so the system makes progress.
            if chunk == 0 && prefill_ids.is_empty() && decode_ids.is_empty() {
                chunk = 1;
            }
            if chunk == 0 {
                break;
            }

            let mut seq = self.waiting_queue.pop_front().expect("queue checked non-empty");
            seq.status = SequenceStatus::Prefilling;
            self.tokens_in_use += seq.input_ids.len();
            // Reserve blocks for the entire request upfront (conservative).
            self.available_blocks = self.available_blocks.saturating_sub(blocks_needed);
            self.running.push(seq);

            let request_id = self.running.last().unwrap().request_id.clone();
            prefill_ids.push(request_id);
            prefill_chunk_lens.push(chunk);
            token_budget = token_budget.saturating_sub(chunk);
            admitted += 1;
        }
    }
}

// ── Legacy admission helpers (used by non-chunked path) ──────────────

#[derive(Default)]
struct AdmissionBatch {
    to_prefill: Vec<super::Sequence>,
    preempted: Vec<super::Sequence>,
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

    fn restore_preempted_to_waiting_queue(&mut self, preempted: Vec<super::Sequence>) {
        for victim in preempted.into_iter().rev() {
            self.waiting_queue.push_front(victim);
        }
    }

    fn promote_prefill_to_running(&mut self, mut to_prefill: Vec<super::Sequence>) -> Vec<String> {
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
}
