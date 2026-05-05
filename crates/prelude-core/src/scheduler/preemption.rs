use super::{SchedulePolicy, Scheduler, Sequence, SequenceStatus};

impl Scheduler {
    pub(crate) fn preempt_one_from_running(&mut self) -> Option<(usize, Sequence)> {
        if self.running.is_empty() {
            return None;
        }

        let victim_idx = self
            .running
            .iter()
            .enumerate()
            .max_by_key(|(_, sequence)| sequence.arrival_time)
            .map(|(index, _)| index)?;

        let mut victim = self.running.remove(victim_idx);
        let freed = victim.total_len();
        self.tokens_in_use = self.tokens_in_use.saturating_sub(freed);

        // Free KV blocks directly (scheduler owns block_manager).
        if !victim.block_table.is_empty() {
            self.free_blocks(&victim.block_table);
        }

        victim.status = SequenceStatus::Waiting;
        victim.kv_computed_len = 0;
        victim.block_table.clear();
        victim.deltanet_slot = None;
        victim.preempt_count += 1;

        self.effective_new_token_ratio = self.config.new_token_ratio;

        Some((freed, victim))
    }

    /// Like `ensure_decode_capacity` but returns whether any preemption occurred.
    /// Used by `get_mixed_batch` to skip waiting admission after preemption
    /// (like vLLM V1's `if not preempted_reqs` guard).
    pub(crate) fn ensure_decode_capacity_tracked(&mut self) -> bool {
        let needed = self.running.len();
        let available = self
            .config
            .max_total_tokens
            .saturating_sub(self.tokens_in_use);

        if needed <= available {
            return false;
        }

        let mut deficit = needed - available;
        let mut had_preemption = false;
        while deficit > 0 {
            if let Some((freed, victim)) = self.preempt_one_from_running() {
                deficit = deficit.saturating_sub(freed);
                self.waiting_queue.push_front(victim);
                had_preemption = true;
            } else {
                break;
            }
        }
        had_preemption
    }

    pub(crate) fn drain_finished(&mut self) {
        let mut index = 0;
        while index < self.running.len() {
            if self.running[index].is_finished() {
                let sequence = self.running.remove(index);
                self.tokens_in_use = self.tokens_in_use.saturating_sub(sequence.total_len());
                self.finished.push(sequence);
            } else {
                index += 1;
            }
        }
    }

    pub(crate) fn sort_waiting_queue(&mut self) {
        match self.config.policy {
            SchedulePolicy::Fcfs => {
                if self
                    .waiting_queue
                    .iter()
                    .any(|sequence| sequence.priority.is_some())
                {
                    let queue = self.waiting_queue.make_contiguous();
                    queue.sort_by(|left, right| {
                        let left_priority = left.priority.unwrap_or(i64::MAX);
                        let right_priority = right.priority.unwrap_or(i64::MAX);
                        left_priority
                            .cmp(&right_priority)
                            .then(left.arrival_time.cmp(&right.arrival_time))
                    });
                }
            }
        }
    }

    pub(crate) fn estimate_running_future_tokens(&self) -> usize {
        self.running
            .iter()
            .map(|sequence| {
                // Cap per-request decode reservation so one oversized request
                // cannot starve admission for the whole batch.
                (sequence.remaining_tokens() as usize).min(self.config.decode_reservation_cap)
            })
            .sum()
    }
}
