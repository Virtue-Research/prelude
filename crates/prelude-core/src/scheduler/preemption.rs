use super::{SchedulePolicy, Scheduler, Sequence, SequenceStatus};

impl Scheduler {
    /// Evict the running request at `idx`: free its KV blocks, re-credit
    /// `tokens_in_use`, and reset it to a clean Waiting state. Shared by the
    /// token-budget preemption path and the block-progress watchdog.
    fn evict_running_at(&mut self, idx: usize) -> (usize, Sequence) {
        let mut victim = self.running.remove(idx);
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

        (freed, victim)
    }

    /// Token-budget preemption victim: the newest by arrival_time. Used by
    /// `ensure_decode_capacity_tracked` — it must be able to recycle even the
    /// sole running request to admit waiting work under token pressure, so it
    /// deliberately applies no oldest/leader protection (that protection is
    /// only for the block-exhaustion wedge path, `preempt_for_progress`).
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
        Some(self.evict_running_at(victim_idx))
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

    /// Force forward progress when a scheduling step produced no runnable
    /// work because the KV block pool is exhausted (not token-budget — that
    /// is handled by `ensure_decode_capacity_tracked`). Preempts one
    /// non-protected running request (never the oldest, never a prefix
    /// leader) and re-queues it to the waiting front, freeing its blocks so
    /// the protected oldest can make progress. Returns true if it preempted.
    pub(crate) fn preempt_for_progress(&mut self) -> bool {
        // Need ≥2 running to safely shed one: never preempt the last/only
        // request, never the oldest by arrival_time (it is closest to
        // completion — recycling it destroys the forward-progress
        // guarantee), and never a current prefix leader (parked same-prefix
        // peers depend on it to populate the shared prefix). Among the
        // remaining candidates prefer the one preempted most often (spreads
        // the cost and converges), tie-breaking on newest.
        if self.running.len() <= 1 {
            return false;
        }
        let oldest_idx = self
            .running
            .iter()
            .enumerate()
            .min_by_key(|(_, sequence)| sequence.arrival_time)
            .map(|(index, _)| index);
        let victim_idx = self
            .running
            .iter()
            .enumerate()
            .filter(|(idx, sequence)| {
                Some(*idx) != oldest_idx
                    && !super::admission::prefix_prefill_leader_needed(sequence)
            })
            .max_by_key(|(_, sequence)| (sequence.preempt_count, sequence.arrival_time))
            .map(|(index, _)| index);
        match victim_idx {
            Some(idx) => {
                let (_, victim) = self.evict_running_at(idx);
                self.waiting_queue.push_front(victim);
                true
            }
            None => false,
        }
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
