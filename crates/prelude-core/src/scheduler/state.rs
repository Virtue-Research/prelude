use std::collections::VecDeque;
use std::time::Instant;

/// Sampling parameters for text generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    /// Adaptive nucleus: drop tokens with p < min_p * p_max.
    pub min_p: Option<f32>,
    /// Divide/multiply logits of previously seen tokens.
    pub repetition_penalty: Option<f32>,
    /// Subtract a fixed value for each seen token.
    pub presence_penalty: Option<f32>,
    /// Subtract count * penalty for each seen token.
    pub frequency_penalty: Option<f32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 1.0,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
        }
    }
}

/// Why a generation request finished.
#[derive(Debug, Clone)]
pub enum FinishReason {
    Stop,
    Length,
    Eos,
    Cancelled,
}

impl FinishReason {
    pub fn as_openai_str(&self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Eos => "stop",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Lifecycle of a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

/// Reason a sequence finished.
#[derive(Debug, Clone)]
pub enum SeqFinishReason {
    Stop,
    Length,
    Eos,
    Abort(String),
}

impl From<&SeqFinishReason> for FinishReason {
    fn from(reason: &SeqFinishReason) -> Self {
        match reason {
            SeqFinishReason::Stop => FinishReason::Stop,
            SeqFinishReason::Length => FinishReason::Length,
            SeqFinishReason::Eos => FinishReason::Eos,
            SeqFinishReason::Abort(_) => FinishReason::Cancelled,
        }
    }
}

/// A single in-flight request tracked by the scheduler.
#[derive(Debug)]
pub struct Sequence {
    pub request_id: String,
    pub status: SequenceStatus,
    pub input_ids: Vec<u32>,
    pub output_ids: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub max_new_tokens: u32,
    pub stop_strings: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub finish_reason: Option<SeqFinishReason>,
    pub arrival_time: Instant,
    pub priority: Option<i64>,
    pub kv_computed_len: usize,
    /// Real GPU block IDs (authoritative source, managed by scheduler).
    pub block_table: Vec<u32>,
    pub deltanet_slot: Option<u32>,
    pub preempt_count: u32,
}

impl Sequence {
    pub fn new(
        request_id: String,
        input_ids: Vec<u32>,
        sampling_params: SamplingParams,
        max_new_tokens: u32,
        stop_strings: Vec<String>,
        stop_token_ids: Vec<u32>,
        priority: Option<i64>,
    ) -> Self {
        Self {
            request_id,
            status: SequenceStatus::Waiting,
            input_ids,
            output_ids: Vec::new(),
            sampling_params,
            max_new_tokens,
            stop_strings,
            stop_token_ids,
            finish_reason: None,
            arrival_time: Instant::now(),
            priority,
            kv_computed_len: 0,
            block_table: Vec::new(),
            deltanet_slot: None,
            preempt_count: 0,
        }
    }

    #[inline]
    pub fn total_len(&self) -> usize {
        self.input_ids.len() + self.output_ids.len()
    }

    #[inline]
    pub fn remaining_tokens(&self) -> u32 {
        self.max_new_tokens
            .saturating_sub(self.output_ids.len() as u32)
    }

    #[inline]
    pub fn prefill_len(&self) -> usize {
        self.input_ids.len().saturating_sub(self.kv_computed_len)
    }

    #[inline]
    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }
}

/// The forward mode for a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardMode {
    Prefill,
    Decode,
    Mixed,
}

/// Output of one scheduling step for executors that index into an external store.
#[derive(Debug)]
pub struct SchedulerOutput {
    pub sequences: Vec<usize>,
    pub forward_mode: ForwardMode,
}

/// How to order the waiting queue before selecting requests for prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulePolicy {
    Fcfs,
}

/// Configuration knobs for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_batch_wait_ms: u64,
    pub max_running_requests: usize,
    /// Per-step total token budget (prefill + decode combined).
    /// This is the primary knob controlling chunked prefill granularity.
    pub max_num_batched_tokens: usize,
    /// Per-request prefill token cap (like vLLM's long_prefill_token_threshold).
    /// 0 means no per-request cap (only limited by max_num_batched_tokens).
    pub long_prefill_token_threshold: usize,
    pub max_total_tokens: usize,
    pub decode_reservation_cap: usize,
    pub new_token_ratio: f32,
    pub min_new_token_ratio: f32,
    pub new_token_ratio_decay: f32,
    pub policy: SchedulePolicy,
    pub chunked_prefill: bool,
    /// KV cache block size (tokens per block). 0 = no block-level tracking.
    pub block_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_batch_wait_ms: 5,
            max_running_requests: 256,
            max_num_batched_tokens: 8192,
            long_prefill_token_threshold: 0,
            max_total_tokens: 32768,
            decode_reservation_cap: 4096,
            new_token_ratio: 0.4,
            min_new_token_ratio: 0.1,
            new_token_ratio_decay: 0.002,
            policy: SchedulePolicy::Fcfs,
            chunked_prefill: false,
            block_size: 16,
        }
    }
}

/// What the scheduler decided for this iteration.
#[derive(Debug)]
pub struct SchedulerStep {
    pub prefill_request_ids: Vec<String>,
    /// How many tokens to process for each prefill request (parallel to prefill_request_ids).
    pub prefill_chunk_lens: Vec<usize>,
    pub decode_request_ids: Vec<String>,
    pub forward_mode: ForwardMode,
}

impl SchedulerStep {
    pub fn prefill(prefill_request_ids: Vec<String>, prefill_chunk_lens: Vec<usize>) -> Self {
        Self {
            prefill_request_ids,
            prefill_chunk_lens,
            decode_request_ids: Vec::new(),
            forward_mode: ForwardMode::Prefill,
        }
    }

    pub fn decode(decode_request_ids: Vec<String>) -> Self {
        Self {
            prefill_request_ids: Vec::new(),
            prefill_chunk_lens: Vec::new(),
            decode_request_ids,
            forward_mode: ForwardMode::Decode,
        }
    }

    pub fn mixed(
        prefill_request_ids: Vec<String>,
        prefill_chunk_lens: Vec<usize>,
        decode_request_ids: Vec<String>,
    ) -> Self {
        Self {
            prefill_request_ids,
            prefill_chunk_lens,
            decode_request_ids,
            forward_mode: ForwardMode::Mixed,
        }
    }
}

/// A minimal continuous-batching scheduler.
pub struct Scheduler {
    pub(crate) config: SchedulerConfig,
    pub(crate) waiting_queue: VecDeque<Sequence>,
    pub(crate) running: Vec<Sequence>,
    pub(crate) finished: Vec<Sequence>,
    pub(crate) effective_new_token_ratio: f32,
    pub(crate) tokens_in_use: usize,
    pub(crate) step_count: u64,
    /// Available KV cache blocks for admission checks.
    /// Synced from block_manager at the start of each scheduling step.
    pub(crate) available_blocks: usize,
    /// Shared block manager (primary owner; CacheManager holds clone for prefix ops).
    block_manager: Option<std::sync::Arc<std::sync::Mutex<super::BlockManager>>>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let ratio = config.new_token_ratio;
        Self {
            config,
            waiting_queue: VecDeque::new(),
            running: Vec::new(),
            finished: Vec::new(),
            effective_new_token_ratio: ratio,
            tokens_in_use: 0,
            step_count: 0,
            available_blocks: usize::MAX,
            block_manager: None,
        }
    }

    /// Attach a shared block manager (called once during ScheduledEngine init).
    pub fn set_block_manager(
        &mut self,
        bm: std::sync::Arc<std::sync::Mutex<super::BlockManager>>,
    ) {
        self.block_manager = Some(bm);
    }

    /// Sync available_blocks from the block manager before scheduling.
    pub fn sync_available_blocks(&mut self) {
        if let Some(ref bm) = self.block_manager {
            if let Ok(bm) = bm.lock() {
                self.available_blocks = bm.available();
            }
        }
    }

    /// Allocate a single KV cache block. Returns None if exhausted.
    pub fn allocate_block(&self) -> Option<u32> {
        self.block_manager.as_ref()?.lock().ok()?.allocate()
    }

    /// Allocate blocks for `num_tokens` tokens. Returns None if not enough blocks.
    pub fn allocate_blocks_for_tokens(&self, num_tokens: usize) -> Option<Vec<u32>> {
        self.block_manager.as_ref()?.lock().ok()?.allocate_for_tokens(num_tokens)
    }

    /// Free blocks back to the pool.
    pub fn free_blocks(&self, block_table: &[u32]) {
        if let Some(ref bm) = self.block_manager {
            if let Ok(mut bm) = bm.lock() {
                bm.free(block_table);
            }
        }
    }

    pub fn add_request(&mut self, seq: Sequence) {
        self.waiting_queue.push_back(seq);
    }

    pub fn abort_request(&mut self, request_id: &str) -> Option<Sequence> {
        if let Some(pos) = self
            .waiting_queue
            .iter()
            .position(|sequence| sequence.request_id == request_id)
        {
            let mut sequence = self
                .waiting_queue
                .remove(pos)
                .expect("queue index was valid");
            sequence.status = SequenceStatus::Finished;
            sequence.finish_reason = Some(SeqFinishReason::Abort("aborted by user".into()));
            return Some(sequence);
        }

        if let Some(pos) = self
            .running
            .iter()
            .position(|sequence| sequence.request_id == request_id)
        {
            let mut sequence = self.running.remove(pos);
            self.tokens_in_use = self.tokens_in_use.saturating_sub(sequence.total_len());
            sequence.status = SequenceStatus::Finished;
            sequence.finish_reason = Some(SeqFinishReason::Abort("aborted by user".into()));
            return Some(sequence);
        }

        None
    }

    pub fn schedule_step(&mut self) -> Option<SchedulerStep> {
        self.step_count += 1;
        self.sync_available_blocks();
        self.drain_finished();

        if self.config.chunked_prefill {
            return self.get_mixed_batch();
        }

        if let Some(prefill) = self.get_new_prefill_batch() {
            return Some(prefill);
        }

        self.get_decode_batch()
    }

    pub fn on_token_generated(&mut self, request_id: &str, token_id: u32) {
        if let Some(sequence) = self
            .running
            .iter_mut()
            .find(|sequence| sequence.request_id == request_id)
        {
            sequence.output_ids.push(token_id);
            self.tokens_in_use += 1;
        }
    }

    pub fn finish_request(&mut self, request_id: &str, reason: SeqFinishReason) {
        if let Some(sequence) = self
            .running
            .iter_mut()
            .find(|sequence| sequence.request_id == request_id)
        {
            sequence.status = SequenceStatus::Finished;
            sequence.finish_reason = Some(reason);
        }
    }

    pub fn rollback_prefill(&mut self, request_ids: &[String]) {
        let deferred: std::collections::HashSet<&str> =
            request_ids.iter().map(String::as_str).collect();
        let mut kept = Vec::with_capacity(self.running.len());
        let mut returned = Vec::new();

        for mut sequence in self.running.drain(..) {
            if deferred.contains(sequence.request_id.as_str()) {
                self.tokens_in_use = self.tokens_in_use.saturating_sub(sequence.input_ids.len());
                if !sequence.block_table.is_empty() {
                    if let Some(ref bm) = self.block_manager {
                        if let Ok(mut bm) = bm.lock() {
                            bm.free(&sequence.block_table);
                        }
                    }
                }
                sequence.status = SequenceStatus::Waiting;
                sequence.kv_computed_len = 0;
                sequence.block_table.clear();
                sequence.deltanet_slot = None;
                returned.push(sequence);
            } else {
                kept.push(sequence);
            }
        }

        self.running = kept;
        for sequence in returned.into_iter().rev() {
            self.waiting_queue.push_front(sequence);
        }
    }

    pub fn take_finished(&mut self) -> Vec<Sequence> {
        std::mem::take(&mut self.finished)
    }

    #[inline]
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    #[inline]
    pub fn num_waiting(&self) -> usize {
        self.waiting_queue.len()
    }

    #[inline]
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    #[inline]
    pub fn has_work(&self) -> bool {
        !self.waiting_queue.is_empty() || !self.running.is_empty()
    }

    /// Get an immutable reference to a running sequence by ID.
    pub fn get_sequence(&self, request_id: &str) -> Option<&Sequence> {
        self.running.iter().find(|s| s.request_id == request_id)
    }

    /// Get a mutable reference to a running sequence by ID.
    pub fn get_sequence_mut(&mut self, request_id: &str) -> Option<&mut Sequence> {
        self.running.iter_mut().find(|s| s.request_id == request_id)
    }
}
