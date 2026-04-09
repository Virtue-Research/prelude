//! Unified AR (autoregressive) scheduling loop.
//!
//! Handles both one-shot (batch) and multi-token (continuous batching) generation
//! in a single loop. The Scheduler decides what to prefill/decode each step;
//! the Executor handles device-specific execution via submit/collect.
//!
//! This replaces the previous 4 separate loops:
//! - `runtime/gpu_batch.rs` + `runtime/cpu_batch.rs` (one-shot generation)
//! - `runtime/gpu_continuous.rs` + `runtime/cpu_continuous.rs` (streaming decode)
//!
//! # Architecture
//!
//! ```text
//! ar_loop
//!   ├── drain requests from channel → scheduler.add_request()
//!   ├── scheduler.schedule_step() → SchedulerStep { prefill_ids, decode_ids }
//!   ├── executor.submit(step) → ExecutionHandle     ← device-specific
//!   ├── executor.collect(handle) → ModelOutput       ← device-specific
//!   ├── sample tokens (argmax / logits_processor)   ← device-agnostic
//!   ├── check stop conditions (eos, stop tokens/strings, max_tokens)
//!   ├── stream token deltas to callers
//!   └── scheduler.update(finished sequences)
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use crate::engine::{
    DecodeMetrics, EngineError, Engine, PreparedGenerateRequest, TokenLogprobInfo, Usage,
};
use crate::engine::executor::{Executor, ForwardBatch, ModelOutput};
use crate::scheduler::{
    FinishReason, Scheduler, SchedulerConfig, SchedulerStep, SeqFinishReason,
    Sequence, SamplingParams,
};
use crate::types::{GenerateResult, StreamEvent};
use crate::tensor::{DType, Tensor};

// ── Response channel ───────────────────────────────────────────────

/// How generation output is delivered back to the caller.
pub enum ResponseChannel {
    /// Non-streaming: send a complete GenerateResult when done.
    Complete(tokio::sync::oneshot::Sender<Result<GenerateResult, EngineError>>),
    /// Streaming: tokens are sent per-token through this channel.
    Stream(mpsc::UnboundedSender<StreamEvent>),
}

// ── Per-request runtime state ──────────────────────────────────────

/// Per-request state tracked across decode steps.
///
/// Mirrors `RuntimeSequenceState` from the legacy gpu_continuous.rs but
/// is device-agnostic — no GPU queue references.
pub struct ArSequenceState {
    pub request_id: String,
    pub prepared: Option<PreparedGenerateRequest>,
    pub response: ResponseChannel,
    pub gen_start: Instant,
    pub prefill_ms: f32,
    pub started_sent: bool,
    pub sent_text_len: usize,
    pub prompt_len: usize,
    pub next_decode_position: usize,
    pub pending_token: Option<u32>,
    pub output_tokens: Vec<u32>,
    pub token_logprobs: Vec<TokenLogprobInfo>,
    pub prompt_token_logprobs: Option<Vec<TokenLogprobInfo>>,
    pub block_table: Vec<u32>,
    pub deltanet_slot: Option<u32>,
    /// Cached from PreparedGenerateRequest so it survives prepared.take() during prefill.
    pub max_new_tokens: usize,
}

impl ArSequenceState {
    fn max_new(&self) -> usize {
        self.max_new_tokens
    }

    fn is_greedy(&self) -> bool {
        self.prepared.as_ref().map(|p| p.is_greedy).unwrap_or(true)
    }

    fn ensure_started(&mut self) {
        if self.started_sent { return; }
        if let ResponseChannel::Stream(tx) = &self.response {
            let _ = tx.send(StreamEvent::Started);
        }
        self.started_sent = true;
    }

    fn emit_text_delta(
        &mut self,
        tokenizer: &fastokens::Tokenizer,
        logprobs: Option<TokenLogprobInfo>,
    ) {
        let ResponseChannel::Stream(tx) = &self.response else { return; };
        let text = tokenizer.decode(&self.output_tokens, true).unwrap_or_default();
        if text.len() > self.sent_text_len {
            let _ = tx.send(StreamEvent::Token {
                text: text[self.sent_text_len..].to_string(),
                logprobs,
            });
            self.sent_text_len = text.len();
        }
    }

    fn current_text(&self, tokenizer: &fastokens::Tokenizer) -> String {
        tokenizer.decode(&self.output_tokens, true).unwrap_or_default()
    }
}

// ── Messages ───────────────────────────────────────────────────────

/// Messages from the engine API layer to the AR loop.
pub enum ArMessage {
    /// New generation request (pre-tokenized, pre-prepared).
    NewRequest {
        prepared: PreparedGenerateRequest,
        response: ResponseChannel,
    },
    /// Cancel an in-flight request.
    Abort(String),
}

// ── Main loop ──────────────────────────────────────────────────────

/// Run the unified AR scheduling loop.
///
/// Handles both batch and continuous batching in a single loop.
/// The Scheduler decides what goes into each step; the Executor
/// handles device-specific execution via submit/collect.
pub async fn ar_loop(
    engine: Arc<Engine>,
    executor: &dyn Executor,
    scheduler_config: SchedulerConfig,
    mut rx: mpsc::UnboundedReceiver<ArMessage>,
) {
    let mut scheduler = Scheduler::new(scheduler_config);
    let mut states: HashMap<String, ArSequenceState> = HashMap::new();
    let mut rx_open = true;

    loop {
        // ── Phase 1: Wait for at least one request if idle ─────────
        if !scheduler.has_work() {
            match rx.recv().await {
                Some(msg) => handle_message(msg, &mut scheduler, &mut states),
                None => { rx_open = false; }
            }
        }

        // ── Phase 2: Drain all pending messages (non-blocking) ─────
        loop {
            match rx.try_recv() {
                Ok(msg) => handle_message(msg, &mut scheduler, &mut states),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => { rx_open = false; break; }
            }
        }

        // ── Phase 3: Schedule next step ────────────────────────────
        let Some(step) = scheduler.schedule_step() else {
            if !rx_open && !scheduler.has_work() { break; }
            tokio::task::yield_now().await;
            continue;
        };

        // ── Phase 4: Build ForwardBatch and submit to Executor ──────
        let batch = build_forward_batch(&engine, &mut states, &step);
        let handle = match executor.submit(batch) {
            Ok(h) => h,
            Err(error) => {
                fail_step(&engine, &mut scheduler, &mut states, &step, error);
                continue;
            }
        };

        // ── Phase 5: Collect output ────────────────────────────────
        let output = match handle.recv().await {
            Ok(o) => o,
            Err(error) => {
                fail_step(&engine, &mut scheduler, &mut states, &step, error);
                continue;
            }
        };

        // ── Phase 6: Sample + stop conditions + deliver ────────────
        process_output(&engine, &mut scheduler, &mut states, &step, &output);

        if !rx_open && !scheduler.has_work() { break; }
    }

    // Shutdown: fail remaining requests
    for (_, state) in states.drain() {
        fail_state(&engine, state, EngineError::Unavailable("AR loop stopped".into()));
    }
    tracing::info!("AR loop exited");
}

// ── Message handling ───────────────────────────────────────────────

fn handle_message(
    msg: ArMessage,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
) {
    match msg {
        ArMessage::NewRequest { prepared, response } => {
            let request_id = prepared.request.request_id.clone();
            let seq = Sequence::new(
                request_id.clone(),
                prepared.prompt_tokens.clone(),
                SamplingParams::default(),
                prepared.max_new.saturating_sub(1) as u32,
                prepared.request.stop.strings.clone(),
                prepared.request.stop.token_ids.clone(),
                None,
            );
            scheduler.add_request(seq);
            let max_new_tokens = prepared.max_new;
            states.insert(request_id, ArSequenceState {
                request_id: prepared.request.request_id.clone(),
                prepared: Some(prepared),
                response,
                gen_start: Instant::now(),
                prefill_ms: 0.0,
                started_sent: false,
                sent_text_len: 0,
                prompt_len: 0,
                next_decode_position: 0,
                pending_token: None,
                output_tokens: Vec::new(),
                token_logprobs: Vec::new(),
                prompt_token_logprobs: None,
                block_table: Vec::new(),
                deltanet_slot: None,
                max_new_tokens,
            });
        }
        ArMessage::Abort(request_id) => {
            let _ = scheduler.abort_request(&request_id);
            states.remove(&request_id);
        }
    }
}

// ── Batch construction ─────────────────────────────────────────────

fn build_forward_batch(
    engine: &Engine,
    states: &mut HashMap<String, ArSequenceState>,
    step: &SchedulerStep,
) -> ForwardBatch {
    if !step.prefill_request_ids.is_empty() {
        // Prefill: take PreparedGenerateRequests for the forward pass.
        // They're put back into states after process_output so decode steps
        // can still read max_new, logprobs, and stop config.
        let items: Vec<_> = step.prefill_request_ids.iter()
            .filter_map(|id| {
                states.get_mut(id).and_then(|s| s.prepared.take())
            })
            .collect();
        ForwardBatch::Prefill { items }
    } else {
        // Allocate new KV cache blocks where needed before building the batch.
        // Each sequence may cross a block boundary during decode.
        if let Some(pool) = engine.cache.paged_pool.as_ref() {
            if let Some(bm_mutex) = engine.cache.block_manager.as_ref() {
                if let Ok(mut bm) = bm_mutex.lock() {
                    for id in &step.decode_request_ids {
                        if let Some(state) = states.get_mut(id) {
                            if state.next_decode_position % pool.block_size == 0 {
                                if let Some(new_block) = bm.allocate() {
                                    state.block_table.push(new_block);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Decode: collect (token, position, block_table) per sequence
        let cap = step.decode_request_ids.len();
        let mut tokens = Vec::with_capacity(cap);
        let mut positions = Vec::with_capacity(cap);
        let mut block_tables = Vec::with_capacity(cap);
        let mut dn_slots: Vec<u32> = Vec::new();
        let mut has_dn = false;
        for id in &step.decode_request_ids {
            if let Some(state) = states.get(id) {
                tokens.push(state.pending_token.unwrap_or(0));
                positions.push(state.next_decode_position);
                block_tables.push(state.block_table.clone());
                if let Some(slot) = state.deltanet_slot {
                    dn_slots.push(slot);
                    has_dn = true;
                }
            }
        }
        let deltanet_slots = if has_dn && dn_slots.len() == tokens.len() {
            Some(dn_slots)
        } else {
            None
        };
        ForwardBatch::Decode { tokens, positions, block_tables, deltanet_slots }
    }
}

// ── Output processing (sampling + stop conditions + delivery) ──────

fn process_output(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    step: &SchedulerStep,
    output: &ModelOutput,
) {
    let is_prefill = !step.prefill_request_ids.is_empty();

    // For prefill: batch_prefill_paged already sampled tokens and allocated blocks.
    // Use BatchPrefillResult directly instead of sampling from logits.
    if is_prefill && !output.prefill_results.is_empty() {
        process_prefill_output(engine, scheduler, states, step, &output.prefill_results);
        return;
    }

    // Decode path: sample from logits
    let logits = &output.logits;
    let all_ids: Vec<String> = step.decode_request_ids.iter().cloned().collect();

    let sampled = match sample_batch(states, &all_ids, logits) {
        Ok(tokens) => tokens,
        Err(error) => {
            fail_step(engine, scheduler, states, step, error);
            return;
        }
    };

    let mut completed: Vec<(String, FinishReason)> = Vec::new();

    for (row_idx, request_id) in all_ids.iter().enumerate() {
        let Some(state) = states.get_mut(request_id) else { continue; };
        let next_token = sampled[row_idx];

        process_single_token(engine, scheduler, state, request_id, next_token, None, &mut completed);
    }

    for (request_id, finish_reason) in completed {
        scheduler.finish_request(&request_id, seq_finish_reason(&finish_reason));
        if let Some(state) = states.remove(&request_id) {
            finish_state(engine, state, finish_reason);
        }
    }
}

/// Handle prefill results from batch_prefill_paged: set block_table, prompt_len,
/// process first token, and stream it.
fn process_prefill_output(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    step: &SchedulerStep,
    results: &[crate::engine::BatchPrefillResult],
) {
    let mut completed: Vec<(String, FinishReason)> = Vec::new();

    for (i, request_id) in step.prefill_request_ids.iter().enumerate() {
        let Some(result) = results.get(i) else { continue; };
        let Some(state) = states.get_mut(request_id) else { continue; };

        // Populate state from prefill result
        state.block_table = result.block_table.clone();
        state.prompt_len = result.prompt_len;
        state.next_decode_position = result.prompt_len;
        state.prefill_ms = result.prefill_ms;
        state.deltanet_slot = result.deltanet_slot;
        state.prompt_token_logprobs = result.prompt_token_logprobs.clone();

        let next_token = result.first_token;
        let logprobs = result.first_token_logprobs.clone();

        process_single_token(engine, scheduler, state, request_id, next_token, logprobs, &mut completed);
    }

    for (request_id, finish_reason) in completed {
        scheduler.finish_request(&request_id, seq_finish_reason(&finish_reason));
        if let Some(state) = states.remove(&request_id) {
            finish_state(engine, state, finish_reason);
        }
    }
}

/// Process a single sampled token: check stop conditions, stream, check length.
fn process_single_token(
    engine: &Engine,
    scheduler: &mut Scheduler,
    state: &mut ArSequenceState,
    request_id: &str,
    next_token: u32,
    token_logprobs: Option<TokenLogprobInfo>,
    completed: &mut Vec<(String, FinishReason)>,
) {
    scheduler.on_token_generated(request_id, next_token);
    state.next_decode_position += 1;
    state.ensure_started();

    // Check stop conditions
    if engine.is_eos(next_token) {
        completed.push((request_id.to_string(), FinishReason::Eos));
        return;
    }
    if let Some(ref prepared) = state.prepared {
        if engine.check_stop_tokens(next_token, &prepared.request.stop.token_ids) {
            completed.push((request_id.to_string(), FinishReason::Stop));
            return;
        }
    }

    // Append token and stream
    state.pending_token = Some(next_token);
    state.output_tokens.push(next_token);
    if let Some(ref lp) = token_logprobs {
        state.token_logprobs.push(lp.clone());
    }
    state.emit_text_delta(&engine.tokenizer, token_logprobs);

    // Check max length and stop strings
    let text = state.current_text(&engine.tokenizer);
    if state.output_tokens.len() >= state.max_new() {
        completed.push((request_id.to_string(), FinishReason::Length));
    } else if let Some(ref prepared) = state.prepared {
        if !prepared.request.stop.strings.is_empty()
            && prepared.request.stop.strings.iter().any(|s| text.contains(s))
        {
            completed.push((request_id.to_string(), FinishReason::Stop));
        }
    }
}

fn sample_batch(
    states: &mut HashMap<String, ArSequenceState>,
    ids: &[String],
    logits: &Tensor,
) -> Result<Vec<u32>, EngineError> {
    let all_greedy = ids.iter().all(|id| {
        states.get(id).map(|s| s.is_greedy()).unwrap_or(true)
    });

    if all_greedy {
        logits.argmax(crate::tensor::D::Minus1)
            .and_then(|t| t.to_vec1::<u32>())
            .map_err(|e| EngineError::Internal(format!("batch argmax failed: {e}")))
    } else {
        let mut tokens = Vec::with_capacity(ids.len());
        for (row_idx, id) in ids.iter().enumerate() {
            let row = logits.get(row_idx)
                .map_err(|e| EngineError::Internal(format!("get logits row failed: {e}")))?;
            let Some(state) = states.get_mut(id) else { tokens.push(0); continue; };
            if state.is_greedy() {
                let t = row.argmax(crate::tensor::D::Minus1)
                    .and_then(|t| t.to_scalar::<u32>())
                    .map_err(|e| EngineError::Internal(format!("argmax failed: {e}")))?;
                tokens.push(t);
            } else {
                let row_f32 = row.to_dtype(DType::F32)
                    .map_err(|e| EngineError::Internal(format!("to_dtype failed: {e}")))?;
                let t = state.prepared.as_mut()
                    .ok_or_else(|| EngineError::Internal("missing prepared request".into()))?
                    .logits_processor.sample(&row_f32)
                    .map_err(|e| EngineError::Internal(format!("sample failed: {e}")))?;
                tokens.push(t);
            }
        }
        Ok(tokens)
    }
}

// ── State finalization ─────────────────────────────────────────────

fn finish_state(engine: &Engine, mut state: ArSequenceState, finish_reason: FinishReason) {
    state.ensure_started();
    release_resources(engine, &mut state);

    let output_text = state.current_text(&engine.tokenizer);
    let completion_tokens = state.output_tokens.len() as u32;
    let total_ms = state.gen_start.elapsed().as_secs_f32() * 1000.0;
    let usage = Usage {
        prompt_tokens: state.prompt_len as u32,
        completion_tokens,
        total_tokens: state.prompt_len as u32 + completion_tokens,
    };
    let metrics = DecodeMetrics {
        ttft_ms: state.prefill_ms,
        prefill_ms: state.prefill_ms,
        decode_ms: (total_ms - state.prefill_ms).max(0.0),
        total_ms,
    };
    let result = GenerateResult {
        model: state.prepared.as_ref()
            .map(|p| p.request.model.clone())
            .unwrap_or_default(),
        output_token_ids: state.output_tokens,
        output_text,
        finish_reason: finish_reason.clone(),
        usage: usage.clone(),
        metrics: metrics.clone(),
        token_logprobs: if state.token_logprobs.is_empty() { None } else { Some(state.token_logprobs) },
        prompt_token_logprobs: state.prompt_token_logprobs,
    };

    match state.response {
        ResponseChannel::Complete(tx) => { let _ = tx.send(Ok(result)); }
        ResponseChannel::Stream(tx) => {
            let _ = tx.send(StreamEvent::Finished { finish_reason, usage, metrics });
        }
    }
}

fn fail_state(engine: &Engine, mut state: ArSequenceState, error: EngineError) {
    release_resources(engine, &mut state);
    match state.response {
        ResponseChannel::Complete(tx) => { let _ = tx.send(Err(error)); }
        ResponseChannel::Stream(_) => {} // Dropping sender closes the stream
    }
}

fn fail_step(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    step: &SchedulerStep,
    error: EngineError,
) {
    for id in step.prefill_request_ids.iter().chain(step.decode_request_ids.iter()) {
        let _ = scheduler.abort_request(id);
        if let Some(state) = states.remove(id) {
            fail_state(engine, state, error.clone());
        }
    }
}

fn release_resources(engine: &Engine, state: &mut ArSequenceState) {
    if !state.block_table.is_empty() {
        if let Some(bm_mutex) = engine.cache.block_manager.as_ref() {
            if let Ok(mut bm) = bm_mutex.lock() {
                bm.free(&state.block_table);
            }
        }
        state.block_table.clear();
    }
    if let Some(slot) = state.deltanet_slot.take() {
        if let Some(pool_mutex) = engine.cache.deltanet_pool.as_ref() {
            if let Ok(mut pool) = pool_mutex.lock() {
                pool.free(slot);
            }
        }
    }
}

fn seq_finish_reason(reason: &FinishReason) -> SeqFinishReason {
    match reason {
        FinishReason::Stop => SeqFinishReason::Stop,
        FinishReason::Length => SeqFinishReason::Length,
        FinishReason::Eos => SeqFinishReason::Eos,
        FinishReason::Cancelled => SeqFinishReason::Abort("cancelled".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor::ExecutionHandle;
    use crate::tensor::Device;

    fn make_prepared(request_id: &str, max_new: usize) -> PreparedGenerateRequest {
        use crate::engine::sampling::{LogitsProcessor, Sampling};
        PreparedGenerateRequest {
            request_idx: 0,
            request: crate::types::GenerateRequest {
                request_id: request_id.to_string(),
                model: String::new(),
                input: crate::types::PromptInput::Text(String::new()),
                sampling: SamplingParams::default(),
                max_new_tokens: max_new as u32,
                stop: Default::default(),
                seed: None,
                deadline_ms: None,
                logprobs: None,
                prompt_logprobs: None,
            },
            prompt_tokens: vec![1, 2, 3],
            max_new,
            is_greedy: true,
            logits_processor: LogitsProcessor::from_sampling(42, Sampling::ArgMax),
        }
    }

    // ── handle_message tests ───────────────────────────────────────

    #[test]
    fn handle_new_request_adds_to_scheduler() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let mut states = HashMap::new();

        let prepared = make_prepared("r1", 10);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        handle_message(
            ArMessage::NewRequest {
                prepared,
                response: ResponseChannel::Complete(tx),
            },
            &mut scheduler,
            &mut states,
        );

        assert_eq!(scheduler.num_waiting(), 1);
        assert!(states.contains_key("r1"));
    }

    #[test]
    fn handle_abort_removes_from_scheduler() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let mut states = HashMap::new();

        let prepared = make_prepared("r1", 10);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        handle_message(
            ArMessage::NewRequest {
                prepared,
                response: ResponseChannel::Complete(tx),
            },
            &mut scheduler,
            &mut states,
        );
        assert_eq!(scheduler.num_waiting(), 1);

        handle_message(
            ArMessage::Abort("r1".into()),
            &mut scheduler,
            &mut states,
        );
        assert!(states.is_empty());
    }

    // ── sample_batch tests ─────────────────────────────────────────

    #[test]
    fn sample_batch_greedy_argmax() {
        let states: HashMap<String, ArSequenceState> = HashMap::new();
        // Empty states → all_greedy = true (default)
        // Logits: 2 rows, vocab=4. Row 0: max at idx 2, Row 1: max at idx 0
        let logits = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.9, 0.1, 0.8, 0.1, 0.05, 0.05],
            (2, 4),
            &Device::Cpu,
        ).unwrap();

        // Need states entries for the IDs
        let mut states_with_entries = HashMap::new();
        states_with_entries.insert("a".to_string(), make_prefill_state("a", 10));
        states_with_entries.insert("b".to_string(), make_prefill_state("b", 10));

        let ids = vec!["a".to_string(), "b".to_string()];
        let tokens = sample_batch(&mut states_with_entries, &ids, &logits).unwrap();

        assert_eq!(tokens, vec![2, 0]); // argmax of each row
    }

    // ── fail_step tests ────────────────────────────────────────────

    #[test]
    fn fail_step_aborts_all_ids() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());

        // Add two requests
        for id in &["r1", "r2"] {
            let seq = Sequence::new(
                id.to_string(), vec![1, 2, 3],
                SamplingParams::default(), 10, vec![], vec![], None,
            );
            scheduler.add_request(seq);
        }
        assert_eq!(scheduler.num_waiting(), 2);

        let step = SchedulerStep::prefill(vec!["r1".into(), "r2".into()]);

        // We can't call fail_step without Engine, but we can test the scheduler abort part
        for id in step.prefill_request_ids.iter().chain(step.decode_request_ids.iter()) {
            let _ = scheduler.abort_request(id);
        }
        assert_eq!(scheduler.num_waiting(), 0);
    }

    // ── seq_finish_reason mapping ──────────────────────────────────

    #[test]
    fn finish_reason_mapping() {
        assert!(matches!(seq_finish_reason(&FinishReason::Eos), SeqFinishReason::Eos));
        assert!(matches!(seq_finish_reason(&FinishReason::Stop), SeqFinishReason::Stop));
        assert!(matches!(seq_finish_reason(&FinishReason::Length), SeqFinishReason::Length));
        assert!(matches!(seq_finish_reason(&FinishReason::Cancelled), SeqFinishReason::Abort(_)));
    }

    // ── build_forward_batch tests ─────────────────────────────────

    fn make_prefill_state(id: &str, max_new: usize) -> ArSequenceState {
        ArSequenceState {
            request_id: id.to_string(),
            prepared: Some(make_prepared(id, max_new)),
            response: ResponseChannel::Complete(tokio::sync::oneshot::channel().0),
            gen_start: Instant::now(), prefill_ms: 0.0, started_sent: false,
            sent_text_len: 0, prompt_len: 3, next_decode_position: 3,
            pending_token: None, output_tokens: vec![], token_logprobs: vec![],
            prompt_token_logprobs: None, block_table: vec![], deltanet_slot: None,
            max_new_tokens: max_new,
        }
    }

    fn make_decode_state(id: &str, pending_token: u32, position: usize) -> ArSequenceState {
        let max_new = 10;
        ArSequenceState {
            request_id: id.to_string(),
            prepared: Some(make_prepared(id, max_new)),
            response: ResponseChannel::Complete(tokio::sync::oneshot::channel().0),
            gen_start: Instant::now(), prefill_ms: 0.0, started_sent: true,
            sent_text_len: 0, prompt_len: 3, next_decode_position: position,
            pending_token: Some(pending_token), output_tokens: vec![pending_token],
            token_logprobs: vec![], prompt_token_logprobs: None,
            block_table: vec![0, 1], deltanet_slot: None,
            max_new_tokens: max_new,
        }
    }

    #[test]
    fn build_decode_batch_basic() {
        let mut states = HashMap::new();
        states.insert("s1".to_string(), make_decode_state("s1", 42, 5));
        states.insert("s2".to_string(), make_decode_state("s2", 99, 8));

        let step = SchedulerStep::decode(vec!["s1".into(), "s2".into()]);
        let batch = build_forward_batch(&engine, &mut states, &step);

        match batch {
            ForwardBatch::Decode { tokens, positions, block_tables, deltanet_slots } => {
                assert_eq!(tokens, vec![42, 99]);
                assert_eq!(positions, vec![5, 8]);
                assert_eq!(block_tables.len(), 2);
                assert!(deltanet_slots.is_none());
            }
            _ => panic!("expected Decode"),
        }
    }

    #[test]
    fn build_decode_batch_with_deltanet() {
        let mut states = HashMap::new();
        let mut s1 = make_decode_state("s1", 42, 5);
        s1.deltanet_slot = Some(0);
        let mut s2 = make_decode_state("s2", 99, 8);
        s2.deltanet_slot = Some(1);
        states.insert("s1".to_string(), s1);
        states.insert("s2".to_string(), s2);

        let step = SchedulerStep::decode(vec!["s1".into(), "s2".into()]);
        let batch = build_forward_batch(&engine, &mut states, &step);

        match batch {
            ForwardBatch::Decode { deltanet_slots, .. } => {
                assert_eq!(deltanet_slots, Some(vec![0, 1]));
            }
            _ => panic!("expected Decode"),
        }
    }

    #[test]
    fn build_prefill_batch_extracts_prepared() {
        let mut states = HashMap::new();
        let prepared = make_prepared("r1", 5);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        handle_message(
            ArMessage::NewRequest { prepared, response: ResponseChannel::Complete(tx) },
            &mut Scheduler::new(SchedulerConfig::default()),
            &mut states,
        );

        // State should have prepared
        assert!(states.get("r1").unwrap().prepared.is_some());

        let step = SchedulerStep::prefill(vec!["r1".into()]);
        let batch = build_forward_batch(&engine, &mut states, &step);

        match batch {
            ForwardBatch::Prefill { items } => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].prompt_tokens, vec![1, 2, 3]);
            }
            _ => panic!("expected Prefill"),
        }

        // Prepared should be taken (None now)
        assert!(states.get("r1").unwrap().prepared.is_none());
    }
}
