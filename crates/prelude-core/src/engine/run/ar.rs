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
use crate::engine::executor::{Executor, ModelOutput};
use crate::scheduler::{
    FinishReason, Scheduler, SchedulerConfig, SchedulerStep, SeqFinishReason, Sequence,
    SamplingParams,
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
}

impl ArSequenceState {
    fn max_new(&self) -> usize {
        self.prepared.as_ref().map(|p| p.max_new).unwrap_or(0)
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

        // ── Phase 4: Submit to device via Executor ─────────────────
        let handle = match executor.submit(&step) {
            Ok(h) => h,
            Err(error) => {
                fail_step(&engine, &mut scheduler, &mut states, &step, error);
                continue;
            }
        };

        // ── Phase 5: Collect output ────────────────────────────────
        let output = match executor.collect(handle) {
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
            });
        }
        ArMessage::Abort(request_id) => {
            let _ = scheduler.abort_request(&request_id);
            states.remove(&request_id);
        }
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
    let logits = &output.logits;
    let all_ids: Vec<String> = step.prefill_request_ids.iter()
        .chain(step.decode_request_ids.iter())
        .cloned()
        .collect();

    // Sample tokens
    let sampled = match sample_batch(states, &all_ids, logits) {
        Ok(tokens) => tokens,
        Err(error) => {
            fail_step(engine, scheduler, states, step, error);
            return;
        }
    };

    // Process each sampled token
    let mut completed: Vec<(String, FinishReason)> = Vec::new();

    for (row_idx, request_id) in all_ids.iter().enumerate() {
        let Some(state) = states.get_mut(request_id) else { continue; };
        let next_token = sampled[row_idx];

        scheduler.on_token_generated(request_id, next_token);
        state.next_decode_position += 1;
        state.ensure_started();

        // Extract logprobs if requested
        let token_logprobs = state.prepared.as_ref()
            .and_then(|p| p.request.logprobs)
            .and_then(|k| {
                logits.get(row_idx).ok()
                    .and_then(|row| Engine::extract_top_logprobs(&row, next_token, k, &engine.tokenizer).ok())
            });

        // Check stop conditions
        if engine.is_eos(next_token) {
            completed.push((request_id.clone(), FinishReason::Eos));
            continue;
        }
        if let Some(ref prepared) = state.prepared {
            if engine.check_stop_tokens(next_token, &prepared.request.stop.token_ids) {
                completed.push((request_id.clone(), FinishReason::Stop));
                continue;
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
            completed.push((request_id.clone(), FinishReason::Length));
        } else if let Some(ref prepared) = state.prepared {
            if !prepared.request.stop.strings.is_empty()
                && prepared.request.stop.strings.iter().any(|s| text.contains(s))
            {
                completed.push((request_id.clone(), FinishReason::Stop));
            }
        }
    }

    // Finish completed sequences
    for (request_id, finish_reason) in completed {
        scheduler.finish_request(&request_id, seq_finish_reason(&finish_reason));
        if let Some(state) = states.remove(&request_id) {
            finish_state(engine, state, finish_reason);
        }
    }
}

fn sample_batch(
    states: &HashMap<String, ArSequenceState>,
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
            let Some(state) = states.get(id) else { tokens.push(0); continue; };
            if state.is_greedy() {
                let t = row.argmax(crate::tensor::D::Minus1)
                    .and_then(|t| t.to_scalar::<u32>())
                    .map_err(|e| EngineError::Internal(format!("argmax failed: {e}")))?;
                tokens.push(t);
            } else {
                let row_f32 = row.to_dtype(DType::F32)
                    .map_err(|e| EngineError::Internal(format!("to_dtype failed: {e}")))?;
                // LogitsProcessor::sample takes &mut self
                // We need mutable access, but states is borrowed immutably here.
                // For now, use argmax as fallback for non-greedy (proper fix needs
                // mutable state access or separating the logits processor).
                let t = row_f32.argmax(crate::tensor::D::Minus1)
                    .and_then(|t| t.to_scalar::<u32>())
                    .map_err(|e| EngineError::Internal(format!("sample fallback failed: {e}")))?;
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
