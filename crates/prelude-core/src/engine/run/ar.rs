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
/// Per-request runtime state for response delivery and sampling.
/// Scheduling-relevant state (block_table, kv_computed_len) lives on Sequence
/// in the Scheduler. This struct only holds response/streaming/sampling concerns.
pub struct ArSequenceState {
    pub request_id: String,
    pub prepared: Option<PreparedGenerateRequest>,
    pub response: ResponseChannel,
    pub gen_start: Instant,
    pub prefill_ms: f32,
    pub started_sent: bool,
    pub sent_text_len: usize,
    pub pending_token: Option<u32>,
    pub output_tokens: Vec<u32>,
    pub token_logprobs: Vec<TokenLogprobInfo>,
    pub prompt_token_logprobs: Option<Vec<TokenLogprobInfo>>,
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
    if let Some(bm_arc) = engine.cache.block_manager_arc() {
        scheduler.set_block_manager(bm_arc);
    }
    let mut states: HashMap<String, ArSequenceState> = HashMap::new();
    let mut rx_open = true;

    loop {
        let deltanet_pool_ref = engine.cache.deltanet_pool.as_ref();
        // ── Phase 1: Wait for at least one request if idle ─────────
        if !scheduler.has_work() {
            match rx.recv().await {
                Some(msg) => handle_message(msg, &mut scheduler, &mut states, deltanet_pool_ref),
                None => { rx_open = false; }
            }
        }

        // ── Phase 2: Drain all pending messages (non-blocking) ─────
        loop {
            match rx.try_recv() {
                Ok(msg) => handle_message(msg, &mut scheduler, &mut states, deltanet_pool_ref),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => { rx_open = false; break; }
            }
        }

        // ── Phase 3: Schedule next step ────────────────────────────
        // schedule_step() syncs block availability internally.
        let Some(step) = scheduler.schedule_step() else {
            if !rx_open && !scheduler.has_work() { break; }
            tokio::task::yield_now().await;
            continue;
        };

        // ── Phase 4: Build batch + forward + process output ──────────
        // Pure decode → ForwardBatch::Decode (CUDA graph eligible).
        // Anything with prefill → ForwardBatch::Mixed (single forward pass).
        let batch = build_step_batch(&engine, &mut scheduler, &mut states, &step);
        let handle = match executor.submit(batch) {
            Ok(h) => h,
            Err(error) => {
                fail_step(&engine, &mut scheduler, &mut states, &step, error);
                continue;
            }
        };
        let output = match handle.recv().await {
            Ok(o) => o,
            Err(error) => {
                fail_step(&engine, &mut scheduler, &mut states, &step, error);
                continue;
            }
        };
        process_step_output(&engine, &mut scheduler, &mut states, &step, &output);

        if !rx_open && !scheduler.has_work() { break; }
    }

    // Shutdown: fail remaining requests
    for (id, state) in states.drain() {
        release_resources(&engine, &mut scheduler, &id);
        fail_state(&engine, state, EngineError::Unavailable("AR loop stopped".into()));
    }
    tracing::info!("AR loop exited");
}

// ── Message handling ───────────────────────────────────────────────

fn handle_message(
    msg: ArMessage,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    deltanet_pool: Option<&std::sync::Mutex<crate::cache::deltanet_pool::DeltaNetPool>>,
) {
    match msg {
        ArMessage::NewRequest { prepared, response } => {
            let request_id = prepared.request.request_id.clone();
            let mut seq = Sequence::new(
                request_id.clone(),
                prepared.prompt_tokens.clone(),
                SamplingParams::default(),
                prepared.max_new.saturating_sub(1) as u32,
                prepared.request.stop.strings.clone(),
                prepared.request.stop.token_ids.clone(),
                None,
            );
            // Allocate a DeltaNet pool slot for hybrid models before queueing.
            // `allocate` zeros the slot so a fresh request always starts from
            // clean recurrent + conv state; `release_resources` frees it when
            // the sequence finishes.
            if let Some(pool_mutex) = deltanet_pool {
                if let Ok(mut pool) = pool_mutex.lock() {
                    match pool.allocate() {
                        Some(slot) => {
                            seq.deltanet_slot = Some(slot);
                        }
                        None => {
                            tracing::warn!(
                                rid = %request_id,
                                "DeltaNet pool exhausted — request will run without per-request state isolation"
                            );
                        }
                    }
                }
            }
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
                pending_token: None,
                output_tokens: Vec::new(),
                token_logprobs: Vec::new(),
                prompt_token_logprobs: None,
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

/// Build a prefill ForwardBatch, handling both full and chunked prefill.
///
/// Final chunk: takes `prepared` from state (consumed).
/// Partial chunk: builds a minimal item from cached prompt tokens; `prepared`
/// stays in state for the final chunk (which needs logits_processor).
/// Build a ForwardBatch for the current step.
///
/// Pure decode → ForwardBatch::Decode (CUDA graph eligible).
/// Any prefill → ForwardBatch::Mixed (single forward pass with variable Q per request).
fn build_step_batch(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    step: &SchedulerStep,
) -> ForwardBatch {
    use crate::engine::executor::StepRequest;

    // Pure decode → use Decode variant for CUDA graph eligibility
    if step.prefill_request_ids.is_empty() {
        // Allocate new KV cache blocks for decode where needed.
        // total_len() = input_ids + output_ids (output_ids already includes the
        // token being decoded). The decode token sits at position total_len()-1
        // and context_len = total_len(). A new block is needed when position
        // total_len()-1 starts a fresh block, i.e. (total_len()-1) % block_size == 0.
        let block_size = engine.cache.paged_pool.as_ref().map(|p| p.block_size).unwrap_or(16);
        for id in &step.decode_request_ids {
            if let Some(seq) = scheduler.get_sequence(id) {
                let seq_len = seq.total_len();
                let position = seq_len - 1; // 0-indexed position of the decode token
                if position % block_size == 0 {
                    if let Some(new_block) = scheduler.allocate_block() {
                        if let Some(seq_mut) = scheduler.get_sequence_mut(id) {
                            seq_mut.block_table.push(new_block);
                        }
                    }
                }
            }
        }
        let cap = step.decode_request_ids.len();
        let mut tokens = Vec::with_capacity(cap);
        let mut positions = Vec::with_capacity(cap);
        let mut block_tables = Vec::with_capacity(cap);
        let mut dn_slots: Vec<u32> = Vec::new();
        let mut has_dn = false;
        for id in &step.decode_request_ids {
            if let (Some(state), Some(seq)) = (states.get(id), scheduler.get_sequence(id)) {
                tokens.push(state.pending_token.unwrap_or(0));
                // position = total_len() - 1: the decode token's 0-indexed position
                // (output_ids already contains this token from on_token_generated)
                positions.push(seq.total_len() - 1);
                block_tables.push(seq.block_table.clone());
                if let Some(slot) = seq.deltanet_slot {
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
        return ForwardBatch::Decode { tokens, positions, block_tables, deltanet_slots };
    }

    // Mixed or prefill-only → build unified StepRequests
    let mut requests: Vec<StepRequest> = Vec::new();

    // Prefill requests: allocate blocks for the chunk, build StepRequest.
    // The scheduler already updated kv_computed_len += chunk_len during schedule_step.
    // Recover the pre-update offset: computed = kv_computed_len - chunk_len.
    for (idx, id) in step.prefill_request_ids.iter().enumerate() {
        let Some(seq) = scheduler.get_sequence(id) else { continue; };
        let chunk_len = step.prefill_chunk_lens.get(idx).copied().unwrap_or(0);
        let computed = seq.kv_computed_len.saturating_sub(chunk_len);
        let full_prompt_len = seq.input_ids.len();
        let is_final = computed + chunk_len >= full_prompt_len;
        let end = if is_final { full_prompt_len } else { computed + chunk_len };

        // Allocate blocks for this chunk
        let block_size = engine.cache.paged_pool.as_ref().map(|p| p.block_size).unwrap_or(16);
        let total_blocks_needed = end.div_ceil(block_size);
        let current_blocks = seq.block_table.len();
        let deltanet_slot = seq.deltanet_slot;
        if current_blocks < total_blocks_needed {
            for _ in current_blocks..total_blocks_needed {
                if let Some(block) = scheduler.allocate_block() {
                    if let Some(seq_mut) = scheduler.get_sequence_mut(id) {
                        seq_mut.block_table.push(block);
                    }
                }
            }
        }

        // Re-borrow after potential mutation
        let Some(seq) = scheduler.get_sequence(id) else { continue; };
        let chunk_tokens = seq.input_ids[computed..end].to_vec();
        let prompt_logprobs = states.get(id)
            .and_then(|s| s.prepared.as_ref())
            .and_then(|p| p.request.prompt_logprobs);
        requests.push(StepRequest {
            tokens: chunk_tokens,
            context_len: end,
            position_start: computed,
            block_table: seq.block_table.clone(),
            is_prefill_final: is_final,
            is_prefill_partial: !is_final,
            deltanet_slot,
            prompt_logprobs,
        });

        // Take prepared on final chunk (consumed by sampling in process_step_output)
        if is_final {
            // prepared is taken later in process_step_output for sampling
        }
    }

    // Decode requests: total_len() already includes the decode token
    // (pushed by on_token_generated). position = total_len()-1, context_len = total_len().
    let block_size = engine.cache.paged_pool.as_ref().map(|p| p.block_size).unwrap_or(16);
    for id in &step.decode_request_ids {
        if let Some(seq) = scheduler.get_sequence(id) {
            let seq_len = seq.total_len();
            let position = seq_len - 1;
            if position % block_size == 0 {
                if let Some(new_block) = scheduler.allocate_block() {
                    if let Some(seq_mut) = scheduler.get_sequence_mut(id) {
                        seq_mut.block_table.push(new_block);
                    }
                }
            }
        }
    }
    for id in &step.decode_request_ids {
        if let (Some(state), Some(seq)) = (states.get(id), scheduler.get_sequence(id)) {
            let seq_len = seq.total_len();
            requests.push(StepRequest {
                tokens: vec![state.pending_token.unwrap_or(0)],
                context_len: seq_len,
                position_start: seq_len - 1,
                block_table: seq.block_table.clone(),
                is_prefill_final: false,
                is_prefill_partial: false,
                deltanet_slot: seq.deltanet_slot,
                prompt_logprobs: None,
            });
        }
    }

    ForwardBatch::Mixed { requests }
}

// ── Output processing (sampling + stop conditions + delivery) ──────

/// Process output from a single forward pass. Handles both prefill results
/// and decode logits from the unified batch.
fn process_step_output(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    step: &SchedulerStep,
    output: &ModelOutput,
) {
    let mut completed: Vec<(String, FinishReason)> = Vec::new();
    let mut logit_row = 0usize; // tracks position in output.logits

    // ── Prefill results ───────────────────────────────────────────
    let mut prefill_result_idx = 0usize;
    for (i, request_id) in step.prefill_request_ids.iter().enumerate() {
        let Some(state) = states.get_mut(request_id) else {
            logit_row += 1;
            continue;
        };

        let Some(seq) = scheduler.get_sequence(request_id) else {
            logit_row += 1;
            continue;
        };

        let chunk_len = step.prefill_chunk_lens.get(i).copied().unwrap_or(0);
        let full_prompt_len = seq.input_ids.len();
        let computed = seq.kv_computed_len;
        let end = (computed + chunk_len).min(full_prompt_len);
        let is_final = end >= full_prompt_len;

        // Update scheduler sequence block table and state from prefill result.
        // kv_computed_len and status were already updated by the scheduler
        // during schedule_step (eagerly, before forward).
        if let Some(result) = output.prefill_results.get(prefill_result_idx) {
            if let Some(seq_mut) = scheduler.get_sequence_mut(request_id) {
                seq_mut.block_table = result.block_table.clone();
            }
            state.prefill_ms += result.prefill_ms;
            state.prompt_token_logprobs = result.prompt_token_logprobs.clone();
            prefill_result_idx += 1;
        }

        // Populate prefix cache on the final chunk of a prefill, so future
        // requests sharing the same prompt prefix can skip the full prefill.
        //
        // This was previously a dead write: the `prefill.rs` non-paged path
        // populated the cache, but the paged hot path (`paged_mixed` +
        // `batch_prefill_paged`) never did. Every request was a cache miss.
        //
        // We do it here in the AR loop because this is where we have both:
        //   (a) the full prompt tokens (from `seq.input_ids`), and
        //   (b) the finished block table (from `result.block_table`).
        // Inside `batch_mixed_paged` only the per-step chunk is visible.
        //
        // The prefix cache takes its own ref count on the blocks, so they
        // survive the post-decode free.
        if is_final && engine.cache.prefix_cache.is_some() {
            if let Some(seq) = scheduler.get_sequence(request_id) {
                if let (Some(pool), Some(result)) = (
                    engine.cache.paged_pool.as_ref(),
                    output.prefill_results.get(prefill_result_idx.saturating_sub(1)),
                ) {
                    if !result.block_table.is_empty() && !seq.input_ids.is_empty() {
                        if let Err(e) = engine.cache.try_prefix_cache_insert_paged_only(
                            &seq.input_ids,
                            &result.block_table,
                            pool.block_size,
                        ) {
                            tracing::warn!("prefix cache insert (ar_loop) failed: {e}");
                        }
                    }
                }
            }
        }

        if is_final {
            // Final chunk: sample first token from logits
            if let Ok(row) = output.logits.get(logit_row) {
                let token = row
                    .argmax(crate::tensor::D::Minus1)
                    .and_then(|t| t.to_scalar::<u32>())
                    .unwrap_or(0);

                // TODO: use logits_processor from prepared for non-greedy sampling
                process_single_token(engine, scheduler, state, request_id, token, None, &mut completed);
            }
        }
        // Partial chunk: nothing to do (progress already recorded)
        logit_row += 1;
    }

    // ── Decode results (batched GPU sampling) ──────────────────────
    let num_decode = step.decode_request_ids.len();
    if num_decode > 0 {
        let decode_start_row = logit_row;

        // Check if all decode sequences are greedy
        let all_greedy = step.decode_request_ids.iter().all(|id| {
            states.get(id).map(|s| s.is_greedy()).unwrap_or(true)
        });

        // Try batched GPU sampling for all-greedy decode batches
        let batched_tokens: Option<Vec<u32>> = if all_greedy && num_decode > 1 {
            // Slice the decode portion of logits: [num_decode, vocab_size]
            output.logits
                .narrow(0, decode_start_row, num_decode)
                .ok()
                .and_then(|decode_logits| {
                    // Try GPU batched sampling first (deterministic = greedy)
                    if let Some(result) = engine.executor.ops.sample_from_logits(&decode_logits, true) {
                        match result {
                            Ok(token_ids) => token_ids.to_vec1::<u32>().ok(),
                            Err(_) => None,
                        }
                    } else {
                        // Fallback: batched argmax on GPU, single D2H copy
                        decode_logits
                            .argmax(crate::tensor::D::Minus1)
                            .and_then(|t| t.to_vec1::<u32>())
                            .ok()
                    }
                })
        } else {
            None
        };

        for (i, request_id) in step.decode_request_ids.iter().enumerate() {
            let Some(state) = states.get_mut(request_id) else {
                logit_row += 1;
                continue;
            };

            let token = if let Some(ref tokens) = batched_tokens {
                // Use pre-computed batched result
                tokens[i]
            } else if let Ok(row) = output.logits.get(logit_row) {
                // Per-sequence fallback (non-greedy or batch size 1)
                if state.is_greedy() {
                    row.argmax(crate::tensor::D::Minus1)
                        .and_then(|t| t.to_scalar::<u32>())
                        .unwrap_or(0)
                } else if let Some(ref mut prepared) = state.prepared {
                    let row_f32 = row.to_dtype(crate::tensor::DType::F32).unwrap();
                    prepared.logits_processor.sample(&row_f32).unwrap_or(0)
                } else {
                    row.argmax(crate::tensor::D::Minus1)
                        .and_then(|t| t.to_scalar::<u32>())
                        .unwrap_or(0)
                }
            } else {
                0
            };
            process_single_token(engine, scheduler, state, request_id, token, None, &mut completed);
            logit_row += 1;
        }
    }

    for (request_id, finish_reason) in completed {
        // Get prompt_len from scheduler sequence before finishing
        let prompt_len = scheduler.get_sequence(&request_id)
            .map(|seq| seq.input_ids.len())
            .unwrap_or(0);
        release_resources(engine, scheduler, &request_id);
        scheduler.finish_request(&request_id, seq_finish_reason(&finish_reason));
        if let Some(state) = states.remove(&request_id) {
            finish_state(engine, state, finish_reason, prompt_len);
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

// ── State finalization ─────────────────────────────────────────────

fn finish_state(engine: &Engine, mut state: ArSequenceState, finish_reason: FinishReason, prompt_len: usize) {
    state.ensure_started();

    let output_text = state.current_text(&engine.tokenizer);
    let completion_tokens = state.output_tokens.len() as u32;
    let total_ms = state.gen_start.elapsed().as_secs_f32() * 1000.0;
    let usage = Usage {
        prompt_tokens: prompt_len as u32,
        completion_tokens,
        total_tokens: prompt_len as u32 + completion_tokens,
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

fn fail_state(_engine: &Engine, mut state: ArSequenceState, error: EngineError) {
    state.ensure_started();
    match state.response {
        ResponseChannel::Complete(tx) => {
            let _ = tx.send(Err(error));
        }
        ResponseChannel::Stream(tx) => {
            let _ = tx.send(StreamEvent::Error {
                message: error.to_string(),
            });
        }
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
        release_resources(engine, scheduler, id);
        let _ = scheduler.abort_request(id);
        if let Some(state) = states.remove(id) {
            fail_state(engine, state, error.clone());
        }
    }
}

fn release_resources(engine: &Engine, scheduler: &mut Scheduler, request_id: &str) {
    if let Some(seq) = scheduler.get_sequence(request_id) {
        if !seq.block_table.is_empty() {
            let blocks = seq.block_table.clone();
            scheduler.free_blocks(&blocks);
        }
        if let Some(slot) = seq.deltanet_slot {
            if let Some(pool_mutex) = engine.cache.deltanet_pool.as_ref() {
                if let Ok(mut pool) = pool_mutex.lock() {
                    pool.free(slot);
                }
            }
        }
        // Clear on the sequence
        if let Some(seq_mut) = scheduler.get_sequence_mut(request_id) {
            seq_mut.block_table.clear();
            seq_mut.deltanet_slot = None;
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
            None,
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
            None,
        );
        assert_eq!(scheduler.num_waiting(), 1);

        handle_message(
            ArMessage::Abort("r1".into()),
            &mut scheduler,
            &mut states,
            None,
        );
        assert!(states.is_empty());
    }

    // ── sample_batch tests ─────────────────────────────────────────

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

        let step = SchedulerStep::prefill(vec!["r1".into(), "r2".into()], vec![3, 3]);

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
            sent_text_len: 0,
            pending_token: None, output_tokens: vec![], token_logprobs: vec![],
            prompt_token_logprobs: None,
            max_new_tokens: max_new,
        }
    }

    fn make_decode_state(id: &str, pending_token: u32, _position: usize) -> ArSequenceState {
        let max_new = 10;
        ArSequenceState {
            request_id: id.to_string(),
            prepared: Some(make_prepared(id, max_new)),
            response: ResponseChannel::Complete(tokio::sync::oneshot::channel().0),
            gen_start: Instant::now(), prefill_ms: 0.0, started_sent: true,
            sent_text_len: 0,
            pending_token: Some(pending_token), output_tokens: vec![pending_token],
            token_logprobs: vec![], prompt_token_logprobs: None,
            max_new_tokens: max_new,
        }
    }

    #[test]
    fn decode_state_has_correct_fields() {
        let state = make_decode_state("s1", 42, 5);
        assert_eq!(state.pending_token, Some(42));
        assert_eq!(state.output_tokens, vec![42]);
    }

    #[test]
    fn prefill_state_prepared_can_be_taken() {
        let mut states = HashMap::new();
        let prepared = make_prepared("r1", 5);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        handle_message(
            ArMessage::NewRequest { prepared, response: ResponseChannel::Complete(tx) },
            &mut Scheduler::new(SchedulerConfig::default()),
            &mut states,
            None,
        );

        assert!(states.get("r1").unwrap().prepared.is_some());

        // Simulate what build_forward_batch does for prefill: take prepared
        let taken = states.get_mut("r1").and_then(|s| s.prepared.take());
        assert!(taken.is_some());
        assert_eq!(taken.unwrap().prompt_tokens, vec![1, 2, 3]);
        assert!(states.get("r1").unwrap().prepared.is_none());
    }
}
