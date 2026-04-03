//! Continuous generation runtime (multi-token decode with paged attention).
//!
//! Requires flash attention (flash-attn-v4 or flashinfer feature). Without it,
//! the loop rejects all requests with an error.
//!
//! State flow per request:
//! ```text
//! NewRequest → scheduler.add_request → execute_prefill_step → decode loop
//!   └─ each decode step: allocate block → submit_decode_paged → sample → emit token
//!   └─ finish: EOS / length / stop string → finish_runtime_state
//! ```

use std::sync::Arc;
use tokio::sync::mpsc;

use crate::engine::{Engine, EngineError};
use crate::runtime::SchedulerConfig;

use crate::engine::scheduled::ContinuousSchedulerMsg;
use super::gpu_queue::GpuQueueTx;

/// Stub: rejects all requests when flash attention is not available.
#[cfg(not(any(feature = "flash-attn-v4", feature = "flashinfer")))]
pub(crate) async fn continuous_generation_loop(
    _engine: Arc<Engine>,
    _config: SchedulerConfig,
    mut rx: mpsc::UnboundedReceiver<ContinuousSchedulerMsg>,
    _gpu_tx: GpuQueueTx,
) {
    while let Some(msg) = rx.recv().await {
        match msg {
            ContinuousSchedulerMsg::NewRequest(inflight) => {
                inflight.fail(EngineError::Unavailable(
                    "continuous generation requires flash attention (flash-attn-v4 or flashinfer feature)".into(),
                ));
            }
            ContinuousSchedulerMsg::Abort(_) => {}
        }
    }
    tracing::info!("continuous generation loop exited");
}

// ===========================================================================
// Real implementation (FA4/FlashInfer)
// ===========================================================================

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
use std::collections::{HashMap, HashSet};
#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
use std::time::Instant;
#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
use crate::tensor::{DType, Tensor};
#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
use tokio::sync::mpsc::error::TryRecvError;
#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
use crate::engine::{
    DecodeMetrics, FinishReason, GenerateResult,
    OwnedBatchDecodeSeq, PreparedGenerateRequest, TokenLogprobInfo, Usage,
};
#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
use crate::runtime::{ForwardMode, Scheduler, SeqFinishReason, Sequence};
#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
use crate::engine::scheduled::{
    ContinuousGenerationRequestState, GenerationResponseChannel,
};
#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
use super::gpu_queue::{submit_decode_paged, submit_prefill_paged};
#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
use crate::types::StreamEvent;

// ---------------------------------------------------------------------------
// Per-request runtime state
// ---------------------------------------------------------------------------

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
struct RuntimeSequenceState {
    request_id: String,
    prepared: Option<PreparedGenerateRequest>,
    response: GenerationResponseChannel,
    gen_start: Instant,
    prefill_ms: f32,
    started_sent: bool,
    sent_text_len: usize,
    prompt_len: usize,
    next_decode_position: usize,
    pending_token: Option<u32>,
    output_tokens: Vec<u32>,
    token_logprobs: Vec<TokenLogprobInfo>,
    prompt_token_logprobs: Option<Vec<TokenLogprobInfo>>,
    block_table: Vec<u32>,
    deltanet_slot: Option<u32>,
}

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
impl RuntimeSequenceState {
    fn new(inflight: ContinuousGenerationRequestState) -> Self {
        Self {
            request_id: inflight.prepared.request.request_id.clone(),
            prepared: Some(inflight.prepared),
            response: inflight.response,
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
        }
    }

    fn request(&self) -> &crate::types::GenerateRequest {
        &self.prepared.as_ref().expect("prepared request missing").request
    }
    fn prepared(&self) -> &PreparedGenerateRequest {
        self.prepared.as_ref().expect("prepared request missing")
    }
    fn prepared_mut(&mut self) -> &mut PreparedGenerateRequest {
        self.prepared.as_mut().expect("prepared request missing")
    }
    fn take_prepared(&mut self) -> PreparedGenerateRequest {
        self.prepared.take().expect("prepared request missing")
    }
    fn put_prepared(&mut self, prepared: PreparedGenerateRequest) {
        self.prepared = Some(prepared);
    }
    fn max_new(&self) -> usize { self.prepared().max_new }

    fn is_active_for_decode(&self) -> bool {
        self.pending_token.is_some() && self.output_tokens.len() < self.max_new()
    }

    fn ensure_started(&mut self) {
        if self.started_sent { return; }
        if let GenerationResponseChannel::Stream(tx) = &self.response {
            let _ = tx.send(StreamEvent::Started);
        }
        self.started_sent = true;
    }

    fn emit_text_delta(
        &mut self,
        tokenizer: &fastokens::Tokenizer,
        logprobs: Option<TokenLogprobInfo>,
    ) {
        let GenerationResponseChannel::Stream(tx) = &self.response else { return; };
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

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
pub(crate) async fn continuous_generation_loop(
    engine: Arc<Engine>,
    config: SchedulerConfig,
    mut rx: mpsc::UnboundedReceiver<ContinuousSchedulerMsg>,
    gpu_tx: GpuQueueTx,
) {
    let mut runtime_config = config;
    runtime_config.mixed_chunked = true;
    runtime_config.max_total_tokens = usize::MAX / 4;

    let mut scheduler = Scheduler::new(runtime_config);
    let mut states: HashMap<String, RuntimeSequenceState> = HashMap::new();
    let mut rx_open = true;

    loop {
        if !scheduler.has_work() {
            match rx.recv().await {
                Some(msg) => handle_runtime_msg(msg, &mut scheduler, &mut states, engine.as_ref()),
                None => rx_open = false,
            }
        }

        loop {
            match rx.try_recv() {
                Ok(msg) => handle_runtime_msg(msg, &mut scheduler, &mut states, engine.as_ref()),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => { rx_open = false; break; }
            }
        }

        if let Some(step) = scheduler.schedule_step() {
            if !step.prefill_request_ids.is_empty() {
                let ready_prefills = fit_prefill_batch_to_resources(
                    engine.as_ref(), &mut scheduler, &states, &step.prefill_request_ids,
                );
                if !ready_prefills.is_empty() {
                    execute_prefill_step(engine.as_ref(), &mut scheduler, &mut states, ready_prefills, &gpu_tx).await;
                }
            }
            if matches!(step.forward_mode, ForwardMode::Decode | ForwardMode::Mixed)
                && !step.decode_request_ids.is_empty()
            {
                execute_decode_step(engine.as_ref(), &mut scheduler, &mut states, step.decode_request_ids, &gpu_tx).await;
                tokio::task::yield_now().await;
            }
        } else if !rx_open {
            break;
        } else if scheduler.has_work() {
            tokio::task::yield_now().await;
        }

        if !rx_open && !scheduler.has_work() { break; }
    }

    for (_, state) in states.drain() {
        fail_runtime_state(engine.as_ref(), state, EngineError::Unavailable("scheduler loop stopped".into()));
    }
    while let Ok(msg) = rx.try_recv() {
        if let ContinuousSchedulerMsg::NewRequest(inflight) = msg {
            inflight.fail(EngineError::Unavailable("scheduler loop stopped".into()));
        }
    }
    tracing::info!("continuous generation loop exited");
}

// ---------------------------------------------------------------------------
// Message handling
// ---------------------------------------------------------------------------

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn handle_runtime_msg(
    msg: ContinuousSchedulerMsg,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, RuntimeSequenceState>,
    engine: &Engine,
) {
    match msg {
        ContinuousSchedulerMsg::NewRequest(inflight) => {
            let request_id = inflight.request_id().to_string();
            let prepared = &inflight.prepared;
            let seq = Sequence::new(
                request_id.clone(),
                prepared.prompt_tokens.clone(),
                prepared.request.sampling.clone(),
                prepared.max_new.saturating_sub(1) as u32,
                prepared.request.stop.strings.clone(),
                prepared.request.stop.token_ids.clone(),
                None,
            );
            scheduler.add_request(seq);
            states.insert(request_id, RuntimeSequenceState::new(inflight));
        }
        ContinuousSchedulerMsg::Abort(request_id) => {
            let _ = scheduler.abort_request(&request_id);
            if let Some(state) = states.remove(&request_id) {
                fail_runtime_state(engine, state, EngineError::Internal("aborted".into()));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill step
// ---------------------------------------------------------------------------

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn fit_prefill_batch_to_resources(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &HashMap<String, RuntimeSequenceState>,
    request_ids: &[String],
) -> Vec<String> {
    if request_ids.is_empty() { return Vec::new(); }

    let mut fit_count = request_ids.len();

    if let Some(pool) = engine.cache.paged_pool.as_ref() {
        if let Some(bm_mutex) = engine.cache.block_manager.as_ref() {
            if let Ok(bm) = bm_mutex.lock() {
                let block_size = pool.block_size.max(1);
                let mut required_blocks = 0usize;
                fit_count = 0;
                for request_id in request_ids {
                    let Some(state) = states.get(request_id) else { break; };
                    required_blocks += state.prepared().prompt_tokens.len().div_ceil(block_size);
                    if required_blocks > bm.available() { break; }
                    fit_count += 1;
                }
            }
        }
    }

    if let Some(pool_mutex) = engine.cache.deltanet_pool.as_ref() {
        if let Ok(pool) = pool_mutex.lock() {
            fit_count = fit_count.min(pool.available());
        }
    }

    let ready = request_ids[..fit_count].to_vec();
    let deferred = &request_ids[fit_count..];
    if !deferred.is_empty() {
        scheduler.rollback_prefill(deferred);
    }
    ready
}

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
async fn execute_prefill_step(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, RuntimeSequenceState>,
    request_ids: Vec<String>,
    gpu_tx: &GpuQueueTx,
) {
    let mut batch_states = take_states(states, &request_ids);
    let mut items: Vec<PreparedGenerateRequest> = batch_states
        .iter_mut().map(RuntimeSequenceState::take_prepared).collect();
    for (idx, item) in items.iter_mut().enumerate() { item.request_idx = idx; }

    let decode_plan = match engine.build_decode_plan(&items) {
        Ok(plan) => plan,
        Err(error) => {
            for (mut state, prepared) in batch_states.into_iter().zip(items.into_iter()) {
                state.put_prepared(prepared);
                let _ = scheduler.abort_request(&state.request_id);
                fail_runtime_state(engine, state, error.clone());
            }
            return;
        }
    };

    let (items, prefill_results) = match submit_prefill_paged(gpu_tx, items, decode_plan.initial_prefill).await {
        Ok(result) => result,
        Err(error) => {
            for state in batch_states.into_iter() {
                let _ = scheduler.abort_request(&state.request_id);
                fail_runtime_state(engine, state, error.clone());
            }
            return;
        }
    };

    for ((mut state, prepared), prefill) in batch_states.into_iter().zip(items).zip(prefill_results) {
        state.put_prepared(prepared);
        state.ensure_started();
        state.prefill_ms = prefill.prefill_ms;
        state.prompt_len = prefill.prompt_len;
        state.next_decode_position = prefill.prompt_len;
        state.block_table = prefill.block_table;
        state.deltanet_slot = prefill.deltanet_slot;
        state.prompt_token_logprobs = prefill.prompt_token_logprobs;

        let first_token = prefill.first_token;
        let first_token_logprobs = prefill.first_token_logprobs.clone();
        let request_id = state.request_id.clone();

        if engine.is_eos(first_token) {
            scheduler.finish_request(&request_id, SeqFinishReason::Eos);
            finish_runtime_state(engine, state, FinishReason::Eos);
            continue;
        }
        if engine.check_stop_tokens(first_token, &state.request().stop.token_ids) {
            scheduler.finish_request(&request_id, SeqFinishReason::Stop);
            finish_runtime_state(engine, state, FinishReason::Stop);
            continue;
        }

        state.pending_token = Some(first_token);
        state.output_tokens.push(first_token);
        if let Some(ref lp) = first_token_logprobs { state.token_logprobs.push(lp.clone()); }
        state.emit_text_delta(&engine.tokenizer, first_token_logprobs);

        let text = state.current_text(&engine.tokenizer);
        if state.output_tokens.len() >= state.max_new() {
            scheduler.finish_request(&request_id, SeqFinishReason::Length);
            finish_runtime_state(engine, state, FinishReason::Length);
        } else if !state.request().stop.strings.is_empty()
            && state.request().stop.strings.iter().any(|s| text.contains(s))
        {
            scheduler.finish_request(&request_id, SeqFinishReason::Stop);
            finish_runtime_state(engine, state, FinishReason::Stop);
        } else {
            states.insert(request_id, state);
        }
    }
}

// ---------------------------------------------------------------------------
// Decode step
// ---------------------------------------------------------------------------

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
async fn execute_decode_step(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, RuntimeSequenceState>,
    request_ids: Vec<String>,
    gpu_tx: &GpuQueueTx,
) {
    let Some(pool) = engine.cache.paged_pool.as_ref() else {
        fail_all(engine, scheduler, states, &request_ids,
            EngineError::Unavailable("paged attention pool unavailable".into()));
        return;
    };

    let mut active_ids = Vec::new();
    let mut exhausted_ids = Vec::new();
    let mut allocation_failed = None::<EngineError>;

    if let Some(bm_mutex) = engine.cache.block_manager.as_ref() {
        match bm_mutex.lock() {
            Ok(mut bm) => {
                for request_id in request_ids {
                    let Some(state) = states.get_mut(&request_id) else { continue; };
                    if !state.is_active_for_decode() { continue; }
                    if state.next_decode_position % pool.block_size == 0 {
                        if let Some(new_block) = bm.allocate() {
                            state.block_table.push(new_block);
                        } else {
                            exhausted_ids.push(request_id.clone());
                            continue;
                        }
                    }
                    active_ids.push(request_id);
                }
            }
            Err(e) => {
                allocation_failed = Some(EngineError::Internal(format!("block manager lock failed: {e}")));
            }
        }
    } else {
        allocation_failed = Some(EngineError::Unavailable("block manager unavailable".into()));
    }

    if let Some(error) = allocation_failed {
        let all_ids: Vec<_> = active_ids.into_iter().chain(exhausted_ids).collect();
        fail_all(engine, scheduler, states, &all_ids, error);
        return;
    }

    for request_id in exhausted_ids {
        scheduler.finish_request(&request_id, SeqFinishReason::Length);
        if let Some(state) = states.remove(&request_id) {
            finish_runtime_state(engine, state, FinishReason::Length);
        }
    }

    if active_ids.is_empty() { return; }

    let owned_seqs: Vec<OwnedBatchDecodeSeq> = active_ids.iter()
        .filter_map(|id| states.get(id))
        .map(|state| OwnedBatchDecodeSeq {
            token: state.pending_token.expect("missing pending token"),
            position: state.next_decode_position,
            context_len: state.next_decode_position + 1,
            block_table: state.block_table.clone(),
            deltanet_slot: state.deltanet_slot,
        })
        .collect();

    let logits_2d = match submit_decode_paged(gpu_tx, owned_seqs).await {
        Ok(logits) => logits,
        Err(error) => {
            fail_all(engine, scheduler, states, &active_ids, error);
            return;
        }
    };

    let sampled_tokens = sample_batch(engine, scheduler, states, &active_ids, &logits_2d);
    let Some(sampled_tokens) = sampled_tokens else { return; };
    process_sampled_tokens(engine, scheduler, states, &active_ids, &sampled_tokens, &logits_2d);
}

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn sample_batch(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, RuntimeSequenceState>,
    active_ids: &[String],
    logits_2d: &Tensor,
) -> Option<Vec<u32>> {
    let all_greedy = active_ids.iter().all(|id| {
        states.get(id).map(|s| s.prepared().is_greedy).unwrap_or(true)
    });

    if all_greedy {
        match logits_2d.argmax(crate::tensor::D::Minus1).and_then(|t| t.to_vec1::<u32>()) {
            Ok(tokens) => Some(tokens),
            Err(error) => {
                fail_all(engine, scheduler, states, active_ids,
                    EngineError::Internal(format!("batch argmax failed: {error}")));
                None
            }
        }
    } else {
        let mut tokens = Vec::with_capacity(active_ids.len());
        let mut failed = Vec::new();
        for (row_idx, request_id) in active_ids.iter().enumerate() {
            let row = match logits_2d.get(row_idx) {
                Ok(row) => row,
                Err(error) => {
                    failed.push((request_id.clone(),
                        EngineError::Internal(format!("get logits row failed: {error}"))));
                    tokens.push(0);
                    continue;
                }
            };
            let Some(state) = states.get_mut(request_id) else { tokens.push(0); continue; };
            match sample_next_token(state, &row) {
                Ok(token) => tokens.push(token),
                Err(error) => { failed.push((request_id.clone(), error)); tokens.push(0); }
            }
        }
        for (request_id, error) in failed {
            let _ = scheduler.abort_request(&request_id);
            if let Some(state) = states.remove(&request_id) {
                fail_runtime_state(engine, state, error);
            }
        }
        Some(tokens)
    }
}

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn process_sampled_tokens(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, RuntimeSequenceState>,
    active_ids: &[String],
    sampled_tokens: &[u32],
    logits_2d: &Tensor,
) {
    let mut completed: Vec<(String, FinishReason)> = Vec::new();

    for (row_idx, request_id) in active_ids.iter().enumerate() {
        let Some(state) = states.get_mut(request_id) else { continue; };
        let next_token = sampled_tokens[row_idx];
        scheduler.on_token_generated(request_id, next_token);
        state.next_decode_position += 1;

        let token_logprobs = if let Some(k) = state.request().logprobs {
            logits_2d.get(row_idx).ok()
                .and_then(|row| Engine::extract_top_logprobs(&row, next_token, k, &engine.tokenizer).ok())
        } else {
            None
        };

        if engine.is_eos(next_token) {
            completed.push((request_id.clone(), FinishReason::Eos));
            continue;
        }
        if engine.check_stop_tokens(next_token, &state.request().stop.token_ids) {
            completed.push((request_id.clone(), FinishReason::Stop));
            continue;
        }

        state.pending_token = Some(next_token);
        state.output_tokens.push(next_token);
        if let Some(ref lp) = token_logprobs { state.token_logprobs.push(lp.clone()); }
        state.emit_text_delta(&engine.tokenizer, token_logprobs);

        let text = state.current_text(&engine.tokenizer);
        if state.output_tokens.len() >= state.max_new() {
            completed.push((request_id.clone(), FinishReason::Length));
        } else if !state.request().stop.strings.is_empty()
            && state.request().stop.strings.iter().any(|s| text.contains(s))
        {
            completed.push((request_id.clone(), FinishReason::Stop));
        }
    }

    for (request_id, finish_reason) in completed {
        scheduler.finish_request(&request_id, seq_finish_reason(&finish_reason));
        if let Some(state) = states.remove(&request_id) {
            finish_runtime_state(engine, state, finish_reason);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn sample_next_token(state: &mut RuntimeSequenceState, row: &Tensor) -> Result<u32, EngineError> {
    if state.prepared().is_greedy {
        row.argmax(crate::tensor::D::Minus1)
            .and_then(|t| t.to_scalar::<u32>())
            .map_err(|e| EngineError::Internal(format!("argmax failed: {e}")))
    } else {
        let row_f32 = row.to_dtype(DType::F32)
            .map_err(|e| EngineError::Internal(format!("to_dtype failed: {e}")))?;
        state.prepared_mut().logits_processor.sample(&row_f32)
            .map_err(|e| EngineError::Internal(format!("sample failed: {e}")))
    }
}

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn take_states(
    states: &mut HashMap<String, RuntimeSequenceState>,
    request_ids: &[String],
) -> Vec<RuntimeSequenceState> {
    request_ids.iter().filter_map(|id| states.remove(id)).collect()
}

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn fail_all(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, RuntimeSequenceState>,
    request_ids: &[String],
    error: EngineError,
) {
    for request_id in request_ids {
        let _ = scheduler.abort_request(request_id);
        if let Some(state) = states.remove(request_id) {
            fail_runtime_state(engine, state, error.clone());
        }
    }
}

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn finish_runtime_state(engine: &Engine, mut state: RuntimeSequenceState, finish_reason: FinishReason) {
    state.ensure_started();
    release_runtime_state_resources(engine, &mut state);

    let output_text = state.current_text(&engine.tokenizer);
    let completion_tokens = state.output_tokens.len() as u32;
    let total_ms = state.gen_start.elapsed().as_secs_f32() * 1000.0;
    let usage = Usage {
        prompt_tokens: state.prompt_len as u32,
        completion_tokens,
        total_tokens: state.prompt_len as u32 + completion_tokens,
    };
    let metrics = DecodeMetrics {
        ttft_ms: state.prefill_ms, prefill_ms: state.prefill_ms,
        decode_ms: (total_ms - state.prefill_ms).max(0.0), total_ms,
    };
    let result = GenerateResult {
        model: state.request().model.clone(),
        output_token_ids: state.output_tokens, output_text,
        finish_reason: finish_reason.clone(),
        usage: usage.clone(), metrics: metrics.clone(),
        token_logprobs: if state.token_logprobs.is_empty() { None } else { Some(state.token_logprobs) },
        prompt_token_logprobs: state.prompt_token_logprobs,
    };
    match state.response {
        GenerationResponseChannel::Complete(tx) => { let _ = tx.send(Ok(result)); }
        GenerationResponseChannel::Stream(tx) => {
            let _ = tx.send(StreamEvent::Finished { finish_reason, usage, metrics });
        }
    }
}

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn fail_runtime_state(engine: &Engine, mut state: RuntimeSequenceState, error: EngineError) {
    release_runtime_state_resources(engine, &mut state);
    match state.response {
        GenerationResponseChannel::Complete(tx) => { let _ = tx.send(Err(error)); }
        GenerationResponseChannel::Stream(_) => {} // Dropping sender closes the stream.
    }
}

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn release_runtime_state_resources(engine: &Engine, state: &mut RuntimeSequenceState) {
    if !state.block_table.is_empty() {
        if let Some(bm_mutex) = engine.cache.block_manager.as_ref() {
            if let Ok(mut bm) = bm_mutex.lock() { bm.free(&state.block_table); }
        }
        state.block_table.clear();
    }
    if let Some(slot) = state.deltanet_slot.take() {
        if let Some(pool_mutex) = engine.cache.deltanet_pool.as_ref() {
            if let Ok(mut pool) = pool_mutex.lock() { pool.free(slot); }
        }
    }
}

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn seq_finish_reason(reason: &FinishReason) -> SeqFinishReason {
    match reason {
        FinishReason::Stop => SeqFinishReason::Stop,
        FinishReason::Length => SeqFinishReason::Length,
        FinishReason::Eos => SeqFinishReason::Eos,
        FinishReason::Cancelled => SeqFinishReason::Abort("cancelled".into()),
    }
}
