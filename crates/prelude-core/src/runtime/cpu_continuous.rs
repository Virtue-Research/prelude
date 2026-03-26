//! CPU continuous generation runtime — per-token streaming decode.
//!
//! Mirrors the GPU continuous runtime structure but uses the model's internal
//! KvBuf cache (`forward_with_cache`) instead of paged attention. Processes
//! one request at a time with per-token streaming.
//!
//! ```text
//! loop {
//!     request = recv()
//!     prefill(prompt)            → emit first token
//!     for step in 1..max_new:
//!         decode(last_token)     → emit token
//!     clear_kv_cache()
//! }
//! ```

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use crate::engine::{Engine, EngineError};
use crate::types::{DecodeMetrics, FinishReason, GenerateResult, StreamEvent, Usage};

use super::request_state::{
    ContinuousGenerationRequestState, ContinuousSchedulerMsg, GenerationResponseChannel,
};

/// Spawn the CPU continuous generation loop on a dedicated OS thread.
///
/// Runs on its own thread (like the GPU worker) to avoid tokio worker pool
/// expansion from `block_in_place`. This prevents thread contention between
/// tokio's replacement workers and llama.cpp's OpenMP pool.
pub(crate) fn spawn_cpu_continuous_worker(
    engine: Arc<Engine>,
    rx: mpsc::UnboundedReceiver<ContinuousSchedulerMsg>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("cpu-continuous".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("cpu continuous worker runtime");
            rt.block_on(cpu_continuous_generation_loop(engine, rx));
        })
        .expect("spawn cpu continuous worker thread")
}

/// CPU continuous generation loop with per-token streaming.
///
/// Receives `ContinuousSchedulerMsg` from the scheduler, processes one request
/// at a time: prefill → per-token decode → finish. Each token is emitted via
/// the response channel as it's generated.
pub(crate) async fn cpu_continuous_generation_loop(
    engine: Arc<Engine>,
    mut rx: mpsc::UnboundedReceiver<ContinuousSchedulerMsg>,
) {
    let mut waiting: VecDeque<ContinuousGenerationRequestState> = VecDeque::new();
    let mut aborted: HashSet<String> = HashSet::new();

    loop {
        // Wait for first request if idle
        if waiting.is_empty() {
            match rx.recv().await {
                Some(msg) => handle_msg(msg, &mut waiting, &mut aborted),
                None => break, // channel closed
            }
        }

        // Drain channel (non-blocking)
        while let Ok(msg) = rx.try_recv() {
            handle_msg(msg, &mut waiting, &mut aborted);
        }

        // Process one request
        if let Some(state) = waiting.pop_front() {
            if aborted.remove(state.request_id()) {
                continue;
            }

            process_request(&engine, state);
        }
    }
}

fn handle_msg(
    msg: ContinuousSchedulerMsg,
    waiting: &mut VecDeque<ContinuousGenerationRequestState>,
    aborted: &mut HashSet<String>,
) {
    match msg {
        ContinuousSchedulerMsg::NewRequest(state) => waiting.push_back(state),
        ContinuousSchedulerMsg::Abort(id) => {
            // Remove from waiting if present, otherwise mark for later skip
            if let Some(pos) = waiting.iter().position(|s| s.request_id() == id) {
                waiting.remove(pos);
            } else {
                aborted.insert(id);
            }
        }
    }
}

fn process_request(engine: &Engine, state: ContinuousGenerationRequestState) {
    let prepared = state.prepared;
    let response = state.response;
    let gen_start = Instant::now();

    let prompt_len = prepared.prompt_tokens.len();
    let max_new = prepared.max_new.max(1);
    let logprobs_k = prepared.request.logprobs;
    let model_id = prepared.request.model.clone();
    let stop_strings = prepared.request.stop.strings.clone();

    // Send Started event for streaming
    if let GenerationResponseChannel::Stream(tx) = &response {
        let _ = tx.send(StreamEvent::Started);
    }

    // ── Fast path: C-side decode loop (llama.cpp FFI) ──
    {
        let mut model = engine.executor.model.lock().unwrap();
        if let Ok(Some((tokens, _logits))) =
            model.generate_direct(&prepared.prompt_tokens, max_new)
        {
            model.clear_kv_cache();
            drop(model);

            let prefill_ms = gen_start.elapsed().as_secs_f32() * 1000.0;
            let mut finish_reason = FinishReason::Length;
            let mut output_tokens: Vec<u32> = Vec::new();
            let output_logprobs = Vec::new();

            for &tok in &tokens {
                output_tokens.push(tok);
                emit_token(engine, &response, &output_tokens, logprobs_k, &output_logprobs);
                if check_stop(
                    engine, tok, &prepared.request.stop, &output_tokens, &mut finish_reason,
                ) {
                    break;
                }
            }

            finish_response(
                response, model_id, output_tokens, output_logprobs,
                finish_reason, prompt_len, gen_start, prefill_ms, engine, &stop_strings,
            );
            return;
        }
        drop(model);
    }

    // ── Standard path: per-token Rust loop ──
    let logits = match engine.cpu_prefill_with_cache(&prepared.prompt_tokens) {
        Ok(l) => l,
        Err(e) => {
            engine.cpu_clear_kv_cache();
            fail_response(response, e);
            return;
        }
    };
    let prefill_ms = gen_start.elapsed().as_secs_f32() * 1000.0;

    // Sample first token
    let mut logits_processor = prepared.logits_processor;
    let mut output_tokens: Vec<u32> = Vec::with_capacity(max_new);
    let mut output_logprobs = Vec::new();
    let mut finish_reason = FinishReason::Length;

    let first_token = match sample_token(
        &logits, &mut logits_processor, logprobs_k, engine,
        &mut output_logprobs,
    ) {
        Ok(t) => t,
        Err(e) => {
            engine.cpu_clear_kv_cache();
            fail_response(response, e);
            return;
        }
    };
    output_tokens.push(first_token);

    // Emit first token
    emit_token(engine, &response, &output_tokens, logprobs_k, &output_logprobs);

    // Check stop after first token
    if check_stop(engine, first_token, &prepared.request.stop, &output_tokens, &mut finish_reason)
    {
        engine.cpu_clear_kv_cache();
        finish_response(
            response, model_id, output_tokens, output_logprobs,
            finish_reason, prompt_len, gen_start, prefill_ms, engine, &stop_strings,
        );
        return;
    }

    // ── Decode loop ──
    for step in 1..max_new {
        let offset = prompt_len + step - 1;
        let last_token = *output_tokens.last().unwrap();

        let logits = match engine.cpu_decode_step(last_token, offset) {
            Ok(l) => l,
            Err(e) => {
                engine.cpu_clear_kv_cache();
                fail_response(response, e);
                return;
            }
        };

        let token = match sample_token(
            &logits, &mut logits_processor, logprobs_k, engine,
            &mut output_logprobs,
        ) {
            Ok(t) => t,
            Err(e) => {
                engine.cpu_clear_kv_cache();
                fail_response(response, e);
                return;
            }
        };
        output_tokens.push(token);

        emit_token(engine, &response, &output_tokens, logprobs_k, &output_logprobs);

        if check_stop(engine, token, &prepared.request.stop, &output_tokens, &mut finish_reason) {
            break;
        }
    }

    engine.cpu_clear_kv_cache();
    finish_response(
        response, model_id, output_tokens, output_logprobs,
        finish_reason, prompt_len, gen_start, prefill_ms, engine, &stop_strings,
    );
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn sample_token(
    logits: &candle_core::Tensor,
    logits_processor: &mut crate::nn_ops::generation::LogitsProcessor,
    logprobs_k: Option<u32>,
    engine: &Engine,
    output_logprobs: &mut Vec<crate::types::TokenLogprobInfo>,
) -> Result<u32, EngineError> {
    let token = logits_processor
        .sample(logits)
        .map_err(|e| EngineError::Internal(e.to_string()))?;
    if let Some(k) = logprobs_k {
        output_logprobs.push(Engine::extract_top_logprobs(
            logits, token, k, &engine.tokenizer,
        )?);
    }
    Ok(token)
}

fn check_stop(
    engine: &Engine,
    token: u32,
    stop: &crate::types::StopConfig,
    output_tokens: &[u32],
    finish_reason: &mut FinishReason,
) -> bool {
    if engine.is_eos(token) || stop.token_ids.contains(&token) {
        *finish_reason = FinishReason::Stop;
        return true;
    }
    if !stop.strings.is_empty() {
        let text = engine.tokenizer.decode(output_tokens, true).unwrap_or_default();
        if stop.strings.iter().any(|s| text.contains(s)) {
            *finish_reason = FinishReason::Stop;
            return true;
        }
    }
    false
}

fn emit_token(
    engine: &Engine,
    response: &GenerationResponseChannel,
    output_tokens: &[u32],
    _logprobs_k: Option<u32>,
    output_logprobs: &[crate::types::TokenLogprobInfo],
) {
    if let GenerationResponseChannel::Stream(tx) = &response {
        // Decode only the latest token for incremental streaming
        let last_token = *output_tokens.last().unwrap();
        let text = engine.tokenizer.decode(&[last_token], true).unwrap_or_default();
        let logprob = output_logprobs.last().cloned();
        let _ = tx.send(StreamEvent::Token { text, logprobs: logprob });
    }
}

fn fail_response(response: GenerationResponseChannel, error: EngineError) {
    match response {
        GenerationResponseChannel::Complete(tx) => {
            let _ = tx.send(Err(error));
        }
        GenerationResponseChannel::Stream(tx) => {
            let _ = tx.send(StreamEvent::Error {
                message: error.to_string(),
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_response(
    response: GenerationResponseChannel,
    model: String,
    output_tokens: Vec<u32>,
    output_logprobs: Vec<crate::types::TokenLogprobInfo>,
    finish_reason: FinishReason,
    prompt_len: usize,
    gen_start: Instant,
    prefill_ms: f32,
    engine: &Engine,
    stop_strings: &[String],
) {
    let total_ms = gen_start.elapsed().as_secs_f32() * 1000.0;
    let decode_ms = total_ms - prefill_ms;
    let output_text = engine
        .tokenizer
        .decode(&output_tokens, true)
        .unwrap_or_default();
    let output_text = Engine::trim_stop_strings(&output_text, stop_strings);
    let completion_tokens = output_tokens.len() as u32;

    let result = GenerateResult {
        model,
        output_token_ids: output_tokens,
        output_text,
        finish_reason,
        usage: Usage {
            prompt_tokens: prompt_len as u32,
            completion_tokens,
            total_tokens: prompt_len as u32 + completion_tokens,
        },
        metrics: DecodeMetrics {
            total_ms,
            prefill_ms,
            decode_ms,
            ttft_ms: prefill_ms,
        },
        token_logprobs: if output_logprobs.is_empty() {
            None
        } else {
            Some(output_logprobs)
        },
        prompt_token_logprobs: None,
    };

    match response {
        GenerationResponseChannel::Complete(tx) => {
            let _ = tx.send(Ok(result));
        }
        GenerationResponseChannel::Stream(tx) => {
            let _ = tx.send(StreamEvent::Finished {
                finish_reason: result.finish_reason,
                usage: result.usage,
                metrics: result.metrics,
            });
        }
    }
}
