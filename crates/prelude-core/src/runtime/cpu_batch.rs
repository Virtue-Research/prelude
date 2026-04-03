//! CPU batch runtime.
//!
//! Simple synchronous loop for CPU inference — no GPU queue, no adaptive wait,
//! no tokenize pipeline.
//!
//! ```text
//! loop {
//!     requests = drain_channel()
//!     tokenize(requests)
//!     forward(requests)          // block_in_place, synchronous
//!     dispatch_results(results)
//! }
//! ```
//!
//! Forward execution naturally blocks new requests from being processed,
//! so requests accumulate in the channel during forward — the next iteration
//! picks them all up as a batch. No adaptive wait needed.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use crate::engine::EngineError;
use crate::engine::{Engine, PreparedGenerateRequest};
use crate::runtime::SchedulerConfig;
use crate::types::GenerateResult;

use crate::engine::{classify_postprocess, embed_postprocess};

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
use crate::engine::{RawGenerateOutput, generate_postprocess};

use super::batch_common::{
    collect_classify_batch, collect_embed_batch, dispatch_classify_results,
    dispatch_embed_results, dispatch_generate_results, tokenize_gen_batch,
};
use super::request_state::{
    GenerationRequestState, InFlightClassifyRequest, InFlightEmbedRequest, SchedulerMsg,
};

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

/// CPU batch runtime — synchronous, zero-overhead loop.
///
/// All work (tokenize + forward + post-process) runs on the current thread
/// via `block_in_place`. No GPU queue, no adaptive wait, no pipeline.
pub(crate) async fn cpu_batch_runtime_loop(
    engine: Arc<Engine>,
    config: SchedulerConfig,
    mut rx: mpsc::UnboundedReceiver<SchedulerMsg>,
) {
    let max_batch_size = config.max_batch_size.max(1);

    let mut gen_waiting: VecDeque<GenerationRequestState> = VecDeque::new();
    let mut classify_queue: VecDeque<InFlightClassifyRequest> = VecDeque::new();
    let mut embed_queue: VecDeque<InFlightEmbedRequest> = VecDeque::new();

    tracing::info!("CPU batch runtime started (synchronous, no GPU queue)");

    let mut iter_count: u64 = 0;

    loop {
        let iter_start = Instant::now();

        // ── Phase 1: Wait for at least one request ──────────────────────
        if gen_waiting.is_empty() && classify_queue.is_empty() && embed_queue.is_empty() {
            match rx.recv().await {
                Some(msg) => dispatch_msg(msg, &mut gen_waiting, &mut classify_queue, &mut embed_queue),
                None => break, // channel closed
            }
        }

        // ── Phase 2: Drain all pending requests (non-blocking) ──────────
        let drain_start = Instant::now();
        let pre_drain = gen_waiting.len();
        while let Ok(msg) = rx.try_recv() {
            dispatch_msg(msg, &mut gen_waiting, &mut classify_queue, &mut embed_queue);
        }
        let drained = gen_waiting.len() - pre_drain;
        let drain_us = drain_start.elapsed().as_micros();

        // ── Phase 3: Execute one task type synchronously ────────────────
        // Priority: generate > classify > embed (most common first)
        if !gen_waiting.is_empty() {
            let batch: Vec<_> = gen_waiting
                .drain(..gen_waiting.len().min(max_batch_size))
                .collect();
            let batch_size = batch.len();
            execute_generate_batch(&engine, batch);

            let iter_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
            iter_count += 1;
            tracing::info!(
                iter = iter_count,
                batch_size,
                drained,
                drain_us,
                iter_ms = format!("{iter_ms:.1}"),
                "cpu_loop iteration"
            );
        } else if !classify_queue.is_empty() {
            if let Some((inflight, items)) =
                collect_classify_batch(&engine, &mut classify_queue, max_batch_size)
            {
                execute_classify_batch(&engine, inflight, items);
            }
        } else if !embed_queue.is_empty() {
            if let Some((inflight, items)) =
                collect_embed_batch(&engine, &mut embed_queue, max_batch_size)
            {
                execute_embed_batch(&engine, inflight, items);
            }
        }
    }

    // Shutdown: fail remaining requests
    for req in gen_waiting {
        req.fail(EngineError::Unavailable("scheduler loop stopped".into()));
    }
    for req in classify_queue {
        let _ = req.response.send(Err(EngineError::Unavailable("scheduler loop stopped".into())));
    }
    for req in embed_queue {
        let _ = req.response.send(Err(EngineError::Unavailable("scheduler loop stopped".into())));
    }

    tracing::info!("CPU batch runtime exited");
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

fn execute_generate_batch(engine: &Arc<Engine>, batch: Vec<GenerationRequestState>) {
    let t0 = Instant::now();
    let batch_size = batch.len();
    let cached_count = batch.iter().filter(|r| r.cached_prepared.is_some()).count();

    let (inflight, prepared) = match tokenize_gen_batch(engine, batch) {
        Some(r) => r,
        None => return,
    };

    let tokenize_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let total_tokens: usize = prepared.iter().map(|p| p.prompt_tokens.len()).sum();

    let t1 = Instant::now();
    let result = tokio::task::block_in_place(|| execute_generate_forward(engine, prepared));
    let forward_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = Instant::now();
    dispatch_generate_results(inflight, result);
    let dispatch_ms = t2.elapsed().as_secs_f64() * 1000.0;

    tracing::info!(
        batch_size,
        cached_count,
        total_tokens,
        tokenize_ms = format!("{tokenize_ms:.1}"),
        forward_ms = format!("{forward_ms:.1}"),
        dispatch_ms = format!("{dispatch_ms:.1}"),
        "cpu_gen_batch"
    );
}

#[cfg(any(feature = "flash-attn-v4", feature = "flashinfer"))]
fn execute_generate_forward(
    engine: &Engine,
    prepared: Vec<PreparedGenerateRequest>,
) -> Result<Vec<GenerateResult>, EngineError> {
    let raw: RawGenerateOutput = engine
        .plan_generate_batch(prepared)
        .and_then(|planned| engine.prefill_forward_only(planned.items))?;
    generate_postprocess(raw, &engine.tokenizer, &engine.eos_token_ids)
}

#[cfg(not(any(feature = "flash-attn-v4", feature = "flashinfer")))]
fn execute_generate_forward(
    engine: &Engine,
    prepared: Vec<PreparedGenerateRequest>,
) -> Result<Vec<GenerateResult>, EngineError> {
    engine
        .plan_generate_batch(prepared)
        .and_then(|planned| engine.generate_prepared_batch(planned))
}

// ---------------------------------------------------------------------------
// Classify
// ---------------------------------------------------------------------------

fn execute_classify_batch(
    engine: &Arc<Engine>,
    inflight: Vec<InFlightClassifyRequest>,
    items: Vec<crate::engine::PreTokenizedClassifyItem>,
) {
    let result = tokio::task::block_in_place(|| {
        engine.classify_forward_only(items).and_then(classify_postprocess)
    });

    dispatch_classify_results(inflight, result);
}

// ---------------------------------------------------------------------------
// Embed
// ---------------------------------------------------------------------------

fn execute_embed_batch(
    engine: &Arc<Engine>,
    inflight: Vec<InFlightEmbedRequest>,
    items: Vec<crate::engine::PreTokenizedEmbedItem>,
) {
    let result = tokio::task::block_in_place(|| {
        engine.embed_forward_only(items).and_then(embed_postprocess)
    });

    dispatch_embed_results(inflight, result);
}

// ---------------------------------------------------------------------------
// Helpers (CPU-specific)
// ---------------------------------------------------------------------------

fn dispatch_msg(
    msg: SchedulerMsg,
    gen_waiting: &mut VecDeque<GenerationRequestState>,
    classify_queue: &mut VecDeque<InFlightClassifyRequest>,
    embed_queue: &mut VecDeque<InFlightEmbedRequest>,
) {
    match msg {
        SchedulerMsg::NewRequest(req) => gen_waiting.push_back(req),
        SchedulerMsg::NewClassifyRequest(req) => classify_queue.push_back(req),
        SchedulerMsg::NewEmbedRequest(req) => embed_queue.push_back(req),
        SchedulerMsg::Abort(id) => {
            // Remove from waiting queue if present
            if let Some(pos) = gen_waiting.iter().position(|r| r.request_id() == id) {
                if let Some(req) = gen_waiting.remove(pos) {
                    req.abort();
                }
            }
        }
    }
}
