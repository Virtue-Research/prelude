//! Unified GPU work queue.
//!
//! All GPU-bound work is serialized through a single FIFO queue consumed by one
//! dedicated OS thread.  Schedulers (batch runtime, continuous runtime) produce
//! [`GpuPacket`]s and send them via `GpuQueueTx`; the worker executes them
//! sequentially and returns results through embedded oneshot channels.
//!
//! The worker runs on its own OS thread (not a tokio task) to avoid
//! `spawn_blocking` overhead per operation — critical for decode steps where
//! per-token GPU time is only a few milliseconds.
//!
//! **Design principle**: the GPU worker does ONLY GPU-bound work (model.forward).
//! All CPU post-processing (to_dtype, to_vec2, argmax, logprob extraction,
//! tokenizer decode, result construction) happens outside the GPU worker —
//! either in the batch runtime or via `spawn_blocking`.  This ensures the GPU
//! is never idle waiting on CPU work.

use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};

use crate::engine::{Engine, EngineError, PreTokenizedClassifyItem, PreTokenizedEmbedItem, PreparedGenerateRequest};
use crate::engine::{RawClassifyOutput, RawEmbedOutput};

#[cfg(feature = "flash-attn-v3")]
use crate::engine::RawGenerateOutput;

#[cfg(feature = "flash-attn-v3")]
use candle_core::Tensor;
#[cfg(feature = "flash-attn-v3")]
use crate::engine::{BatchPrefillResult, OwnedBatchDecodeSeq, PrefillPlan};

// ---------------------------------------------------------------------------
// GPU result type alias (cfg-gated, matches batch_runtime::GenGpuResult)
// ---------------------------------------------------------------------------

#[cfg(feature = "flash-attn-v3")]
type GenGpuResult = RawGenerateOutput;
#[cfg(not(feature = "flash-attn-v3"))]
type GenGpuResult = Vec<crate::types::GenerateResult>;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Sender half of the GPU queue — cloned into each runtime that produces work.
pub(crate) type GpuQueueTx = mpsc::UnboundedSender<GpuPacket>;

/// A unit of GPU work.  The variant fully describes what the worker should
/// execute; the model is a dumb executor.
///
/// Classify/Embed/Generate packets return raw GPU outputs (tensors + metadata).
/// CPU post-processing is done by the caller after receiving the raw output.
pub(crate) enum GpuPacket {
    /// Batch generation: prefill_pipeline → raw logits tensor.
    /// CPU post-processing (argmax, logprobs, decode) done by batch runtime.
    GenerateBatch {
        prepared_requests: Vec<PreparedGenerateRequest>,
        result_tx: oneshot::Sender<Result<GenGpuResult, EngineError>>,
    },

    /// Paged prefill: varlen forward + write KV to paged cache.
    /// Used by the continuous runtime for multi-token generation prefill.
    #[cfg(feature = "flash-attn-v3")]
    PrefillPaged {
        items: Vec<PreparedGenerateRequest>,
        prefill_plan: PrefillPlan,
        result_tx: oneshot::Sender<
            Result<(Vec<PreparedGenerateRequest>, Vec<BatchPrefillResult>), EngineError>,
        >,
    },

    /// Batched paged decode: Q=1 per sequence, read/write paged KV cache.
    /// Used by the continuous runtime for autoregressive decode steps.
    #[cfg(feature = "flash-attn-v3")]
    DecodePaged {
        seqs: Vec<OwnedBatchDecodeSeq>,
        result_tx: oneshot::Sender<Result<Tensor, EngineError>>,
    },

    /// Batch classification: prefill_pipeline → raw tensor + metadata.
    ClassifyBatch {
        items: Vec<PreTokenizedClassifyItem>,
        result_tx: oneshot::Sender<Result<RawClassifyOutput, EngineError>>,
    },

    /// Batch embedding: prefill_pipeline → raw tensor + metadata.
    EmbedBatch {
        items: Vec<PreTokenizedEmbedItem>,
        result_tx: oneshot::Sender<Result<RawEmbedOutput, EngineError>>,
    },
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Spawns the GPU worker on a dedicated OS thread.  Returns a JoinHandle that
/// can be stored to keep the thread alive.  The worker exits when all senders
/// are dropped (channel closed).
pub(crate) fn spawn_gpu_worker(
    rx: mpsc::UnboundedReceiver<GpuPacket>,
    engine: Arc<Engine>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("gpu-worker".into())
        .spawn(move || {
            // Tiny single-threaded tokio runtime just for channel recv.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("gpu worker runtime");
            rt.block_on(gpu_worker_loop(rx, engine));
        })
        .expect("spawn gpu worker thread")
}

/// Core worker loop — runs on a dedicated OS thread.
async fn gpu_worker_loop(
    mut rx: mpsc::UnboundedReceiver<GpuPacket>,
    engine: Arc<Engine>,
) {
    tracing::info!("GPU worker started");

    while let Some(packet) = rx.recv().await {
        execute_gpu_packet(&engine, packet);
    }

    tracing::info!("GPU worker exited");
}

/// Execute a single GPU packet synchronously.
///
/// This is the unified entry point for all GPU-bound work.  The packet
/// variant fully describes the execution mode.
///
/// **Only GPU work happens here.** CPU post-processing is NOT done here —
/// the raw GPU output is returned via the oneshot channel.
fn execute_gpu_packet(engine: &Engine, packet: GpuPacket) {
    match packet {
        GpuPacket::GenerateBatch {
            prepared_requests,
            result_tx,
        } => {
            let result = engine.plan_generate_batch(prepared_requests).and_then(|planned| {
                #[cfg(feature = "flash-attn-v3")]
                { engine.prefill_forward_only(planned.items) }
                #[cfg(not(feature = "flash-attn-v3"))]
                { engine.generate_prepared_batch(planned) }
            });
            let _ = result_tx.send(result);
        }

        #[cfg(feature = "flash-attn-v3")]
        GpuPacket::PrefillPaged {
            mut items,
            prefill_plan,
            result_tx,
        } => {
            let result = engine
                .batch_prefill_paged(&mut items, &prefill_plan)
                .map(|r| (items, r));
            let _ = result_tx.send(result);
        }

        #[cfg(feature = "flash-attn-v3")]
        GpuPacket::DecodePaged { seqs, result_tx } => {
            let borrowed: Vec<_> = seqs.iter().map(|s| s.as_borrowed()).collect();
            let result = engine.batch_decode_paged(&borrowed);
            let _ = result_tx.send(result);
        }

        GpuPacket::ClassifyBatch { items, result_tx } => {
            let _ = result_tx.send(engine.classify_forward_only(items));
        }

        GpuPacket::EmbedBatch { items, result_tx } => {
            let _ = result_tx.send(engine.embed_forward_only(items));
        }
    }
}

// ---------------------------------------------------------------------------
// Submit helpers
// ---------------------------------------------------------------------------

/// Send a packet to the GPU queue and return a JoinHandle for the result.
fn submit_and_spawn<R: Send + 'static>(
    gpu_tx: &GpuQueueTx,
    make_packet: impl FnOnce(oneshot::Sender<Result<R, EngineError>>) -> GpuPacket,
) -> tokio::task::JoinHandle<Result<R, EngineError>> {
    let (result_tx, result_rx) = oneshot::channel();
    let _ = gpu_tx.send(make_packet(result_tx));
    tokio::spawn(async move {
        result_rx
            .await
            .unwrap_or_else(|_| Err(EngineError::Internal("GPU queue dropped".into())))
    })
}

pub(crate) fn submit_generate_batch(
    gpu_tx: &GpuQueueTx,
    prepared_requests: Vec<PreparedGenerateRequest>,
) -> tokio::task::JoinHandle<Result<GenGpuResult, EngineError>> {
    submit_and_spawn(gpu_tx, |tx| GpuPacket::GenerateBatch {
        prepared_requests,
        result_tx: tx,
    })
}

pub(crate) fn submit_classify_batch(
    gpu_tx: &GpuQueueTx,
    items: Vec<PreTokenizedClassifyItem>,
) -> tokio::task::JoinHandle<Result<RawClassifyOutput, EngineError>> {
    submit_and_spawn(gpu_tx, |tx| GpuPacket::ClassifyBatch {
        items,
        result_tx: tx,
    })
}

pub(crate) fn submit_embed_batch(
    gpu_tx: &GpuQueueTx,
    items: Vec<PreTokenizedEmbedItem>,
) -> tokio::task::JoinHandle<Result<RawEmbedOutput, EngineError>> {
    submit_and_spawn(gpu_tx, |tx| GpuPacket::EmbedBatch {
        items,
        result_tx: tx,
    })
}

/// Sends a `PrefillPaged` packet — awaited directly (no JoinHandle).
#[cfg(feature = "flash-attn-v3")]
pub(crate) async fn submit_prefill_paged(
    gpu_tx: &GpuQueueTx,
    items: Vec<PreparedGenerateRequest>,
    prefill_plan: PrefillPlan,
) -> Result<(Vec<PreparedGenerateRequest>, Vec<BatchPrefillResult>), EngineError> {
    let (result_tx, result_rx) = oneshot::channel();
    let _ = gpu_tx.send(GpuPacket::PrefillPaged {
        items,
        prefill_plan,
        result_tx,
    });
    result_rx
        .await
        .unwrap_or_else(|_| Err(EngineError::Internal("GPU queue dropped".into())))
}

/// Sends a `DecodePaged` packet — awaited directly (no JoinHandle).
#[cfg(feature = "flash-attn-v3")]
pub(crate) async fn submit_decode_paged(
    gpu_tx: &GpuQueueTx,
    seqs: Vec<OwnedBatchDecodeSeq>,
) -> Result<Tensor, EngineError> {
    let (result_tx, result_rx) = oneshot::channel();
    let _ = gpu_tx.send(GpuPacket::DecodePaged { seqs, result_tx });
    result_rx
        .await
        .unwrap_or_else(|_| Err(EngineError::Internal("GPU queue dropped".into())))
}
