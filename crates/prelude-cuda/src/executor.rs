//! CUDA Executor: async GPU queue + CUDA graph replay.
//!
//! The CudaExecutor owns a dedicated GPU worker thread. `submit()` sends
//! work to this thread via a channel (non-blocking), enabling the AR loop
//! to overlap CPU preparation with GPU execution. `collect()` blocks until
//! the GPU thread completes the forward pass.
//!
//! TODO: migrate CUDA graph capture/replay from runtime/cuda_graph.rs.

use std::sync::Arc;

use prelude_core::engine::executor::{ExecutionHandle, Executor, ForwardBatch, ModelOutput};
use prelude_core::engine::{Engine, EngineError};
use prelude_core::tensor::Tensor;

// ── GPU work packet ────────────────────────────────────────────────

/// A unit of work sent to the GPU thread.
struct GpuWork {
    batch: ForwardBatch,
    result_tx: tokio::sync::oneshot::Sender<Result<ModelOutput, EngineError>>,
}

/// Handle holding the oneshot receiver for an in-flight GPU submission.
struct CudaPending {
    result_rx: tokio::sync::oneshot::Receiver<Result<ModelOutput, EngineError>>,
}

// ── CudaExecutor ───────────────────────────────────────────────────

/// CUDA execution strategy: dedicated GPU worker thread.
pub struct CudaExecutor {
    engine: Arc<Engine>,
    work_tx: tokio::sync::mpsc::UnboundedSender<GpuWork>,
    // The worker thread handle — kept alive for the executor's lifetime.
    _worker: std::thread::JoinHandle<()>,
}

impl CudaExecutor {
    pub fn new(engine: Arc<Engine>) -> Self {
        let (work_tx, work_rx) = tokio::sync::mpsc::unbounded_channel();
        let worker_engine = engine.clone();
        let worker = std::thread::Builder::new()
            .name("gpu-executor".into())
            .spawn(move || {
                gpu_worker_loop(work_rx, worker_engine);
            })
            .expect("spawn GPU executor worker thread");
        Self {
            engine,
            work_tx,
            _worker: worker,
        }
    }
}

impl Executor for CudaExecutor {
    fn submit(
        &self,
        batch: ForwardBatch,
    ) -> Result<ExecutionHandle, EngineError> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        self.work_tx
            .send(GpuWork { batch, result_tx })
            .map_err(|_| EngineError::Internal("GPU worker thread has exited".into()))?;
        Ok(ExecutionHandle::new(CudaPending { result_rx }))
    }

    fn collect(
        &self,
        handle: ExecutionHandle,
    ) -> Result<ModelOutput, EngineError> {
        let pending = handle
            .downcast::<CudaPending>()
            .ok_or_else(|| EngineError::Internal("CudaExecutor: invalid handle type".into()))?;
        pending
            .result_rx
            .blocking_recv()
            .map_err(|_| EngineError::Internal("GPU worker dropped result channel".into()))?
    }
}

// ── GPU worker thread ──────────────────────────────────────────────

fn gpu_worker_loop(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<GpuWork>,
    engine: Arc<Engine>,
) {
    // Tiny single-threaded tokio runtime just for channel recv.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("GPU executor worker runtime");

    rt.block_on(async {
        while let Some(work) = rx.recv().await {
            let result = execute_forward(&engine, work.batch);
            let _ = work.result_tx.send(result);
        }
    });

    tracing::info!("GPU executor worker exited");
}

fn execute_forward(
    engine: &Engine,
    batch: ForwardBatch,
) -> Result<ModelOutput, EngineError> {
    match batch {
        ForwardBatch::Prefill { items } => {
            // TODO: use engine.prefill_forward_only() for varlen prefill
            // (returns raw logits, not post-sampled results).
            // For now, use the same path as CpuExecutor.
            engine
                .plan_generate_batch(items)
                .and_then(|planned| engine.generate_prepared_batch(planned))
                .map(|_results| {
                    // Placeholder: generate_prepared_batch returns GenerateResult,
                    // not raw logits. Full integration needs Engine forward/postprocess split.
                    ModelOutput {
                        logits: Tensor::zeros(
                            (0, 0),
                            prelude_core::tensor::DType::F32,
                            &prelude_core::tensor::Device::Cpu,
                        ).unwrap(),
                    }
                })
        }
        ForwardBatch::Decode { tokens, positions, block_tables } => {
            // TODO: build paged attention tensors (cu_seqlens, block_table tensor,
            // slot_mapping) and call engine model.forward() with paged KV cache.
            // Also: CUDA graph replay for eligible batch sizes.
            Err(EngineError::Unavailable(format!(
                "CudaExecutor: paged decode not yet connected ({} seqs)", tokens.len()
            )))
        }
    }
}
