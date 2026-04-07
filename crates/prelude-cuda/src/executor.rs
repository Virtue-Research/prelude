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

struct GpuWork {
    batch: ForwardBatch,
    result_tx: tokio::sync::oneshot::Sender<Result<ModelOutput, EngineError>>,
}

/// CUDA execution strategy: dedicated GPU worker thread.
pub struct CudaExecutor {
    engine: Arc<Engine>,
    work_tx: tokio::sync::mpsc::UnboundedSender<GpuWork>,
    _worker: std::thread::JoinHandle<()>,
}

impl CudaExecutor {
    pub fn new(engine: Arc<Engine>) -> Self {
        let (work_tx, mut work_rx) = tokio::sync::mpsc::unbounded_channel::<GpuWork>();
        let worker_engine = engine.clone();
        let worker = std::thread::Builder::new()
            .name("gpu-executor".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("GPU executor worker runtime");
                rt.block_on(async {
                    while let Some(work) = work_rx.recv().await {
                        let result = worker_engine.forward_batch(work.batch);
                        let _ = work.result_tx.send(result);
                    }
                });
                tracing::info!("GPU executor worker exited");
            })
            .expect("spawn GPU executor worker thread");
        Self { engine, work_tx, _worker: worker }
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
        Ok(ExecutionHandle::new(result_rx))
    }
}
