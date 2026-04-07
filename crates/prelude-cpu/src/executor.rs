//! CPU Executor: dedicated worker thread, same async pattern as GPU.
//!
//! Sends work to a worker thread via channel, returns an async oneshot handle.
//! Same architecture as CudaExecutor — the AR loop always does `handle.recv().await`.

use std::sync::Arc;

use prelude_core::engine::executor::{ExecutionHandle, Executor, ForwardBatch};
use prelude_core::engine::{Engine, EngineError};

struct CpuWork {
    batch: ForwardBatch,
    result_tx: tokio::sync::oneshot::Sender<Result<prelude_core::engine::executor::ModelOutput, EngineError>>,
}

pub struct CpuExecutor {
    work_tx: tokio::sync::mpsc::UnboundedSender<CpuWork>,
    _worker: std::thread::JoinHandle<()>,
}

impl CpuExecutor {
    pub fn new(engine: Arc<Engine>) -> Self {
        let (work_tx, mut work_rx) = tokio::sync::mpsc::unbounded_channel::<CpuWork>();
        let worker = std::thread::Builder::new()
            .name("cpu-executor".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("CPU executor worker runtime");
                rt.block_on(async {
                    while let Some(work) = work_rx.recv().await {
                        let result = engine.forward_batch(work.batch);
                        let _ = work.result_tx.send(result);
                    }
                });
            })
            .expect("spawn CPU executor worker thread");
        Self { work_tx, _worker: worker }
    }
}

impl Executor for CpuExecutor {
    fn submit(
        &self,
        batch: ForwardBatch,
    ) -> Result<ExecutionHandle, EngineError> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        self.work_tx
            .send(CpuWork { batch, result_tx })
            .map_err(|_| EngineError::Internal("CPU worker thread has exited".into()))?;
        Ok(ExecutionHandle::new(result_rx))
    }
}
