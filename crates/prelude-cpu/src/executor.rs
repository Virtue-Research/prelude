//! CPU Executor: synchronous block_in_place execution.
//!
//! The simplest Executor implementation — runs model.forward() synchronously
//! on the calling thread via `tokio::task::block_in_place`. No GPU queue,
//! no graph capture, no double buffering.
//!
//! Since `submit()` completes synchronously, `collect()` always returns
//! immediately with the already-computed result.

use std::sync::Arc;

use prelude_core::engine::executor::{ExecutionHandle, Executor, ForwardBatch, ModelOutput};
use prelude_core::engine::{Engine, EngineError};

/// Completed result stored inside the ExecutionHandle.
struct CpuResult {
    output: Result<ModelOutput, EngineError>,
}

/// CPU execution strategy: synchronous, zero-overhead.
pub struct CpuExecutor {
    engine: Arc<Engine>,
}

impl CpuExecutor {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self { engine }
    }
}

impl Executor for CpuExecutor {
    fn submit(
        &self,
        batch: ForwardBatch,
    ) -> Result<ExecutionHandle, EngineError> {
        let result = self.engine.forward_batch(batch);
        Ok(ExecutionHandle::new(CpuResult { output: result }))
    }

    fn collect(
        &self,
        handle: ExecutionHandle,
    ) -> Result<ModelOutput, EngineError> {
        let result = handle
            .downcast::<CpuResult>()
            .ok_or_else(|| EngineError::Internal("CpuExecutor: invalid handle type".into()))?;
        result.output
    }
}
