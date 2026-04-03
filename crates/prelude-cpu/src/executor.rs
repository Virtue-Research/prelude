//! CPU Executor: synchronous block_in_place execution.
//!
//! The simplest Executor implementation — runs model.forward() synchronously
//! on the calling thread via `tokio::task::block_in_place`. No GPU queue,
//! no graph capture, no double buffering.
//!
//! Since `submit()` completes synchronously, `collect()` always returns
//! immediately with the already-computed result.

use prelude_core::engine::executor::{ExecutionHandle, Executor, ModelOutput};
use prelude_core::engine::EngineError;
use prelude_core::scheduler::SchedulerStep;

/// Completed result stored inside the ExecutionHandle.
struct CpuResult {
    output: Result<ModelOutput, EngineError>,
}

/// CPU execution strategy: synchronous, zero-overhead.
pub struct CpuExecutor;

impl CpuExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl Executor for CpuExecutor {
    fn submit(
        &self,
        _step: &SchedulerStep,
    ) -> Result<ExecutionHandle, EngineError> {
        // TODO: build tensors from SchedulerStep, call model.forward() via block_in_place
        // For now, return a placeholder error since the full tensor preparation
        // logic is still in runtime/cpu_batch.rs
        let result = Err(EngineError::Unavailable(
            "CpuExecutor: not yet connected to model forward".into(),
        ));
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
