//! CUDA Executor: async GPU queue + CUDA graph replay.
//!
//! Implements the Executor trait with device-specific optimizations:
//! - Dedicated GPU worker thread (serialized forward passes)
//! - CUDA graph capture/replay for decode steps (Q=1)
//! - Non-blocking submit() enables double buffering in the AR loop
//!
//! The full implementation will absorb the logic currently in:
//! - `runtime/gpu_queue.rs` (GPU work queue)
//! - `runtime/cuda_graph.rs` (graph capture/replay)
//! - `runtime/gpu_batch.rs` + `runtime/gpu_continuous.rs` (tensor preparation)

use prelude_core::engine::executor::{ExecutionHandle, Executor, ModelOutput};
use prelude_core::engine::EngineError;
use prelude_core::scheduler::SchedulerStep;

/// Pending GPU result — will hold a oneshot receiver when connected to GPU queue.
struct CudaResult {
    output: Result<ModelOutput, EngineError>,
}

/// CUDA execution strategy: async GPU queue with CUDA graph support.
///
/// When fully implemented, this will own:
/// - A GPU worker thread (dedicated OS thread for model.forward())
/// - A CUDA graph cache (keyed by batch_size for decode steps)
/// - A channel to submit work to the GPU thread
pub struct CudaExecutor;

impl CudaExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl Executor for CudaExecutor {
    fn submit(
        &self,
        _step: &SchedulerStep,
    ) -> Result<ExecutionHandle, EngineError> {
        // TODO: build paged attention tensors (block_tables, cu_seqlens, slot_mapping)
        // from SchedulerStep, submit to GPU worker thread via channel.
        // For now, return a placeholder error.
        let result = Err(EngineError::Unavailable(
            "CudaExecutor: not yet connected to GPU queue".into(),
        ));
        Ok(ExecutionHandle::new(CudaResult { output: result }))
    }

    fn collect(
        &self,
        handle: ExecutionHandle,
    ) -> Result<ModelOutput, EngineError> {
        let result = handle
            .downcast::<CudaResult>()
            .ok_or_else(|| EngineError::Internal("CudaExecutor: invalid handle type".into()))?;
        result.output
    }
}
