//! CUDA Executor: async GPU queue + CUDA graph replay.
//!
//! Implements the Executor trait with device-specific optimizations:
//! - Dedicated GPU worker thread (serialized forward passes)
//! - CUDA graph capture/replay for decode steps (Q=1)
//! - Non-blocking submit() enables double buffering in the AR loop

use std::sync::Arc;

use prelude_core::engine::executor::{ExecutionHandle, Executor, ForwardBatch, ModelOutput};
use prelude_core::engine::{Engine, EngineError};

struct CudaResult {
    output: Result<ModelOutput, EngineError>,
}

/// CUDA execution strategy: async GPU queue with CUDA graph support.
pub struct CudaExecutor {
    #[allow(dead_code)]
    engine: Arc<Engine>,
}

impl CudaExecutor {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self { engine }
    }
}

impl Executor for CudaExecutor {
    fn submit(
        &self,
        batch: ForwardBatch,
    ) -> Result<ExecutionHandle, EngineError> {
        // TODO: build paged attention tensors, submit to GPU worker thread
        let result = match batch {
            ForwardBatch::Prefill { items } => Err(EngineError::Unavailable(
                format!("CudaExecutor: prefill not yet connected ({} items)", items.len()),
            )),
            ForwardBatch::Decode { tokens, .. } => Err(EngineError::Unavailable(
                format!("CudaExecutor: decode not yet connected ({} tokens)", tokens.len()),
            )),
        };
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
