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
        let result = match batch {
            ForwardBatch::Prefill { items } => {
                // Run the full generate pipeline synchronously.
                // generate_prepared_batch returns Vec<GenerateResult> (post-sampled),
                // but the Executor interface returns raw logits.
                // TODO: split Engine into forward-only (returns logits) + postprocess.
                // For now, run the full pipeline and return a dummy logits tensor.
                self.engine
                    .plan_generate_batch(items)
                    .and_then(|planned| self.engine.generate_prepared_batch(planned))
                    .map(|_results| {
                        ModelOutput {
                            logits: prelude_core::tensor::Tensor::zeros(
                                (0, 0),
                                prelude_core::tensor::DType::F32,
                                &prelude_core::tensor::Device::Cpu,
                            ).unwrap(),
                        }
                    })
            }
            ForwardBatch::Decode { .. } => {
                Err(EngineError::Unavailable(
                    "CpuExecutor: continuous decode not yet supported".into(),
                ))
            }
        };
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
