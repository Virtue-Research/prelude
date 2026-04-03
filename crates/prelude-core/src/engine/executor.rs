//! Device execution trait.
//!
//! `Executor` defines how scheduled batches are submitted to and collected from
//! a device. Device crates (prelude-cuda, prelude-cpu, ...) implement this trait
//! with device-specific optimizations:
//!
//! - **CudaExecutor**: async GPU queue + CUDA graph replay + double buffering
//! - **CpuExecutor**: synchronous `block_in_place` execution
//!
//! Core's scheduling loops (`engine/run/`) call `submit`/`collect` without knowing
//! which device is running. `submit()` is non-blocking on GPU (queues work and
//! returns immediately), so the scheduling loop naturally overlaps CPU preparation
//! with device execution — no separate batch/continuous variants per device.
//!
//! # Registration
//!
//! Device crates register their Executor factory via `ctor` at link time,
//! just like Ops registration:
//!
//! ```ignore
//! // prelude-cuda/src/lib.rs
//! #[ctor::ctor]
//! fn _register() {
//!     prelude_core::engine::register_executor(|| Box::new(CudaExecutor::new()));
//! }
//! ```

use std::sync::OnceLock;

use crate::tensor::Tensor;

use super::EngineError;

// ── Execution handle ───────────────────────────────────────────────

/// Opaque handle returned by `Executor::submit()`.
///
/// Represents an in-flight batch on the device. Call `Executor::collect()`
/// to await completion and retrieve the output.
pub struct ExecutionHandle {
    inner: Box<dyn std::any::Any + Send>,
}

impl ExecutionHandle {
    /// Create a new handle wrapping device-specific state.
    pub fn new<T: Send + 'static>(inner: T) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    /// Downcast to the device-specific handle type.
    pub fn downcast<T: 'static>(self) -> Option<T> {
        self.inner.downcast::<T>().ok().map(|b| *b)
    }
}

// ── Model output ───────────────────────────────────────────────────

/// Output from a single forward pass.
///
/// Contains the raw logits tensor. Sampling and postprocessing
/// happen in the scheduling loop (core), not in the Executor.
#[derive(Debug)]
pub struct ModelOutput {
    /// Raw logits: `[batch_size, vocab_size]` (for decode)
    /// or `[total_tokens, vocab_size]` (for prefill).
    pub logits: Tensor,
}

// ── Executor trait ─────────────────────────────────────────────────

/// Device execution strategy.
///
/// Implementors handle:
/// - Converting `ScheduledBatch` (logical) → device tensors (physical)
/// - Running `model.forward()` on the device
/// - Device-specific optimizations (GPU queue, CUDA graph, HIP graph, ...)
///
/// The trait is intentionally minimal — two methods. All scheduling paradigm
/// logic (AR, diffusion LLM, TTS, ...) lives in `engine/run/` and calls
/// these two methods uniformly.
pub trait Executor: Send + Sync + 'static {
    /// Submit a forward pass to the device. Returns immediately.
    ///
    /// - **GPU**: queues work to a dedicated GPU thread, returns a pending handle.
    /// - **CPU**: runs `model.forward()` synchronously, returns a completed handle.
    fn submit(
        &self,
        step: &crate::scheduler::SchedulerStep,
    ) -> Result<ExecutionHandle, EngineError>;

    /// Await completion of a submitted forward pass and retrieve output.
    ///
    /// - **GPU**: blocks until the GPU thread signals completion.
    /// - **CPU**: returns immediately (work already done in `submit`).
    fn collect(
        &self,
        handle: ExecutionHandle,
    ) -> Result<ModelOutput, EngineError>;
}

// ── Registration ───────────────────────────────────────────────────

type ExecutorFactory = fn() -> Box<dyn Executor>;

static EXECUTOR_FACTORY: OnceLock<ExecutorFactory> = OnceLock::new();

/// Register a device executor factory. Called by device crates via `ctor`.
///
/// Only the first registration wins (consistent with Ops registration).
pub fn register_executor(factory: ExecutorFactory) {
    EXECUTOR_FACTORY.set(factory).ok();
}

/// Create the registered executor, or return `None` if no device crate
/// registered one.
pub fn create_executor() -> Option<Box<dyn Executor>> {
    EXECUTOR_FACTORY.get().map(|f| f())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{SchedulerStep, ForwardMode};

    /// Mock result stored in the handle.
    struct MockResult(Tensor);

    /// A test executor that returns a fixed logits tensor.
    struct TestExecutor {
        vocab_size: usize,
    }

    impl Executor for TestExecutor {
        fn submit(
            &self,
            step: &SchedulerStep,
        ) -> Result<ExecutionHandle, EngineError> {
            let batch_size = step.prefill_request_ids.len() + step.decode_request_ids.len();
            let logits = Tensor::zeros(
                (batch_size, self.vocab_size),
                crate::tensor::DType::F32,
                &crate::tensor::Device::Cpu,
            ).map_err(|e| EngineError::Internal(format!("{e}")))?;
            Ok(ExecutionHandle::new(MockResult(logits)))
        }

        fn collect(
            &self,
            handle: ExecutionHandle,
        ) -> Result<ModelOutput, EngineError> {
            let result = handle.downcast::<MockResult>()
                .ok_or_else(|| EngineError::Internal("wrong handle type".into()))?;
            Ok(ModelOutput { logits: result.0 })
        }
    }

    #[test]
    fn submit_collect_roundtrip() {
        let executor = TestExecutor { vocab_size: 100 };
        let step = SchedulerStep::prefill(vec!["r1".into(), "r2".into()]);

        let handle = executor.submit(&step).unwrap();
        let output = executor.collect(handle).unwrap();

        assert_eq!(output.logits.dims(), &[2, 100]);
    }

    #[test]
    fn submit_decode_step() {
        let executor = TestExecutor { vocab_size: 50 };
        let step = SchedulerStep::decode(vec!["r1".into(), "r2".into(), "r3".into()]);

        let handle = executor.submit(&step).unwrap();
        let output = executor.collect(handle).unwrap();

        assert_eq!(output.logits.dims(), &[3, 50]);
    }

    #[test]
    fn submit_mixed_step() {
        let executor = TestExecutor { vocab_size: 32 };
        let step = SchedulerStep::mixed(
            vec!["p1".into()],
            vec!["d1".into(), "d2".into()],
        );

        let handle = executor.submit(&step).unwrap();
        let output = executor.collect(handle).unwrap();

        // 1 prefill + 2 decode = 3 total
        assert_eq!(output.logits.dims(), &[3, 32]);
    }

    #[test]
    fn collect_wrong_handle_type() {
        let executor = TestExecutor { vocab_size: 10 };

        // Create a handle with the wrong inner type
        let bad_handle = ExecutionHandle::new(42u32);
        let result = executor.collect(bad_handle);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("wrong handle type"));
    }

    #[test]
    fn handle_downcast_success() {
        let handle = ExecutionHandle::new(String::from("hello"));
        let value = handle.downcast::<String>();
        assert_eq!(value, Some(String::from("hello")));
    }

    #[test]
    fn handle_downcast_wrong_type() {
        let handle = ExecutionHandle::new(42u32);
        let value = handle.downcast::<String>();
        assert!(value.is_none());
    }
}
