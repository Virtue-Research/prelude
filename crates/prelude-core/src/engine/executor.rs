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
