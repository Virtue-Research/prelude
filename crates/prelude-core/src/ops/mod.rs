//! Kernel operations — trait definitions and device ops registry.
//!
//! - `traits`: Op trait definitions (AttentionOps, GemmOps, NormOps, FusedOps, ...)
//! - `fallback_ops`: Built-in basic CPU ops using tensor abstraction (always available)
//! - Device crates (prelude-cpu, prelude-cuda) register optimized implementations
//!   via `#[ctor::ctor]` at link time, overriding the fallback.

pub mod traits;
mod naive_ops;

// Re-export the Ops bundle and all trait types at crate level for convenience.
pub use traits::*;

// ── Device ops registry ────────────────────────────────────────────

use std::sync::OnceLock;
use crate::tensor::Device;

type OpsFactory = fn() -> &'static Ops;
static CPU_OPS_FACTORY: OnceLock<OpsFactory> = OnceLock::new();
static GPU_OPS_FACTORY: OnceLock<OpsFactory> = OnceLock::new();

/// Register optimized CPU ops. Called by prelude-cpu via `#[ctor::ctor]`.
pub fn register_cpu_ops(factory: OpsFactory) {
    CPU_OPS_FACTORY.set(factory).ok();
}

/// Register GPU ops. Called by prelude-cuda via `#[ctor::ctor]`.
pub fn register_gpu_ops(factory: OpsFactory) {
    GPU_OPS_FACTORY.set(factory).ok();
}

/// Auto-select ops based on device.
///
/// Priority: GPU ops → optimized CPU ops → built-in fallback.
pub fn select_ops(device: &Device) -> &'static Ops {
    if device.is_cuda() {
        if let Some(factory) = GPU_OPS_FACTORY.get() {
            return factory();
        }
        tracing::warn!("CUDA device selected but no GPU ops registered — falling back to CPU ops");
    }
    if let Some(factory) = CPU_OPS_FACTORY.get() {
        return factory();
    }
    naive_ops::naive_ops()
}

// ── Thread-local Ops context ──────────────────────────────────────

use std::cell::Cell;

thread_local! {
    static THREAD_OPS: Cell<Option<&'static Ops>> = const { Cell::new(None) };
}

/// Get the current thread-local Ops, or the naive fallback.
///
/// Used by operator overloads (`a + b`) to dispatch through the correct backend.
/// Set via [`with_ops`] at the start of a forward pass.
pub fn current_ops() -> &'static Ops {
    THREAD_OPS.with(|cell| cell.get().unwrap_or_else(|| naive_ops::naive_ops()))
}

/// Execute `f` with the given Ops set as the thread-local context.
///
/// Called by the engine at the start of each forward pass so that
/// operator overloads on Tensor can dispatch through the right backend.
pub fn with_ops<R>(ops: &'static Ops, f: impl FnOnce() -> R) -> R {
    THREAD_OPS.with(|cell| {
        let prev = cell.replace(Some(ops));
        let result = f();
        cell.set(prev);
        result
    })
}

/// Execute `f` within a full forward-pass scope: sets thread-local Ops,
/// calls `session.begin_forward()`, runs `f`, calls `session.end_forward()`.
pub fn forward_scope<R>(ops: &'static Ops, f: impl FnOnce() -> R) -> R {
    with_ops(ops, || {
        ops.session.begin_forward();
        let result = f();
        ops.session.end_forward();
        result
    })
}

/// RAII guard that sets thread-local Ops on creation and restores on drop.
pub struct OpsGuard {
    prev: Option<&'static Ops>,
}

impl OpsGuard {
    pub fn new(ops: &'static Ops) -> Self {
        let prev = THREAD_OPS.with(|cell| cell.replace(Some(ops)));
        Self { prev }
    }
}

impl Drop for OpsGuard {
    fn drop(&mut self) {
        THREAD_OPS.with(|cell| cell.set(self.prev));
    }
}
