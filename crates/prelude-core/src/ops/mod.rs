//! Kernel operations — unified `Ops` trait + two parallel backends.
//!
//! - `traits/ops.rs`: The unified `Ops` trait — one trait, override any method.
//! - `cubecl_backend/`: CubeCL runtime backend (Storage::CubeCL).
//! - `device_backend/`: Pure Rust backend (Storage::Device).
//! - Device crates register optimized overrides via `register()` at startup.

pub mod traits;
pub mod cubecl_backend;
pub mod device_backend;

pub use traits::*;

// ── Device ops registry ────────────────────────────────────────────

use std::sync::OnceLock;
use crate::tensor::Device;

type OpsFactory = fn() -> &'static dyn Ops;
static CPU_OPS_FACTORY: OnceLock<OpsFactory> = OnceLock::new();
static GPU_OPS_FACTORY: OnceLock<OpsFactory> = OnceLock::new();

pub fn register_cpu_ops(factory: OpsFactory) {
    CPU_OPS_FACTORY.set(factory).ok();
}

pub fn register_gpu_ops(factory: OpsFactory) {
    GPU_OPS_FACTORY.set(factory).ok();
}

pub fn select_ops(device: &Device) -> &'static dyn Ops {
    if device.is_cuda() {
        if let Some(factory) = GPU_OPS_FACTORY.get() {
            return factory();
        }
        tracing::warn!("CUDA device selected but no GPU ops registered — falling back to CPU ops");
    }
    if let Some(factory) = CPU_OPS_FACTORY.get() {
        return factory();
    }
    bare_ops()
}

/// Device-aware ops: thread-local first, then device-based fallback.
pub fn ops_for(device: &Device) -> &'static dyn Ops {
    THREAD_OPS.with(|cell| {
        if let Some(ops) = cell.get() {
            return ops;
        }
        select_ops(device)
    })
}

// ── Thread-local ops context ─────────────────────────────────────

use std::cell::Cell;

thread_local! {
    static THREAD_OPS: Cell<Option<&'static dyn Ops>> = const { Cell::new(None) };
}

/// Bare primitives (Device or CubeCL), WITHOUT device crate overlay.
/// Used by device crates' `default_impl()` to avoid recursion.
pub fn bare_ops() -> &'static dyn Ops {
    use std::sync::LazyLock;
    // Default: Device backend. Set PRELUDE_TENSOR_BACKEND=cubecl to use CubeCL.
    static USE_CUBECL: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("PRELUDE_TENSOR_BACKEND")
            .map(|v| v == "cubecl")
            .unwrap_or(false)
    });
    if *USE_CUBECL {
        cubecl_backend::cubecl_ops()
    } else {
        device_backend::device_ops()
    }
}

pub fn with_ops<R>(ops: &'static dyn Ops, f: impl FnOnce() -> R) -> R {
    THREAD_OPS.with(|cell| {
        let prev = cell.replace(Some(ops));
        let result = f();
        cell.set(prev);
        result
    })
}

pub fn forward_scope<R>(ops: &'static dyn Ops, f: impl FnOnce() -> R) -> R {
    with_ops(ops, || {
        ops.begin_forward();
        let result = f();
        ops.end_forward();
        result
    })
}

pub struct OpsGuard {
    prev: Option<&'static dyn Ops>,
}

impl OpsGuard {
    pub fn new(ops: &'static dyn Ops) -> Self {
        let prev = THREAD_OPS.with(|cell| cell.replace(Some(ops)));
        Self { prev }
    }
}

impl Drop for OpsGuard {
    fn drop(&mut self) {
        THREAD_OPS.with(|cell| cell.set(self.prev));
    }
}
