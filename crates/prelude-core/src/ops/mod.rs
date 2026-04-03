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
