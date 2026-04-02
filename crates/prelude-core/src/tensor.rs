//! Prelude Tensor abstraction layer.
//!
//! This module is the ONLY place in prelude-core that imports candle_core types.
//! All model code, ops, and engine code import from here — never from candle_core.
//!
//! Current state: re-exports from candle_core (transition layer).
//! Next step: newtype wrapper with design doc API (view ops only, no compute methods).
//! Final state: own implementation with DeviceBuffer (candle_core fully removed).

// ── Core types ──────────────────────────────────────────────────────

pub use candle_core::Tensor;
pub use candle_core::DType;
pub use candle_core::Device;
pub use candle_core::Shape;
pub use candle_core::D;

// ── Error handling ──────────────────────────────────────────────────

pub use candle_core::Error;
pub use candle_core::Result;

/// Re-export candle's bail! macro under our namespace.
pub use candle_core::bail;

// ── Traits ──────────────────────────────────────────────────────────

pub use candle_core::Module;

// ── Backend internals (for device crates only, not model code) ──────

pub mod backend {
    pub use candle_core::backend::BackendStorage;
    pub use candle_core::Storage;
    pub use candle_core::CpuStorage;
    pub use candle_core::op::BackpropOp;
    pub use candle_core::Layout;
    #[cfg(feature = "cuda")]
    pub use candle_core::CudaStorage;
    #[cfg(feature = "cuda")]
    pub use candle_core::cuda_backend;
}

// ── Shape helpers ──────────────────────────────────────────────────

pub mod shape {
    pub use candle_core::shape::Dim;
}

// ── Safetensors I/O ────────────────────────────────────────────────

pub mod safetensors {
    pub use candle_core::safetensors::*;
}

// ── Quantization (kept for GGUF loading) ────────────────────────────

pub mod quantized {
    pub use candle_core::quantized::*;
}
