//! GPU kernel operations — PTX kernel wrappers and sub-crate dispatch.
//!
//! Each module wraps CUDA kernels and exposes Rust functions.
//! CudaOps calls these; they are not public API.

pub(crate) mod elementwise;
pub(crate) mod gather_log_softmax;
pub(crate) mod gdn_post_conv;
pub(crate) mod gemm;
pub(crate) mod kv_cache;
pub(crate) mod moe;
pub(crate) mod rmsnorm;
pub(crate) mod rope;

pub(crate) mod sampling;
#[cfg(feature = "quant-gemm")]
pub(crate) mod tiled_mmq;
