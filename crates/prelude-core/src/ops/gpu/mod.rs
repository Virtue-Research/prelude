//! Fused CUDA kernels for BF16 element-wise operations.
//!
//! These replace candle's default element-wise kernels which process 1 BF16/thread
//! with vectorized versions that process 8 BF16/thread via 128-bit loads.
//! Kernels are compiled to PTX at build time and loaded at runtime via cudarc.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, DType, Layout, Result, Shape, Tensor};

// PTX modules - each compiled from separate .cu files for maintainability
pub(crate) const PTX_ADD: &str = include_str!(concat!(env!("OUT_DIR"), "/add.ptx"));
pub(crate) const PTX_SILU_MUL: &str = include_str!(concat!(env!("OUT_DIR"), "/silu_mul.ptx"));
pub(crate) const PTX_RMSNORM: &str = include_str!(concat!(env!("OUT_DIR"), "/rmsnorm.ptx"));
pub(crate) const PTX_ADD_RMSNORM: &str =
    include_str!(concat!(env!("OUT_DIR"), "/add_rmsnorm.ptx"));
pub(crate) const PTX_QKNORM_ROPE: &str =
    include_str!(concat!(env!("OUT_DIR"), "/qknorm_rope.ptx"));
pub(crate) const PTX_MOE_ROUTING: &str =
    include_str!(concat!(env!("OUT_DIR"), "/moe_routing.ptx"));
pub(crate) const PTX_KNORM_ROPE_KV_WRITE: &str =
    include_str!(concat!(env!("OUT_DIR"), "/knorm_rope_kv_write.ptx"));
pub(crate) const PTX_SCATTER_KV_CACHE: &str =
    include_str!(concat!(env!("OUT_DIR"), "/scatter_kv_cache.ptx"));
// PTX dequantize moved to prelude-quant-gemm (FFI static library)

// Module names for cudarc caching
pub(crate) const MOD_ADD: &str = "elementwise_add";
pub(crate) const MOD_SILU_MUL: &str = "elementwise_silu_mul";
pub(crate) const MOD_RMSNORM: &str = "normalization_rmsnorm";
pub(crate) const MOD_ADD_RMSNORM: &str = "normalization_add_rmsnorm";
pub(crate) const MOD_QKNORM_ROPE: &str = "rope_qknorm";
pub(crate) const MOD_MOE_ROUTING: &str = "moe_routing";
pub(crate) const MOD_KNORM_ROPE_KV_WRITE: &str = "kvcache_knorm_rope_kv_write";
pub(crate) const MOD_SCATTER_KV_CACHE: &str = "kvcache_scatter_kv_cache";
// MOD_DEQUANTIZE moved to prelude-quant-gemm

pub mod elementwise;
pub mod gemm;
pub mod rmsnorm;
pub mod rope;
pub mod moe;
pub mod kv_cache;
// Quant ops (dequantize, MMVQ, tiled MMQ) moved to prelude-quant-gemm.
// prelude-core re-exports via ops::gpu::quant for backward compat.
#[cfg(feature = "quant-gemm")]
pub mod quant;
#[cfg(feature = "quant-gemm")]
pub mod tiled_mmq;

pub use self::elementwise::*;
pub use self::rmsnorm::*;
pub use self::rope::*;
pub use self::moe::*;
pub use self::kv_cache::*;
