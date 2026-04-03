//! CUDA device implementation for Prelude inference engine.
//!
//! This crate owns:
//! - PTX kernel compilation (build.rs) and runtime loading constants
//! - GPU kernel wrappers (ops/) and attention backends (attn/)
//! - CudaOps: implements all Ops traits from prelude-core
//! - Feature-gated re-exports of kernel sub-crates (deepgemm, cutlass-gemm, etc.)

// ── PTX modules compiled from src/kernels/ ──────────────────────────

pub const PTX_ADD: &str = include_str!(concat!(env!("OUT_DIR"), "/add.ptx"));
pub const PTX_SILU_MUL: &str = include_str!(concat!(env!("OUT_DIR"), "/silu_mul.ptx"));
pub const PTX_RMSNORM: &str = include_str!(concat!(env!("OUT_DIR"), "/rmsnorm.ptx"));
pub const PTX_ADD_RMSNORM: &str = include_str!(concat!(env!("OUT_DIR"), "/add_rmsnorm.ptx"));
pub const PTX_QKNORM_ROPE: &str = include_str!(concat!(env!("OUT_DIR"), "/qknorm_rope.ptx"));
pub const PTX_MOE_ROUTING: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_routing.ptx"));
pub const PTX_MOE_GATEUP: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_gateup.ptx"));
pub const PTX_MOE_DOWN: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_down.ptx"));
pub const PTX_KV_APPEND: &str = include_str!(concat!(env!("OUT_DIR"), "/kv_append.ptx"));
pub const PTX_KNORM_ROPE_KV_WRITE: &str =
    include_str!(concat!(env!("OUT_DIR"), "/knorm_rope_kv_write.ptx"));
pub const PTX_SCATTER_KV_CACHE: &str =
    include_str!(concat!(env!("OUT_DIR"), "/scatter_kv_cache.ptx"));

// ── Module names for cudarc caching ─────────────────────────────────

pub const MOD_ADD: &str = "elementwise_add";
pub const MOD_SILU_MUL: &str = "elementwise_silu_mul";
pub const MOD_RMSNORM: &str = "normalization_rmsnorm";
pub const MOD_ADD_RMSNORM: &str = "normalization_add_rmsnorm";
pub const MOD_QKNORM_ROPE: &str = "rope_qknorm";
pub const MOD_MOE_ROUTING: &str = "moe_routing";
pub const MOD_MOE_GATEUP: &str = "moe_gateup";
pub const MOD_MOE_DOWN: &str = "moe_down";
pub const MOD_KV_APPEND: &str = "kvcache_kv_append";
pub const MOD_KNORM_ROPE_KV_WRITE: &str = "kvcache_knorm_rope_kv_write";
pub const MOD_SCATTER_KV_CACHE: &str = "kvcache_scatter_kv_cache";

// ── Auto-register GPU ops at link time ──────────────────────────────

#[ctor::ctor]
fn _register_gpu_ops() {
    prelude_core::ops::register_gpu_ops(cuda_ops::cuda_ops);
}

// ── GPU kernel modules ──────────────────────────────────────────────

pub(crate) mod ops;
pub(crate) mod attn;
mod cuda_ops;

pub use cuda_ops::{create_cuda_ops, cuda_ops};

// ── Sub-crate re-exports ────────────────────────────────────────────

#[cfg(feature = "deepgemm")]
pub use prelude_deepgemm;

#[cfg(feature = "cutlass-gemm")]
pub use prelude_cutlass_gemm;

#[cfg(feature = "flash-attn-v4")]
pub use prelude_flash_attn_v4;

#[cfg(feature = "flashinfer")]
pub use prelude_flashinfer;

#[cfg(feature = "quant-gemm")]
pub use prelude_quant_gemm;

#[cfg(feature = "cula")]
pub use prelude_cula;
