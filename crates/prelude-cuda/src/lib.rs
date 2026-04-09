//! CUDA device implementation for Prelude inference engine.
//!
//! This crate owns:
//! - PTX kernel compilation (build.rs) and runtime loading constants
//! - GPU kernel wrappers (ops/) and attention backends (attn/)
//! - CudaOps: implements all Ops traits from prelude-core
//! - Re-exports of kernel sub-crates (deepgemm, cutlass-gemm, etc.)

// ── PTX modules compiled from src/kernels/ ──────────────────────────

pub const PTX_ADD: &str = include_str!(concat!(env!("OUT_DIR"), "/add.ptx"));
pub const PTX_SILU_MUL: &str = include_str!(concat!(env!("OUT_DIR"), "/silu_mul.ptx"));
pub const PTX_RMSNORM: &str = include_str!(concat!(env!("OUT_DIR"), "/rmsnorm.ptx"));
pub const PTX_QKNORM_ROPE: &str = include_str!(concat!(env!("OUT_DIR"), "/qknorm_rope.ptx"));
pub const PTX_MOE_ROUTING: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_routing.ptx"));
pub const PTX_MOE_GATEUP: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_gateup.ptx"));
pub const PTX_MOE_DOWN: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_down.ptx"));
pub const PTX_KV_APPEND: &str = include_str!(concat!(env!("OUT_DIR"), "/kv_append.ptx"));
pub const PTX_KNORM_ROPE_KV_WRITE: &str =
    include_str!(concat!(env!("OUT_DIR"), "/knorm_rope_kv_write.ptx"));
pub const PTX_SCATTER_KV_CACHE: &str =
    include_str!(concat!(env!("OUT_DIR"), "/scatter_kv_cache.ptx"));

// ── General-purpose PTX kernels (ported from candle-kernels) ────
pub const PTX_UNARY: &str = include_str!(concat!(env!("OUT_DIR"), "/candle_unary.ptx"));
pub const PTX_BINARY: &str = include_str!(concat!(env!("OUT_DIR"), "/candle_binary.ptx"));
pub const PTX_CAST: &str = include_str!(concat!(env!("OUT_DIR"), "/candle_cast.ptx"));
pub const PTX_REDUCE: &str = include_str!(concat!(env!("OUT_DIR"), "/candle_reduce.ptx"));
pub const PTX_INDEXING: &str = include_str!(concat!(env!("OUT_DIR"), "/candle_indexing.ptx"));
pub const PTX_TERNARY: &str = include_str!(concat!(env!("OUT_DIR"), "/candle_ternary.ptx"));
pub const PTX_AFFINE: &str = include_str!(concat!(env!("OUT_DIR"), "/candle_affine.ptx"));
pub const PTX_FILL: &str = include_str!(concat!(env!("OUT_DIR"), "/candle_fill.ptx"));
pub const PTX_SORT: &str = include_str!(concat!(env!("OUT_DIR"), "/candle_sort.ptx"));

// ── Module names for cudarc caching ─────────────────────────────────

pub const MOD_ADD: &str = "elementwise_add";
pub const MOD_SILU_MUL: &str = "elementwise_silu_mul";
pub const MOD_RMSNORM: &str = "normalization_rmsnorm";
pub const MOD_QKNORM_ROPE: &str = "rope_qknorm";
pub const MOD_MOE_ROUTING: &str = "moe_routing";
pub const MOD_MOE_GATEUP: &str = "moe_gateup";
pub const MOD_MOE_DOWN: &str = "moe_down";
pub const MOD_KV_APPEND: &str = "kvcache_kv_append";
pub const MOD_KNORM_ROPE_KV_WRITE: &str = "kvcache_knorm_rope_kv_write";
pub const MOD_SCATTER_KV_CACHE: &str = "kvcache_scatter_kv_cache";

// ── General-purpose module names ────────────────────────────────
pub const MOD_UNARY: &str = "candle_unary";
pub const MOD_BINARY: &str = "candle_binary";
pub const MOD_CAST: &str = "candle_cast";
pub const MOD_REDUCE: &str = "candle_reduce";
pub const MOD_INDEXING: &str = "candle_indexing";
pub const MOD_TERNARY: &str = "candle_ternary";
pub const MOD_AFFINE: &str = "candle_affine";
pub const MOD_FILL: &str = "candle_fill";
pub const MOD_SORT: &str = "candle_sort";

/// Probe whether CUDA is usable on this machine.
/// Creates a context for device 0; the context is cached for later use.
fn cuda_probe() -> bool {
    device::cuda_probe(0)
}

/// Register GPU ops and executor. Call once at startup.
pub fn register() {
    prelude_core::ops::register_backend(prelude_core::ops::OpsBackend {
        name: "cuda",
        priority: 100,
        probe: cuda_probe,
        supports: |d| d.is_cuda(),
        create_ops: cuda_ops::cuda_ops,
    });
    prelude_core::engine::executor::register_executor(
        prelude_core::engine::executor::ExecutorBackend {
            name: "cuda",
            priority: 100,
            probe: cuda_probe,
            supports: |d| d.is_cuda(),
            create: |engine| Box::new(executor::CudaExecutor::new(engine)),
        },
    );
}

// ── Own CUDA storage ─────────────────

pub mod device;
pub(crate) mod tensor_ops_kernels;

// ── GPU kernel modules ──────────────────────────────────────────────

pub(crate) mod ops;
pub(crate) mod attn;
pub(crate) mod moe_ffi;
mod cuda_ops;
mod cuda_graph;
mod quant_backends;
pub mod executor;

pub use cuda_ops::cuda_ops;
pub use attn::flashinfer::fi_fused_add_rmsnorm;

// ── Sub-crate re-exports ────────────────────────────────────────────

pub use prelude_deepgemm;
pub use prelude_cutlass_gemm;
pub use prelude_flash_attn_v4;
pub use prelude_flashinfer;
pub use prelude_quant_gemm;
pub use prelude_cula;
