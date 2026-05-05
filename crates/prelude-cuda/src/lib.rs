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
pub const PTX_RMSNORM_GATED: &str = include_str!(concat!(env!("OUT_DIR"), "/rmsnorm_gated.ptx"));
pub const PTX_QKNORM_ROPE: &str = include_str!(concat!(env!("OUT_DIR"), "/qknorm_rope.ptx"));
pub const PTX_MOE_ROUTING: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_routing.ptx"));
pub const PTX_MOE_GATEUP: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_gateup.ptx"));
pub const PTX_MOE_DOWN: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_down.ptx"));
pub const PTX_KV_APPEND: &str = include_str!(concat!(env!("OUT_DIR"), "/kv_append.ptx"));
pub const PTX_KNORM_ROPE_KV_WRITE: &str =
    include_str!(concat!(env!("OUT_DIR"), "/knorm_rope_kv_write.ptx"));
pub const PTX_SCATTER_KV_CACHE: &str =
    include_str!(concat!(env!("OUT_DIR"), "/scatter_kv_cache.ptx"));
pub const PTX_GDN_POST_CONV: &str = include_str!(concat!(env!("OUT_DIR"), "/gdn_post_conv.ptx"));
pub const PTX_GATHER_LOG_SOFTMAX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/gather_log_softmax.ptx"));

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
pub const MOD_GDN_POST_CONV: &str = "gdn_post_conv";
pub const MOD_GATHER_LOG_SOFTMAX: &str = "logprobs_gather_log_softmax";

/// Probe whether CUDA is usable on this machine.
/// Creates a context for device 0; the context is cached for later use.
fn cuda_probe() -> bool {
    device::cuda_probe(0)
}

/// Register GPU ops and executor. Call once at startup.
pub fn register() {
    // Install the GEMM dispatch immediately so that any model-construction
    // matmul (e.g. RoPE inv_freq * positions) that runs before the first
    // `cuda_ops()` lookup still goes through our dispatch.
    if cuda_probe() {
        crate::ops::gemm::register_gpu_gemm();
    }
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

// ── GPU kernel modules ──────────────────────────────────────────────

pub(crate) mod attn;
mod cuda_graph;
mod cuda_ops;
pub mod executor;
pub(crate) mod moe_ffi;
pub(crate) mod ops;
#[cfg(feature = "quant-gemm")]
mod quant_backends;

pub use attn::flashinfer::fi_fused_add_rmsnorm;
pub use cuda_ops::cuda_ops;

// ── Sub-crate re-exports ────────────────────────────────────────────

pub use cula;
pub use cutlass_gemm;
pub use deepgemm;
pub use flash_attn_v4;
pub use flashinfer;
#[cfg(feature = "quant-gemm")]
pub use quant_gemm;
