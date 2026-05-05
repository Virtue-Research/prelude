//! CPU device implementation for Prelude inference engine.
//!
//! This crate owns:
//! - Pure Rust AVX-512 kernels (rmsnorm, attention, rope, silu_mul)
//! - oneDNN integration (brgemm GEMM, VNNI packing)
//! - Quantized format support (Q4_0, Q4_K, etc.)
//! - CpuOps: implements all Ops traits from prelude-core

/// Register CPU ops and executor. Call once at startup.
pub fn register() {
    prelude_core::ops::register_backend(prelude_core::ops::OpsBackend {
        name: "cpu",
        priority: 10,
        probe: || true,
        supports: |d| d.is_cpu(),
        create_ops: cpu_ops::cpu_ops,
    });
    prelude_core::engine::executor::register_executor(
        prelude_core::engine::executor::ExecutorBackend {
            name: "cpu",
            priority: 10,
            probe: || true,
            supports: |d| d.is_cpu(),
            create: |engine| Box::new(executor::CpuExecutor::new(engine)),
        },
    );
}

// ── CPU kernel modules ─────────────────────────────────────────────

pub mod attn_cpu;
mod cpu_ops;
#[cfg(feature = "onednn")]
pub mod onednn;
pub mod ops;

pub use cpu_ops::cpu_ops;
pub mod executor;
mod linear_backends;
