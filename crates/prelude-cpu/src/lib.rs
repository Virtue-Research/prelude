//! CPU device implementation for Prelude inference engine.
//!
//! This crate owns:
//! - Pure Rust AVX-512 kernels (rmsnorm, attention, rope, silu_mul)
//! - oneDNN integration (brgemm GEMM, VNNI packing)
//! - Quantized format support (Q4_0, Q4_K, etc.)
//! - CpuOps: implements all Ops traits from prelude-core

/// Register CPU ops and executor. Call once at startup.
pub fn register() {
    prelude_core::ops::register_cpu_ops(cpu_ops::cpu_ops);
    prelude_core::engine::executor::register_executor(|engine| Box::new(executor::CpuExecutor::new(engine)));
}

// ── CPU kernel modules ─────────────────────────────────────────────

pub mod ops;
#[cfg(feature = "onednn")]
pub mod onednn;
mod cpu_ops;
pub mod raw_cpu;
pub mod attn_cpu;

pub use cpu_ops::cpu_ops;
pub mod executor;
mod linear_backends;
