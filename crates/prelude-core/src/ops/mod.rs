//! Kernel operations — trait definitions and compute backends.
//!
//! - `traits`: Op trait definitions (AttentionOps, GemmOps, NormOps, FusedOps, ...)
//! - `cpu`: Pure Rust AVX-512 kernels (rmsnorm, attention, rope, silu_mul)
//! - `onednn`: Intel oneDNN integration (brgemm GEMM, VNNI packing)
//! - `cpu_ops`: CpuOps struct implementing all Ops traits

pub mod traits;
pub mod cpu;
pub mod onednn;
pub mod cpu_ops;

// Re-export the Ops bundle and all trait types at crate level for convenience.
pub use traits::*;
pub use cpu_ops::{CpuOps, cpu_ops};
