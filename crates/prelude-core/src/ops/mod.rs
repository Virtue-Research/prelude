//! Kernel operations — trait definitions and compute backends.
//!
//! - `traits`: Op trait definitions (AttentionOps, GemmOps, NormOps, FusedOps, ...)
//! - `cpu`: Pure Rust AVX-512 kernels (rmsnorm, attention, rope, silu_mul)
//! - `onednn`: Intel oneDNN integration (brgemm GEMM, VNNI packing)

pub mod traits;
pub mod cpu;
pub mod onednn;

// Re-export the Ops bundle and all trait types at crate level for convenience.
pub use traits::*;
