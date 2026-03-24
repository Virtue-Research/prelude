//! Kernel operations — all compute backends organized by device target.
//!
//! - `cpu`: Pure Rust AVX-512 kernels (rmsnorm, attention, rope, silu_mul)
//! - `gpu`: CUDA PTX kernels (fused ops loaded via cudarc)
//! - `onednn`: Intel oneDNN integration (brgemm GEMM, VNNI packing)

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "onednn")]
pub mod onednn;
