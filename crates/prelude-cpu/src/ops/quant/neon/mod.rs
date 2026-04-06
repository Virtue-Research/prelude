//! ARM NEON SIMD kernels for quantized dot products.
//!
//! NEON is baseline on all aarch64 CPUs, so no runtime feature detection is needed.
//! These kernels use `std::arch::aarch64` intrinsics for 128-bit SIMD operations.
//!
//! Reference: llama.cpp `ggml-cpu/ggml-cpu-quants.c` and candle-core neon.rs.

pub mod q4_0;
pub mod q4_1;
pub mod q5_0;
pub mod q5_1;
pub mod q2_k;
pub mod q3_k;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod quantize;
