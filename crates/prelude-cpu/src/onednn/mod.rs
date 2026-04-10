//! oneDNN integration for BF16/F32/INT8/FP8 CPU GEMM and brgemm micro-kernels.
//!
//! This module wraps the `crates/onednn-ffi` shared library, providing:
//! - Raw FFI bindings ([`ffi`])
//! - Safe Rust wrappers for GEMM, fused SiLU×Mul, and packed-weight management ([`ops`])
//! - [`OnednnLinear`]: drop-in replacement for `DenseLinear` with oneDNN dispatch
//! - INT8 W8A8 quantized GEMM via brgemm micro-kernels
//! - FP8 (E4M3) GEMM (requires AVX10.2 AMX-2 hardware)
//! - Post-ops fusion (bias, GELU, ReLU) into brgemm output path
//!
//! Only available when the `onednn` feature is enabled.

pub mod ffi;
pub mod ops;

// Re-export key public items so callers can use `crate::onednn::init`, etc.
pub use ops::{
    brgemm_available, brgemm_fused_silu_mul, brgemm_fused_silu_mul_raw, brgemm_gemm_forward_pub,
    brgemm_gemm_raw, bind_threads, init, set_num_threads, BrgemmPackedWeight, OnednnF32PackedWeight,
    OnednnLinear,
    // INT8 W8A8
    brgemm_s8_available, brgemm_quantize_bf16_s8, brgemm_s8_gemm_forward, BrgemmS8PackedWeight,
    // FP8 E4M3
    brgemm_f8_available, BrgemmF8PackedWeight,
    // Post-ops
    brgemm_gemm_forward_postops, BRGEMM_POSTOP_BIAS, BRGEMM_POSTOP_GELU_TANH,
    BRGEMM_POSTOP_GELU_ERF, BRGEMM_POSTOP_RELU,
};
