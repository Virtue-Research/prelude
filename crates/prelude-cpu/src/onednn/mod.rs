//! oneDNN integration for BF16/F32/INT8/FP8 CPU GEMM and brgemm micro-kernels.
//!
//! This module layers on top of the standalone `onednn-ffi` crate, adding:
//! - Safe Rust wrappers for GEMM, fused SiLU×Mul, and packed-weight management ([`ops`])
//! - [`OnednnLinear`]: drop-in replacement for `DenseLinear` with oneDNN dispatch
//! - INT8 W8A8 quantized GEMM via brgemm micro-kernels
//! - FP8 (E4M3) GEMM (requires AVX10.2 AMX-2 hardware)
//! - Post-ops fusion (bias, GELU, ReLU) into brgemm output path
//!
//! Only available when the `onednn` feature is enabled.

// Re-export the external `onednn-ffi` crate as `super::ffi` so the existing
// `ops.rs` path (`super::ffi::onednn_init()`) keeps working after the
// extraction. Tests and downstream code should still go through the safe
// wrappers in `ops` rather than touching the raw FFI directly.
pub use onednn_ffi as ffi;

pub mod ops;

// Re-export key public items so callers can use `crate::onednn::init`, etc.
pub use ops::{
    BRGEMM_POSTOP_BIAS,
    BRGEMM_POSTOP_GELU_ERF,
    BRGEMM_POSTOP_GELU_TANH,
    BRGEMM_POSTOP_RELU,
    BrgemmF8PackedWeight,
    BrgemmPackedWeight,
    BrgemmS8PackedWeight,
    OnednnF32PackedWeight,
    OnednnLinear,
    bind_threads,
    brgemm_available,
    // FP8 E4M3
    brgemm_f8_available,
    brgemm_fused_silu_mul,
    brgemm_fused_silu_mul_raw,
    // Post-ops
    brgemm_gemm_forward_postops,
    brgemm_gemm_forward_pub,
    brgemm_gemm_raw,
    brgemm_quantize_bf16_s8,
    // INT8 W8A8
    brgemm_s8_available,
    brgemm_s8_gemm_forward,
    init,
    set_num_threads,
};
