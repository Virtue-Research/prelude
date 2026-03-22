//! oneDNN integration for BF16/F32 CPU GEMM and brgemm micro-kernels.
//!
//! This module wraps the `crates/onednn-ffi` shared library, providing:
//! - Raw FFI bindings ([`ffi`])
//! - Safe Rust wrappers for GEMM, fused SiLU×Mul, and packed-weight management ([`ops`])
//! - [`OnednnLinear`]: drop-in replacement for `candle_nn::Linear` with oneDNN dispatch
//!
//! Only available when the `onednn` feature is enabled.

pub mod ffi;
pub mod ops;

// Re-export key public items so callers can use `crate::ops::onednn::init`, etc.
pub use ops::{
    brgemm_available, brgemm_fused_silu_mul, brgemm_fused_silu_mul_raw, brgemm_gemm_forward_pub,
    brgemm_gemm_raw, bind_threads, init, set_num_threads, BrgemmPackedWeight, OnednnLinear,
};
