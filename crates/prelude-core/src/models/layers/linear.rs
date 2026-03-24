// Shared linear & RmsNorm type aliases and wrapper functions.
//
// Priority for GEMM: onednn > candle (fallback).
// Norms: CpuRmsNorm (pure Rust AVX-512) on CPU, candle RmsNorm on CUDA.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, RmsNorm};

// ── QwenLinear type alias ────────────────────────────────────────────────

#[cfg(feature = "onednn")]
pub(crate) type QwenLinear = crate::ops::onednn::OnednnLinear;

#[cfg(not(feature = "onednn"))]
pub(crate) type QwenLinear = Linear;

// ── QwenRmsNorm type alias ───────────────────────────────────────────────

pub(crate) type QwenRmsNorm = crate::ops::cpu::CpuRmsNorm;

// ── wrap_linear ──────────────────────────────────────────────────────────

#[cfg(feature = "onednn")]
pub(crate) fn wrap_linear(linear: Linear) -> Result<QwenLinear> {
    crate::ops::onednn::OnednnLinear::new(linear)
}

#[cfg(not(feature = "onednn"))]
pub(crate) fn wrap_linear(linear: Linear) -> Result<QwenLinear> {
    Ok(linear)
}

// ── wrap_rmsnorm ─────────────────────────────────────────────────────────

pub(crate) fn wrap_rmsnorm(norm: RmsNorm, eps: f64, weight: Tensor) -> QwenRmsNorm {
    crate::ops::cpu::CpuRmsNorm::new(norm, eps, weight)
}
