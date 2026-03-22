// Shared fused/debug ops used across all model architectures.
//
// Extracted from qwen3/mod.rs to be reusable.

// Debug setters are public API for runtime bisection; not called within the crate.
#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Module, Result, Tensor};

use super::linear::QwenRmsNorm;

// ── Debug flags ────────────────────────────────────────────────────────

static DEBUG_DISABLE_FAST_RMSNORM: AtomicBool = AtomicBool::new(false);
static DEBUG_DISABLE_FUSED_QKNORM_ROPE: AtomicBool = AtomicBool::new(false);
static DEBUG_DISABLE_VECTORIZED_ADD: AtomicBool = AtomicBool::new(false);
static DEBUG_DISABLE_FUSED_SILU_MUL: AtomicBool = AtomicBool::new(false);
static DEBUG_DISABLE_FUSED_ADD_RMSNORM: AtomicBool = AtomicBool::new(false);
static DEBUG_DISABLE_FLASH_ATTN_PATH: AtomicBool = AtomicBool::new(false);

/// Runtime debug switch for pinpointing numeric drift.
/// When enabled, CUDA path falls back to candle's reference RMSNorm.
pub fn set_debug_disable_fast_rmsnorm(disable: bool) {
    DEBUG_DISABLE_FAST_RMSNORM.store(disable, Ordering::Relaxed);
}

/// Runtime debug switch for pinpointing numeric drift.
/// When enabled, CUDA path falls back to split QK-norm + RoPE kernels.
pub fn set_debug_disable_fused_qknorm_rope(disable: bool) {
    DEBUG_DISABLE_FUSED_QKNORM_ROPE.store(disable, Ordering::Relaxed);
}

/// Runtime debug switch for pinpointing numeric drift.
/// When enabled, residual add falls back to candle's default add.
pub fn set_debug_disable_vectorized_add(disable: bool) {
    DEBUG_DISABLE_VECTORIZED_ADD.store(disable, Ordering::Relaxed);
}

/// Runtime debug switch for pinpointing numeric drift.
/// When enabled, MLP SiLU*up falls back to split ops.
pub fn set_debug_disable_fused_silu_mul(disable: bool) {
    DEBUG_DISABLE_FUSED_SILU_MUL.store(disable, Ordering::Relaxed);
}

/// Runtime debug switch for pinpointing numeric drift.
/// When enabled, decoder falls back to split residual add + RMSNorm.
pub fn set_debug_disable_fused_add_rmsnorm(disable: bool) {
    DEBUG_DISABLE_FUSED_ADD_RMSNORM.store(disable, Ordering::Relaxed);
}

/// Runtime debug switch for pinpointing numeric drift.
/// When enabled, attention falls back from flash-attn path to standard matmul path.
pub fn set_debug_disable_flash_attn_path(disable: bool) {
    DEBUG_DISABLE_FLASH_ATTN_PATH.store(disable, Ordering::Relaxed);
}

#[inline]
pub(crate) fn debug_disable_fast_rmsnorm() -> bool {
    DEBUG_DISABLE_FAST_RMSNORM.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn debug_disable_fused_qknorm_rope() -> bool {
    DEBUG_DISABLE_FUSED_QKNORM_ROPE.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn debug_disable_vectorized_add() -> bool {
    DEBUG_DISABLE_VECTORIZED_ADD.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn debug_disable_fused_silu_mul() -> bool {
    DEBUG_DISABLE_FUSED_SILU_MUL.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn debug_disable_fused_add_rmsnorm() -> bool {
    DEBUG_DISABLE_FUSED_ADD_RMSNORM.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn debug_disable_flash_attn_path() -> bool {
    DEBUG_DISABLE_FLASH_ATTN_PATH.load(Ordering::Relaxed)
}

// ── Shared utility functions ───────────────────────────────────────────

/// Vectorized BF16 add on CUDA, falls back to candle's default add.
pub(crate) fn fast_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if a.device().is_cuda() && !debug_disable_vectorized_add() {
        return crate::ops::gpu::vectorized_add(a, b);
    }
    a + b
}

/// Custom fast RMSNorm on CUDA, falls back to candle's default RmsNorm.
pub(crate) fn fast_rms_norm(
    x: &Tensor,
    norm: &QwenRmsNorm,
    weight: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && !debug_disable_fast_rmsnorm() {
        return crate::ops::gpu::fast_rmsnorm(x, weight, eps);
    }
    norm.forward(x)
}

/// Extract the last token per sequence from a packed `[total_tokens, ...]` tensor.
///
/// Given `seq_lens = [3, 5, 2]`, returns rows at indices `[2, 7, 9]`.
pub(crate) fn last_token_select(hidden: &Tensor, seq_lens: &[usize]) -> Result<Tensor> {
    let batch_size = seq_lens.len();
    let mut last_indices = Vec::with_capacity(batch_size);
    let mut off = 0usize;
    for &len in seq_lens {
        last_indices.push((off + len - 1) as u32);
        off += len;
    }
    let indices = Tensor::from_vec(last_indices, (batch_size,), hidden.device())?;
    hidden.index_select(&indices, 0)
}

/// Extract the first token per sequence from a packed `[total_tokens, ...]` tensor.
///
/// Given `seq_lens = [3, 5, 2]`, returns rows at indices `[0, 3, 8]`.
pub(crate) fn first_token_select(hidden: &Tensor, seq_lens: &[usize]) -> Result<Tensor> {
    let batch_size = seq_lens.len();
    let mut first_indices = Vec::with_capacity(batch_size);
    let mut off = 0usize;
    for &len in seq_lens {
        first_indices.push(off as u32);
        off += len;
    }
    let indices = Tensor::from_vec(first_indices, (batch_size,), hidden.device())?;
    hidden.index_select(&indices, 0)
}

/// Fused residual add + RMSNorm: CUDA fused kernel or cpu_ops BF16.
#[allow(unused_variables)]
pub(crate) fn fused_add_rmsnorm(
    residual: &Tensor,
    h: &Tensor,
    norm: &QwenRmsNorm,
    weight: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if residual.device().is_cuda() && !debug_disable_fused_add_rmsnorm() {
        return crate::ops::gpu::fused_add_rmsnorm(residual, h, weight, eps);
    }
    crate::ops::cpu::cpu_fused_add_rmsnorm(h, residual, weight, eps)
}

/// Fused SiLU × Mul: CUDA fused kernel or cpu_ops (BF16/F32).
pub(crate) fn fast_silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if gate.device().is_cuda() && !debug_disable_fused_silu_mul() {
        return crate::ops::gpu::fused_silu_mul(gate, up);
    }
    let gate_up = Tensor::cat(&[gate, up], gate.dims().len() - 1)?;
    crate::ops::cpu::cpu_silu_and_mul(&gate_up)
}

// ── QK-Norm + RoPE dispatch ─────────────────────────────────────────

/// QK-Norm + RoPE for varlen layout `[total, H, D]`.
///
/// CUDA: fused qknorm_rope kernel.
/// CPU BF16: cpu_ops rmsnorm + cpu_ops rotary_embedding.
/// Fallback: fast_rms_norm + rotary_emb.apply_varlen.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qknorm_rope_varlen(
    q: &Tensor,
    k: &Tensor,
    q_norm_weight: &Tensor,
    k_norm_weight: &Tensor,
    q_norm: &QwenRmsNorm,
    k_norm: &QwenRmsNorm,
    rotary_emb: &super::rotary::Qwen3RotaryEmbedding,
    position_ids: &Tensor,
    total: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
) -> Result<(Tensor, Tensor)> {
    // ── CUDA: fused QK-norm + RoPE kernel ──
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() && !debug_disable_fused_qknorm_rope() {
        let q = crate::ops::gpu::fused_qknorm_rope_varlen(
            q, q_norm_weight, &rotary_emb.cos, &rotary_emb.sin, position_ids, rms_norm_eps,
        )?;
        let k = crate::ops::gpu::fused_qknorm_rope_varlen(
            k, k_norm_weight, &rotary_emb.cos, &rotary_emb.sin, position_ids, rms_norm_eps,
        )?;
        return Ok((q, k));
    }

    // ── CPU BF16: cpu_ops rmsnorm + rotary_embedding ──
    if q.device().is_cpu() && q.dtype() == DType::BF16 {
        if let Some(ref cache) = rotary_emb.cos_sin_cache {
            let q = crate::ops::cpu::cpu_rmsnorm(&q.flatten(0, 1)?, q_norm_weight, rms_norm_eps)?
                .reshape((total, num_heads, head_dim))?;
            let k = crate::ops::cpu::cpu_rmsnorm(&k.flatten(0, 1)?, k_norm_weight, rms_norm_eps)?
                .reshape((total, num_kv_heads, head_dim))?;

            let positions: Vec<i64> = position_ids
                .to_dtype(DType::I64)?
                .to_vec1::<i64>()?;
            let (q_out, k_out) = crate::ops::cpu::cpu_rotary_embedding_with_positions(
                &q.reshape((1, total, num_heads, head_dim))?,
                &k.reshape((1, total, num_kv_heads, head_dim))?,
                cache,
                &positions,
                num_heads,
                num_kv_heads,
            )?;

            return Ok((
                q_out.reshape((total, num_heads, head_dim))?,
                k_out.reshape((total, num_kv_heads, head_dim))?,
            ));
        }
    }

    // ── Generic fallback ──
    let q = fast_rms_norm(&q.flatten(0, 1)?, q_norm, q_norm_weight, rms_norm_eps)?
        .reshape((total, num_heads, head_dim))?;
    let k = fast_rms_norm(&k.flatten(0, 1)?, k_norm, k_norm_weight, rms_norm_eps)?
        .reshape((total, num_kv_heads, head_dim))?;
    rotary_emb.apply_varlen(&q, &k, position_ids)
}

