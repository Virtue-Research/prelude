// Shared fused/debug ops used across all model architectures.
//
// Extracted from qwen3/mod.rs to be reusable.

// Debug setters are public API for runtime bisection; not called within the crate.
#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};

use crate::tensor::{DType, Module, Result, Tensor};

use super::linear::RmsNorm;

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

/// Vectorized add: tries Ops fused kernel, falls back to candle add.
pub(crate) fn fast_add(ops: &crate::ops::Ops, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if let Some(result) = ops.fused.fused_add(a, b) {
        if !debug_disable_vectorized_add() {
            return result;
        }
    }
    a + b
}

/// Custom fast RMSNorm: tries Ops norm kernel, fallback candle.
pub(crate) fn fast_rms_norm(
    ops: &crate::ops::Ops,
    x: &Tensor,
    norm: &RmsNorm,
    weight: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    if !debug_disable_fast_rmsnorm() {
        if let Ok(result) = ops.norm.rms_norm(x, weight, eps as f32) {
            return Ok(result);
        }
    }
    norm.forward(x)
}

/// Extract the last token per sequence from a packed `[total_tokens, ...]` tensor.
///
/// Given `seq_lens = [3, 5, 2]`, returns rows at indices `[2, 7, 9]`.
pub(crate) fn last_token_select(hidden: &Tensor, seq_lens: &[usize]) -> Result<Tensor> {
    // Fast path: all seq_lens == 1 (decode Q=1).
    // Hidden is already [batch_size, hidden_dim] — no index_select needed.
    // Also required for CUDA graph safety (avoids Tensor::from_vec during replay).
    if seq_lens.iter().all(|&l| l == 1) {
        return Ok(hidden.clone());
    }
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

/// Fused residual add + RMSNorm: tries Ops fused kernel, falls back to separate add + norm.
pub(crate) fn fused_add_rmsnorm(
    ops: &crate::ops::Ops,
    residual: &Tensor,
    h: &Tensor,
    norm: &RmsNorm,
    weight: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    if let Some(result) = ops.fused.fused_add_rmsnorm(residual, h, weight, eps as f32) {
        if !debug_disable_fused_add_rmsnorm() {
            return result;
        }
    }
    // Fallback: separate add + rmsnorm
    let sum = (residual + h)?;
    let normed = ops.norm.rms_norm(&sum, weight, eps as f32)?;
    Ok((sum, normed))
}

/// Fused SiLU × Mul: tries Ops fused kernel, falls back to candle ops.
pub(crate) fn fast_silu_mul(ops: &crate::ops::Ops, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if let Some(result) = ops.fused.fused_silu_mul(gate, up) {
        if !debug_disable_fused_silu_mul() {
            return result;
        }
    }
    // Fallback: separate silu + mul via Ops activation
    let silu_gate = ops.act.silu(gate)?;
    &silu_gate * up
}

// ── QK-Norm + RoPE dispatch ─────────────────────────────────────────

/// QK-Norm + RoPE for varlen layout `[total, H, D]`.
///
/// Ops fused: fused qknorm_rope kernel.
/// CPU BF16: cpu_ops rmsnorm + cpu_ops rotary_embedding.
/// Fallback: fast_rms_norm + rotary_emb.apply_varlen.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qknorm_rope_varlen(
    ops: &crate::ops::Ops,
    q: &Tensor,
    k: &Tensor,
    q_norm_weight: &Tensor,
    k_norm_weight: &Tensor,
    q_norm: &RmsNorm,
    k_norm: &RmsNorm,
    rotary_emb: &super::rotary::RotaryEmbedding,
    position_ids: &Tensor,
    total: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
) -> Result<(Tensor, Tensor)> {
    // ── Fused QK-norm + RoPE kernel via Ops trait ──
    if let Some(result) = ops.fused.fused_qknorm_rope(
        q, k, q_norm_weight, k_norm_weight,
        &rotary_emb.cos, &rotary_emb.sin,
        position_ids, rms_norm_eps as f32,
    ) {
        if !debug_disable_fused_qknorm_rope() {
            return result;
        }
    }

    // ── Generic fallback (CpuOps handles CPU-specific kernels via Ops trait) ──
    let q = fast_rms_norm(ops, &q.flatten(0, 1)?, q_norm, q_norm_weight, rms_norm_eps)?
        .reshape((total, num_heads, head_dim))?;
    let k = fast_rms_norm(ops, &k.flatten(0, 1)?, k_norm, k_norm_weight, rms_norm_eps)?
        .reshape((total, num_kv_heads, head_dim))?;
    rotary_emb.apply_varlen(&q, &k, position_ids)
}
