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

/// Vectorized add: tries Ops fused kernel first, CPU F32 uses in-place add,
/// everything else falls back to candle's default add.
pub(crate) fn fast_add(ops: &crate::ops::Ops, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if let Some(result) = ops.fused.fused_add(a, b) {
        if !debug_disable_vectorized_add() {
            return result;
        }
    }
    if a.device().is_cpu() && a.dtype() == crate::tensor::DType::F32 {
        return cpu_add_f32(a, b);
    }
    a + b
}

/// Element-wise F32 add on CPU, avoiding candle tensor overhead.
fn cpu_add_f32(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_c = a.contiguous()?;
    let b_c = b.contiguous()?;
    let a_slice = crate::ops::cpu::tensor_as_f32_slice(&a_c)?;
    let b_slice = crate::ops::cpu::tensor_as_f32_slice(&b_c)?;
    let n = a_slice.len();
    debug_assert_eq!(n, b_slice.len());
    let mut out = vec![0.0f32; n];
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        while i + 16 <= n {
            unsafe {
                use core::arch::x86_64::*;
                let va = _mm512_loadu_ps(a_slice.as_ptr().add(i));
                let vb = _mm512_loadu_ps(b_slice.as_ptr().add(i));
                _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_add_ps(va, vb));
            }
            i += 16;
        }
    }
    while i < n {
        out[i] = a_slice[i] + b_slice[i];
        i += 1;
    }
    Tensor::from_vec(out, a.dims(), a.device())
}

/// Custom fast RMSNorm: tries Ops norm kernel, CPU custom kernel, fallback candle.
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
    if x.device().is_cpu() {
        return crate::ops::cpu::cpu_rmsnorm(x, weight, eps);
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

/// Fused residual add + RMSNorm: tries Ops fused kernel, falls back to cpu_ops.
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
    crate::ops::cpu::cpu_fused_add_rmsnorm(h, residual, weight, eps)
}

/// Fused SiLU × Mul: tries Ops fused kernel, falls back to cpu_ops (BF16/F32).
pub(crate) fn fast_silu_mul(ops: &crate::ops::Ops, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if let Some(result) = ops.fused.fused_silu_mul(gate, up) {
        if !debug_disable_fused_silu_mul() {
            return result;
        }
    }
    let gate_up = Tensor::cat(&[gate, up], gate.dims().len() - 1)?;
    crate::ops::cpu::cpu_silu_and_mul(&gate_up)
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

    // ── CPU: cpu_ops rmsnorm + rotary_embedding (BF16 and F32) ──
    if q.device().is_cpu() && (q.dtype() == DType::BF16 || q.dtype() == DType::F32) {
        if let Some(ref cache) = rotary_emb.cos_sin_cache {
            let q = crate::ops::cpu::cpu_rmsnorm(&q.flatten(0, 1)?, q_norm_weight, rms_norm_eps)?
                .reshape((total, num_heads, head_dim))?;
            let k = crate::ops::cpu::cpu_rmsnorm(&k.flatten(0, 1)?, k_norm_weight, rms_norm_eps)?
                .reshape((total, num_kv_heads, head_dim))?;

            let positions: Vec<i64> = position_ids
                .to_dtype(DType::I64)?
                .to_vec1::<i64>()?;
            let q4 = q.reshape((1, total, num_heads, head_dim))?;
            let k4 = k.reshape((1, total, num_kv_heads, head_dim))?;
            let (q_out, k_out) = if q.dtype() == DType::F32 {
                crate::ops::cpu::cpu_rotary_embedding_f32_with_positions(
                    &q4, &k4, cache, &positions, num_heads, num_kv_heads,
                )?
            } else {
                crate::ops::cpu::cpu_rotary_embedding_with_positions(
                    &q4, &k4, cache, &positions, num_heads, num_kv_heads,
                )?
            };

            return Ok((
                q_out.reshape((total, num_heads, head_dim))?,
                k_out.reshape((total, num_kv_heads, head_dim))?,
            ));
        }
    }

    // ── Generic fallback ──
    let q = fast_rms_norm(ops, &q.flatten(0, 1)?, q_norm, q_norm_weight, rms_norm_eps)?
        .reshape((total, num_heads, head_dim))?;
    let k = fast_rms_norm(ops, &k.flatten(0, 1)?, k_norm, k_norm_weight, rms_norm_eps)?
        .reshape((total, num_kv_heads, head_dim))?;
    rotary_emb.apply_varlen(&q, &k, position_ids)
}
