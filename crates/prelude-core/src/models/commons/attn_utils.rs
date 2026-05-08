//! Shared attention utilities: RoPE, fused QKV projection, helpers.

use crate::models::config::Qwen3Config;
use crate::tensor::{D, DType, Device, Result, Tensor};

use super::linear::Linear;

// ── Rotary Position Embedding (RoPE) ───────────────────────────────

/// Precomputed sin/cos tables for rotary position embedding.
///
/// Supports standard [B, H, L, D], THD [B, L, H, D], and varlen [total, H, D] layouts.
#[derive(Debug, Clone)]
pub(crate) struct RotaryEmbedding {
    pub(crate) sin: Tensor,
    pub(crate) cos: Tensor,
}

impl RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &Qwen3Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.broadcast_mul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self { sin, cos })
    }
}

pub(crate) fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let part1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let part2 = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;
    Tensor::cat(&[&part1, &part2], D::Minus1)
}

pub(crate) fn apply_partial_rotary_varlen(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
    position_ids: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.index_select(position_ids, 0)?.unsqueeze(1)?;
    let sin = sin.index_select(position_ids, 0)?.unsqueeze(1)?;

    let q_rot = q.narrow(D::Minus1, 0, rotary_dim)?;
    let q_pass = q.narrow(D::Minus1, rotary_dim, q.dim(D::Minus1)? - rotary_dim)?;
    let q_rot = apply_rotary_emb(&q_rot, &cos, &sin)?;
    let q = Tensor::cat(&[q_rot, q_pass], D::Minus1)?;

    let k_rot = k.narrow(D::Minus1, 0, rotary_dim)?;
    let k_pass = k.narrow(D::Minus1, rotary_dim, k.dim(D::Minus1)? - rotary_dim)?;
    let k_rot = apply_rotary_emb(&k_rot, &cos, &sin)?;
    let k = Tensor::cat(&[k_rot, k_pass], D::Minus1)?;

    Ok((q, k))
}

// ── Fused QKV Projection ───────────────────────────────────────────

/// Project Q, K, V and reshape to `[total_tokens, H, D]` (varlen layout).
///
/// Uses fused QKV GEMM when available (1 GEMM instead of 3).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_qkv_projection(
    x: &Tensor,
    q_proj: &Linear,
    k_proj: &Linear,
    v_proj: &Linear,
    qkv_proj: Option<&Linear>,
    total: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    ctx: &super::BatchState,
    ops: &dyn crate::ops::Ops,
) -> Result<(Tensor, Tensor, Tensor)> {
    if let Some(qkv) = qkv_proj {
        let qkv_out = qkv.forward(x, ctx, ops)?;
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        // narrow creates non-contiguous views (stride[0] = N_fused > q_size).
        // Q/K are consumed by stride-aware fused_qknorm_rope.
        // V is consumed by stride-aware scatter_kv_cache_flash on the paged path.
        // For the rare non-paged varlen path, the attention wrapper contiguous-ifies.
        let q = qkv_out
            .narrow(1, 0, q_size)?
            .reshape((total, num_heads, head_dim))?;
        let k = qkv_out
            .narrow(1, q_size, kv_size)?
            .reshape((total, num_kv_heads, head_dim))?;
        let v = qkv_out.narrow(1, q_size + kv_size, kv_size)?.reshape((
            total,
            num_kv_heads,
            head_dim,
        ))?;
        return Ok((q, k, v));
    }
    let q = q_proj.forward(x, ctx, ops)?;
    let k = k_proj.forward(x, ctx, ops)?;
    let v = v_proj.forward(x, ctx, ops)?;
    Ok((
        q.reshape((total, num_heads, head_dim))?,
        k.reshape((total, num_kv_heads, head_dim))?,
        v.reshape((total, num_kv_heads, head_dim))?,
    ))
}
