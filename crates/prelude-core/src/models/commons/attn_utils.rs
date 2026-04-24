//! Shared attention utilities: RoPE, fused QKV projection, helpers.

use crate::tensor::{DType, Device, Result, Tensor};
use crate::models::config::Qwen3Config;

use super::linear::Linear;

// ── Rotary Position Embedding (RoPE) ───────────────────────────────

/// Precomputed sin/cos tables for rotary position embedding.
///
/// Supports standard [B, H, L, D], THD [B, L, H, D], and varlen [total, H, D] layouts.
#[derive(Debug, Clone)]
pub(crate) struct RotaryEmbedding {
    pub(crate) sin: Tensor,
    pub(crate) cos: Tensor,
    /// Packed [cos || sin] cache for cpu_ops / sgl-kernel RoPE: [max_seq_len, head_dim]
    pub(crate) cos_sin_cache: Option<Tensor>,
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

        let cos_sin_cache = if dev.is_cpu() && (dtype == DType::BF16 || dtype == DType::F32) {
            Some(Tensor::cat(&[&cos, &sin], 1)?)
        } else {
            None
        };

        Ok(Self {
            sin,
            cos,
            cos_sin_cache,
        })
    }

    /// Apply RoPE to packed Q/K in [total_tokens, H, D] format with explicit position_ids.
    /// Used by varlen attention where sequences have different lengths.
    pub(crate) fn apply_varlen(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Fast path: cpu_ops RoPE (in-place, no Tensor allocs)
        // candle rope_thd
        // q: (total_tokens, num_heads, head_dim), position_ids: (total_tokens,)
        let cos = self.cos.index_select(position_ids, 0)?;
        let sin = self.sin.index_select(position_ids, 0)?;
        // Wrap in batch dim=1 for rope_thd: (1, total_tokens, H, D)
        let (total, h_q, d) = q.dims3()?;
        let h_k = k.dim(1)?;
        let q4 = q.reshape((1, total, h_q, d))?;
        let k4 = k.reshape((1, total, h_k, d))?;
        let q_embed = crate::ops::rope_thd(&q4, &cos, &sin)?;
        let k_embed = crate::ops::rope_thd(&k4, &cos, &sin)?;
        Ok((
            q_embed.reshape((total, h_q, d))?,
            k_embed.reshape((total, h_k, d))?,
        ))
    }
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
        let q = qkv_out
            .narrow(1, 0, q_size)?
            .reshape((total, num_heads, head_dim))?;
        let k = qkv_out
            .narrow(1, q_size, kv_size)?
            .reshape((total, num_kv_heads, head_dim))?;
        let v = qkv_out
            .narrow(1, q_size + kv_size, kv_size)?
            .reshape((total, num_kv_heads, head_dim))?;
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

// ── Helpers ────────────────────────────────────────────────────────

/// FA4 tile_n for paged attention (causal, non-windowed).
pub(crate) fn fa4_tile_n(head_dim: usize, head_dim_v: usize) -> usize {
    match head_dim {
        d if d <= 64 => 128,
        d if d <= 96 => 128,
        d if d <= 128 => 128,
        d if d <= 192 => if head_dim_v <= 128 { 128 } else { 112 },
        _ => 80,
    }
}

/// Convert cumulative sequence lengths to per-sequence lengths.
pub(crate) fn cu_seqlens_to_lens(cu_seqlens: &Tensor) -> Result<Tensor> {
    let n = cu_seqlens.dim(0)? - 1;
    let hi = cu_seqlens.narrow(0, 1, n)?;
    let lo = cu_seqlens.narrow(0, 0, n)?;
    hi.sub(&lo)
}
