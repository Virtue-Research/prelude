// Shared rotary position embedding (RoPE).
//
// Supports standard [B, H, L, D], THD [B, L, H, D], and varlen [total, H, D] layouts.

use candle_core::{DType, Device, Result, Tensor};
use candle_transformers::models::qwen3::Config as Qwen3Config;

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
        let freqs = t.matmul(&inv_freq)?;
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
        if q.device().is_cpu() {
            if let Some(ref cache) = self.cos_sin_cache {
                let (total, h_q, d) = q.dims3()?;
                let h_k = k.dim(1)?;
                let q4 = q.reshape((1, total, h_q, d))?;
                let k4 = k.reshape((1, total, h_k, d))?;
                let positions: Vec<i64> = position_ids.to_dtype(DType::I64)?.to_vec1::<i64>()?;
                let (q_out, k_out) = if q.dtype() == DType::F32 {
                    crate::ops::cpu::cpu_rotary_embedding_f32_with_positions(
                        &q4, &k4, cache, &positions, h_q, h_k,
                    )?
                } else {
                    crate::ops::cpu::cpu_rotary_embedding_with_positions(
                        &q4, &k4, cache, &positions, h_q, h_k,
                    )?
                };
                return Ok((
                    q_out.reshape((total, h_q, d))?,
                    k_out.reshape((total, h_k, d))?,
                ));
            }
        }

        // Fallback: candle rope_thd (GPU)
        // q: (total_tokens, num_heads, head_dim), position_ids: (total_tokens,)
        let cos = self.cos.index_select(position_ids, 0)?;
        let sin = self.sin.index_select(position_ids, 0)?;
        // Wrap in batch dim=1 for rope_thd: (1, total_tokens, H, D)
        let (total, h_q, d) = q.dims3()?;
        let h_k = k.dim(1)?;
        let q4 = q.reshape((1, total, h_q, d))?;
        let k4 = k.reshape((1, total, h_k, d))?;
        let q_embed = candle_nn::rotary_emb::rope_thd(&q4, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_thd(&k4, &cos, &sin)?;
        Ok((
            q_embed.reshape((total, h_q, d))?,
            k_embed.reshape((total, h_k, d))?,
        ))
    }
}
