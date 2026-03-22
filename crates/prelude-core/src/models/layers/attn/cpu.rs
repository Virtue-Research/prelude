//! CPU attention fallback (matmul SDPA / tiled BF16).
//!
//! Used when no GPU attention backend is compiled or when running on CPU.

use candle_core::{Result, Tensor};

/// Causal varlen attention on CPU.
/// BF16: optimized tiled kernel. F32: matmul SDPA with causal mask.
pub fn varlen_causal(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, _cu_seqlens_k: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let cu_q: Vec<u32> = cu_seqlens_q.to_vec1()?;
    let seq_lens: Vec<usize> = cu_q.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let num_heads = q.dim(1)?;
    let num_kv_heads = k.dim(1)?;
    let head_dim = q.dim(2)?;

    // BF16: optimized tiled attention kernel
    if q.dtype() == candle_core::DType::BF16 {
        return crate::ops::cpu::cpu_prefill_attention(
            q, k, v, &seq_lens, num_heads, num_kv_heads, head_dim, softmax_scale as f64,
        );
    }

    // F32: matmul-based SDPA (per-sequence causal attention)
    matmul_sdpa(q, k, v, &seq_lens, num_heads, num_kv_heads, head_dim, softmax_scale, true)
}

/// Non-causal (bidirectional) varlen attention on CPU.
pub fn varlen_bidirectional(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let cu_q: Vec<u32> = cu_seqlens_q.to_vec1()?;
    let seq_lens: Vec<usize> = cu_q.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let num_heads = q.dim(1)?;
    let num_kv_heads = k.dim(1)?;
    let head_dim = q.dim(2)?;

    matmul_sdpa(q, k, v, &seq_lens, num_heads, num_kv_heads, head_dim, softmax_scale, false)
}

/// Generic matmul-based scaled dot-product attention.
#[allow(clippy::too_many_arguments)]
fn matmul_sdpa(
    q: &Tensor, k: &Tensor, v: &Tensor,
    seq_lens: &[usize],
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    softmax_scale: f32, causal: bool,
) -> Result<Tensor> {
    let gqa_ratio = num_heads / num_kv_heads;
    let mut outputs = Vec::with_capacity(seq_lens.len());
    let mut offset = 0usize;

    for &slen in seq_lens {
        let q_seq = q.narrow(0, offset, slen)?;
        let k_seq = k.narrow(0, offset, slen)?;
        let v_seq = v.narrow(0, offset, slen)?;

        let k_exp = if gqa_ratio > 1 {
            k_seq.unsqueeze(2)?.expand((slen, num_kv_heads, gqa_ratio, head_dim))?
                .reshape((slen, num_heads, head_dim))?
        } else {
            k_seq.clone()
        };
        let v_exp = if gqa_ratio > 1 {
            v_seq.unsqueeze(2)?.expand((slen, num_kv_heads, gqa_ratio, head_dim))?
                .reshape((slen, num_heads, head_dim))?
        } else {
            v_seq.clone()
        };

        // [num_heads, slen, head_dim]
        let q_t = q_seq.transpose(0, 1)?;
        let k_t = k_exp.transpose(0, 1)?;
        let v_t = v_exp.transpose(0, 1)?;

        let scores = q_t.matmul(&k_t.transpose(1, 2)?)?;
        let scores = (scores * (softmax_scale as f64))?;

        let scores = if causal {
            let mut mask_data = vec![0.0f32; slen * slen];
            for i in 0..slen {
                for j in (i + 1)..slen {
                    mask_data[i * slen + j] = f32::NEG_INFINITY;
                }
            }
            let causal_mask = Tensor::from_vec(mask_data, (slen, slen), q.device())?
                .to_dtype(scores.dtype())?;
            scores.broadcast_add(&causal_mask)?
        } else {
            scores
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn_weights.matmul(&v_t)?;
        outputs.push(out.transpose(0, 1)?);
        offset += slen;
    }
    Tensor::cat(&outputs, 0)
}
