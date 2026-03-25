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

/// Dispatch: oneDNN path (when available) or candle fallback.
#[allow(clippy::too_many_arguments)]
fn matmul_sdpa(
    q: &Tensor, k: &Tensor, v: &Tensor,
    seq_lens: &[usize],
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    softmax_scale: f32, causal: bool,
) -> Result<Tensor> {
    #[cfg(feature = "onednn")]
    {
        return matmul_sdpa_onednn(q, k, v, seq_lens, num_heads, num_kv_heads, head_dim, softmax_scale, causal);
    }
    #[cfg(not(feature = "onednn"))]
    {
        matmul_sdpa_candle(q, k, v, seq_lens, num_heads, num_kv_heads, head_dim, softmax_scale, causal)
    }
}

// ── oneDNN F32 SDPA ──────────────────────────────────────────────────────

/// F32 SDPA using raw `CpuTensorF32` + per-head oneDNN matmul + custom softmax,
/// parallelized across heads with rayon. No candle compute in the hot path.
#[cfg(feature = "onednn")]
#[allow(clippy::too_many_arguments)]
fn matmul_sdpa_onednn(
    q: &Tensor, k: &Tensor, v: &Tensor,
    seq_lens: &[usize],
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    softmax_scale: f32, causal: bool,
) -> Result<Tensor> {
    use rayon::prelude::*;
    use crate::ops::cpu::buf_tensor::CpuTensorF32;

    crate::ops::onednn::init();

    let gqa_ratio = num_heads / num_kv_heads;

    // Extract raw F32 slices once at the boundary — no candle after this
    let q_cont = q.contiguous()?;
    let k_cont = k.contiguous()?;
    let v_cont = v.contiguous()?;
    let q_all = CpuTensorF32::from_candle(&q_cont)?;
    let k_all = CpuTensorF32::from_candle(&k_cont)?;
    let v_all = CpuTensorF32::from_candle(&v_cont)?;

    let total_tokens: usize = seq_lens.iter().sum();
    let mut out_flat = vec![0.0f32; total_tokens * num_heads * head_dim];
    let out_base_ptr = out_flat.as_mut_ptr() as usize;

    let mut offset = 0usize;
    for &slen in seq_lens {
        let q_seq = q_all.narrow(0, offset, slen); // [slen, num_heads, head_dim]
        let k_seq = k_all.narrow(0, offset, slen); // [slen, num_kv_heads, head_dim]
        let v_seq = v_all.narrow(0, offset, slen);

        let q_data = q_seq.as_slice();
        let k_data = k_seq.as_slice();
        let v_data = v_seq.as_slice();

        let q_row_stride = num_heads * head_dim;
        let kv_row_stride = num_kv_heads * head_dim;

        // Gather per-head contiguous Q/K/V blocks: [slen, head_dim] each
        // Input layout is [slen, num_heads, head_dim] (interleaved heads),
        // so we gather head h's data with stride.
        let mut q_heads = vec![0.0f32; num_heads * slen * head_dim];
        let mut k_heads = vec![0.0f32; num_heads * slen * head_dim];
        let mut v_heads = vec![0.0f32; num_heads * slen * head_dim];

        let head_block = slen * head_dim;
        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;
            for t in 0..slen {
                let q_src = t * q_row_stride + h * head_dim;
                let kv_src = t * kv_row_stride + kv_h * head_dim;
                let dst = h * head_block + t * head_dim;
                q_heads[dst..dst + head_dim].copy_from_slice(&q_data[q_src..q_src + head_dim]);
                k_heads[dst..dst + head_dim].copy_from_slice(&k_data[kv_src..kv_src + head_dim]);
                v_heads[dst..dst + head_dim].copy_from_slice(&v_data[kv_src..kv_src + head_dim]);
            }
        }

        let mut scores_buf = vec![0.0f32; num_heads * slen * slen];
        let mut head_out = vec![0.0f32; num_heads * head_block];

        let scores_ptr = scores_buf.as_mut_ptr() as usize;
        let head_out_ptr = head_out.as_mut_ptr() as usize;
        let head_scores = slen * slen;

        (0..num_heads).into_par_iter().for_each(|h| {
            let q_h = &q_heads[h * head_block..(h + 1) * head_block];
            let k_h = &k_heads[h * head_block..(h + 1) * head_block];
            let v_h = &v_heads[h * head_block..(h + 1) * head_block];
            let s_h = unsafe {
                std::slice::from_raw_parts_mut((scores_ptr as *mut f32).add(h * head_scores), head_scores)
            };
            let o_h = unsafe {
                std::slice::from_raw_parts_mut((head_out_ptr as *mut f32).add(h * head_block), head_block)
            };

            // QK^T: [slen, head_dim] @ [slen, head_dim]^T → [slen, slen]
            unsafe {
                crate::ops::onednn::ffi::onednn_f32_linear(
                    q_h.as_ptr() as *const _, k_h.as_ptr() as *const _,
                    s_h.as_mut_ptr() as *mut _,
                    slen as i64, head_dim as i64, slen as i64,
                );
            }

            if causal {
                for i in 0..slen {
                    for j in 0..=i { s_h[i * slen + j] *= softmax_scale; }
                    for j in (i + 1)..slen { s_h[i * slen + j] = f32::NEG_INFINITY; }
                }
            } else {
                for v in s_h.iter_mut() { *v *= softmax_scale; }
            }

            crate::ops::cpu::softmax::softmax_f32_inplace(s_h, slen, slen);

            // score @ V: [slen, slen] @ [slen, head_dim] → [slen, head_dim]
            unsafe {
                crate::ops::onednn::ffi::onednn_f32_matmul(
                    s_h.as_ptr() as *const _, v_h.as_ptr() as *const _,
                    o_h.as_mut_ptr() as *mut _,
                    slen as i64, slen as i64, head_dim as i64,
                );
            }
        });

        // Scatter back: [num_heads, slen, head_dim] → [slen, num_heads, head_dim]
        let out_seq = unsafe {
            std::slice::from_raw_parts_mut(
                (out_base_ptr as *mut f32).add(offset * num_heads * head_dim),
                slen * num_heads * head_dim,
            )
        };
        for h in 0..num_heads {
            for t in 0..slen {
                let src = h * head_block + t * head_dim;
                let dst = t * q_row_stride + h * head_dim;
                out_seq[dst..dst + head_dim].copy_from_slice(&head_out[src..src + head_dim]);
            }
        }

        offset += slen;
    }

    // Single candle Tensor construction at the very end
    Tensor::from_vec(out_flat, &[total_tokens, num_heads, head_dim], q.device())
}

// ── F32 cross-attention (Q and KV may have different lengths) ────────────

/// F32 cross-attention using oneDNN: Q attends to full KV cache.
/// Q: [seq_q, num_heads, head_dim], K: [seq_kv, num_kv_heads, head_dim],
/// V: [seq_kv, num_kv_heads, head_dim].
/// `position_offset` is the KV position of Q's first token (for causal mask).
/// Returns: [seq_q, num_heads, head_dim]
#[cfg(feature = "onednn")]
#[allow(clippy::too_many_arguments)]
pub fn cross_attention_f32_onednn(
    q: &Tensor, k: &Tensor, v: &Tensor,
    seq_q: usize, seq_kv: usize,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    softmax_scale: f32, causal: bool, position_offset: usize,
) -> Result<Tensor> {
    use rayon::prelude::*;
    use crate::ops::cpu::buf_tensor::CpuTensorF32;

    crate::ops::onednn::init();

    let gqa_ratio = num_heads / num_kv_heads;

    let q_cont = q.contiguous()?;
    let k_cont = k.contiguous()?;
    let v_cont = v.contiguous()?;
    let q_all = CpuTensorF32::from_candle(&q_cont)?;
    let k_all = CpuTensorF32::from_candle(&k_cont)?;
    let v_all = CpuTensorF32::from_candle(&v_cont)?;

    let q_data = q_all.as_slice();
    let k_data = k_all.as_slice();
    let v_data = v_all.as_slice();

    let q_row_stride = num_heads * head_dim;
    let kv_row_stride = num_kv_heads * head_dim;

    let head_q_block = seq_q * head_dim;
    let head_kv_block = seq_kv * head_dim;

    let mut q_heads = vec![0.0f32; num_heads * head_q_block];
    let mut k_heads = vec![0.0f32; num_heads * head_kv_block];
    let mut v_heads = vec![0.0f32; num_heads * head_kv_block];

    for h in 0..num_heads {
        let kv_h = h / gqa_ratio;
        for t in 0..seq_q {
            let q_src = t * q_row_stride + h * head_dim;
            let dst = h * head_q_block + t * head_dim;
            q_heads[dst..dst + head_dim].copy_from_slice(&q_data[q_src..q_src + head_dim]);
        }
        for t in 0..seq_kv {
            let kv_src = t * kv_row_stride + kv_h * head_dim;
            let k_dst = h * head_kv_block + t * head_dim;
            k_heads[k_dst..k_dst + head_dim].copy_from_slice(&k_data[kv_src..kv_src + head_dim]);
            v_heads[k_dst..k_dst + head_dim].copy_from_slice(&v_data[kv_src..kv_src + head_dim]);
        }
    }

    let head_scores = seq_q * seq_kv;
    let mut scores_buf = vec![0.0f32; num_heads * head_scores];
    let mut head_out = vec![0.0f32; num_heads * head_q_block];

    let scores_ptr = scores_buf.as_mut_ptr() as usize;
    let head_out_ptr = head_out.as_mut_ptr() as usize;

    (0..num_heads).into_par_iter().for_each(|h| {
        let q_h = &q_heads[h * head_q_block..(h + 1) * head_q_block];
        let k_h = &k_heads[h * head_kv_block..(h + 1) * head_kv_block];
        let v_h = &v_heads[h * head_kv_block..(h + 1) * head_kv_block];
        let s_h = unsafe {
            std::slice::from_raw_parts_mut((scores_ptr as *mut f32).add(h * head_scores), head_scores)
        };
        let o_h = unsafe {
            std::slice::from_raw_parts_mut((head_out_ptr as *mut f32).add(h * head_q_block), head_q_block)
        };

        unsafe {
            crate::ops::onednn::ffi::onednn_f32_linear(
                q_h.as_ptr() as *const _, k_h.as_ptr() as *const _,
                s_h.as_mut_ptr() as *mut _,
                seq_q as i64, head_dim as i64, seq_kv as i64,
            );
        }

        if causal {
            for i in 0..seq_q {
                for j in 0..seq_kv {
                    if j <= position_offset + i {
                        s_h[i * seq_kv + j] *= softmax_scale;
                    } else {
                        s_h[i * seq_kv + j] = f32::NEG_INFINITY;
                    }
                }
            }
        } else {
            for val in s_h.iter_mut() { *val *= softmax_scale; }
        }

        crate::ops::cpu::softmax::softmax_f32_inplace(s_h, seq_q, seq_kv);

        unsafe {
            crate::ops::onednn::ffi::onednn_f32_matmul(
                s_h.as_ptr() as *const _, v_h.as_ptr() as *const _,
                o_h.as_mut_ptr() as *mut _,
                seq_q as i64, seq_kv as i64, head_dim as i64,
            );
        }
    });

    let mut out_flat = vec![0.0f32; seq_q * num_heads * head_dim];
    for h in 0..num_heads {
        for t in 0..seq_q {
            let src = h * head_q_block + t * head_dim;
            let dst = t * q_row_stride + h * head_dim;
            out_flat[dst..dst + head_dim].copy_from_slice(&head_out[src..src + head_dim]);
        }
    }

    Tensor::from_vec(out_flat, &[seq_q, num_heads, head_dim], q.device())
}

// ── Candle reference SDPA ────────────────────────────────────────────────

/// Candle-based F32 SDPA (reference / fallback when oneDNN is unavailable).
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn matmul_sdpa_candle(
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

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn deterministic_f32_tensor(shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as f64 + seed as f64) * 0.0073;
                (x.sin() * 0.5) as f32
            })
            .collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn test_f32_sdpa_candle_vs_onednn() {
        let seq_len = 32;
        let num_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = deterministic_f32_tensor(&[seq_len, num_heads, head_dim], 1);
        let k = deterministic_f32_tensor(&[seq_len, num_kv_heads, head_dim], 2);
        let v = deterministic_f32_tensor(&[seq_len, num_kv_heads, head_dim], 3);

        let candle_out = matmul_sdpa_candle(
            &q, &k, &v, &[seq_len], num_heads, num_kv_heads, head_dim, scale, true,
        ).unwrap();

        #[cfg(feature = "onednn")]
        {
            let onednn_out = matmul_sdpa_onednn(
                &q, &k, &v, &[seq_len], num_heads, num_kv_heads, head_dim, scale, true,
            ).unwrap();

            let c: Vec<f32> = candle_out.flatten_all().unwrap().to_vec1().unwrap();
            let o: Vec<f32> = onednn_out.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(c.len(), o.len());

            let max_diff = c.iter().zip(o.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("correctness: max_diff = {max_diff:.2e} ({} elements)", c.len());
            assert!(max_diff < 1e-4, "max diff {max_diff} exceeds tolerance");
        }
    }

    #[test]
    fn test_f32_sdpa_varlen() {
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = 32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_lens = [8, 16, 4];
        let total: usize = seq_lens.iter().sum();

        let q = deterministic_f32_tensor(&[total, num_heads, head_dim], 10);
        let k = deterministic_f32_tensor(&[total, num_kv_heads, head_dim], 20);
        let v = deterministic_f32_tensor(&[total, num_kv_heads, head_dim], 30);

        let candle_out = matmul_sdpa_candle(
            &q, &k, &v, &seq_lens, num_heads, num_kv_heads, head_dim, scale, true,
        ).unwrap();

        #[cfg(feature = "onednn")]
        {
            let onednn_out = matmul_sdpa_onednn(
                &q, &k, &v, &seq_lens, num_heads, num_kv_heads, head_dim, scale, true,
            ).unwrap();

            let c: Vec<f32> = candle_out.flatten_all().unwrap().to_vec1().unwrap();
            let o: Vec<f32> = onednn_out.flatten_all().unwrap().to_vec1().unwrap();
            let max_diff = c.iter().zip(o.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("varlen correctness: max_diff = {max_diff:.2e}");
            assert!(max_diff < 1e-4, "max diff {max_diff} exceeds tolerance");
        }
    }

    #[test]
    fn test_f32_sdpa_benchmark() {
        let seq_len = 512;
        let num_heads = 32;
        let num_kv_heads = 8;
        let head_dim = 128;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let warmup = 3;
        let iters = 10;

        let q = deterministic_f32_tensor(&[seq_len, num_heads, head_dim], 42);
        let k = deterministic_f32_tensor(&[seq_len, num_kv_heads, head_dim], 43);
        let v = deterministic_f32_tensor(&[seq_len, num_kv_heads, head_dim], 44);

        for _ in 0..warmup {
            let _ = matmul_sdpa_candle(
                &q, &k, &v, &[seq_len], num_heads, num_kv_heads, head_dim, scale, true,
            );
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = matmul_sdpa_candle(
                &q, &k, &v, &[seq_len], num_heads, num_kv_heads, head_dim, scale, true,
            );
        }
        let candle_us = t0.elapsed().as_micros() / iters as u128;

        #[cfg(feature = "onednn")]
        {
            for _ in 0..warmup {
                let _ = matmul_sdpa_onednn(
                    &q, &k, &v, &[seq_len], num_heads, num_kv_heads, head_dim, scale, true,
                );
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = matmul_sdpa_onednn(
                    &q, &k, &v, &[seq_len], num_heads, num_kv_heads, head_dim, scale, true,
                );
            }
            let onednn_us = t0.elapsed().as_micros() / iters as u128;

            eprintln!();
            eprintln!("=== F32 SDPA Benchmark (seq={seq_len}, H={num_heads}, Hkv={num_kv_heads}, D={head_dim}) ===");
            eprintln!("candle:  {candle_us} us/iter");
            eprintln!("oneDNN:  {onednn_us} us/iter");
            if onednn_us > 0 {
                let speedup = candle_us as f64 / onednn_us as f64;
                eprintln!("speedup: {speedup:.2}x");
            }
        }

        #[cfg(not(feature = "onednn"))]
        {
            eprintln!("candle-only: {candle_us} us/iter (onednn feature disabled)");
        }
    }
}
