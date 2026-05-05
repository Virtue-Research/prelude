//! CPU attention fallback (matmul SDPA / tiled BF16).
//!
//! Used when no GPU attention backend is compiled or when running on CPU.

use prelude_core::tensor::{Result, Tensor};

/// Causal varlen attention on CPU.
/// BF16: optimized tiled kernel (when Q and K have equal lengths).
/// F32 or cross-attention: matmul SDPA.
pub fn varlen_causal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let cu_q: Vec<u32> = cu_seqlens_q.to_vec1()?;
    let cu_k: Vec<u32> = cu_seqlens_k.to_vec1()?;
    let seq_lens_q: Vec<usize> = cu_q.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let seq_lens_k: Vec<usize> = cu_k.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let num_heads = q.dim(1)?;
    let num_kv_heads = k.dim(1)?;
    let head_dim = q.dim(2)?;

    let same_lengths = seq_lens_q == seq_lens_k;

    // BF16 fast path: only when Q and K have equal per-sequence lengths
    // (the tiled kernel assumes Q and K share the same seq_lens).
    if q.dtype() == prelude_core::tensor::DType::BF16 && same_lengths {
        return crate::ops::cpu_prefill_attention(
            q,
            k,
            v,
            &seq_lens_q,
            num_heads,
            num_kv_heads,
            head_dim,
            softmax_scale as f64,
        );
    }

    // Cross-attention or F32: per-sequence matmul SDPA with proper Q/K lengths.
    // Convert to F32 if needed (BF16 cross-attention).
    let q_f32 = q.to_dtype(prelude_core::tensor::DType::F32)?;
    let k_f32 = k.to_dtype(prelude_core::tensor::DType::F32)?;
    let v_f32 = v.to_dtype(prelude_core::tensor::DType::F32)?;
    let out = matmul_sdpa_cross(
        &q_f32,
        &k_f32,
        &v_f32,
        &seq_lens_q,
        &seq_lens_k,
        num_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        true,
    )?;
    out.to_dtype(q.dtype())
}

/// Non-causal (bidirectional) varlen attention on CPU.
pub fn varlen_bidirectional(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let cu_q: Vec<u32> = cu_seqlens_q.to_vec1()?;
    let seq_lens: Vec<usize> = cu_q.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let num_heads = q.dim(1)?;
    let num_kv_heads = k.dim(1)?;
    let head_dim = q.dim(2)?;

    // Always use F32 path (the BF16 tiled kernel only supports causal mask)
    let q_f32 = q.to_dtype(prelude_core::tensor::DType::F32)?;
    let k_f32 = k.to_dtype(prelude_core::tensor::DType::F32)?;
    let v_f32 = v.to_dtype(prelude_core::tensor::DType::F32)?;
    let out = matmul_sdpa(
        &q_f32,
        &k_f32,
        &v_f32,
        &seq_lens,
        num_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        false,
    )?;
    out.to_dtype(q.dtype())
}

/// Dispatch: oneDNN path (when available) or naive fallback.
#[allow(clippy::too_many_arguments)]
fn matmul_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_lens: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    matmul_sdpa_onednn(
        q,
        k,
        v,
        seq_lens,
        num_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        causal,
    )
}

/// Dispatch for cross-attention: Q and K may have different per-sequence lengths.
/// Falls back to per-sequence cross_attention_f32_onednn.
#[allow(clippy::too_many_arguments)]
fn matmul_sdpa_cross(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_lens_q: &[usize],
    seq_lens_k: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let batch = seq_lens_q.len();
    let mut outputs = Vec::with_capacity(batch);
    let mut q_offset = 0usize;
    let mut k_offset = 0usize;

    for i in 0..batch {
        let sq = seq_lens_q[i];
        let sk = seq_lens_k[i];
        let q_seq = q.narrow(0, q_offset, sq)?;
        let k_seq = k.narrow(0, k_offset, sk)?;
        let v_seq = v.narrow(0, k_offset, sk)?;

        let position_offset = sk.saturating_sub(sq); // causal offset for cross-attention
        let out = cross_attention_f32_onednn(
            &q_seq,
            &k_seq,
            &v_seq,
            sq,
            sk,
            num_heads,
            num_kv_heads,
            head_dim,
            softmax_scale,
            causal,
            position_offset,
        )?;
        outputs.push(out);
        q_offset += sq;
        k_offset += sk;
    }

    if outputs.len() == 1 {
        Ok(outputs.into_iter().next().unwrap())
    } else {
        Tensor::cat(&outputs, 0)
    }
}

// ── oneDNN F32 SDPA ──────────────────────────────────────────────────────

/// F32 SDPA using raw `CpuTensorF32` + per-head oneDNN matmul + custom softmax,
/// parallelized across heads with rayon. No tensor overhead in the hot path.
#[allow(clippy::too_many_arguments)]
fn matmul_sdpa_onednn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_lens: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    use crate::ops::buf_tensor::CpuTensorF32;
    use rayon::prelude::*;

    crate::onednn::init();

    let gqa_ratio = num_heads / num_kv_heads;

    // Extract raw F32 slices once at the boundary
    let q_cont = q.contiguous()?;
    let k_cont = k.contiguous()?;
    let v_cont = v.contiguous()?;
    let q_all = CpuTensorF32::from_tensor(&q_cont)?;
    let k_all = CpuTensorF32::from_tensor(&k_cont)?;
    let v_all = CpuTensorF32::from_tensor(&v_cont)?;

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

        let head_scores = slen * slen;

        scores_buf
            .par_chunks_mut(head_scores)
            .zip(head_out.par_chunks_mut(head_block))
            .enumerate()
            .for_each(|(h, (s_h, o_h))| {
                let q_h = &q_heads[h * head_block..(h + 1) * head_block];
                let k_h = &k_heads[h * head_block..(h + 1) * head_block];
                let v_h = &v_heads[h * head_block..(h + 1) * head_block];

                // QK^T: [slen, head_dim] @ [slen, head_dim]^T → [slen, slen]
                unsafe {
                    crate::onednn::ffi::onednn_f32_linear(
                        q_h.as_ptr() as *const _,
                        k_h.as_ptr() as *const _,
                        s_h.as_mut_ptr() as *mut _,
                        slen as i64,
                        head_dim as i64,
                        slen as i64,
                    );
                }

                if causal {
                    for i in 0..slen {
                        for j in 0..=i {
                            s_h[i * slen + j] *= softmax_scale;
                        }
                        for j in (i + 1)..slen {
                            s_h[i * slen + j] = f32::NEG_INFINITY;
                        }
                    }
                } else {
                    for v in s_h.iter_mut() {
                        *v *= softmax_scale;
                    }
                }

                crate::ops::softmax::softmax_f32_inplace(s_h, slen, slen);

                // score @ V: [slen, slen] @ [slen, head_dim] → [slen, head_dim]
                unsafe {
                    crate::onednn::ffi::onednn_f32_matmul(
                        s_h.as_ptr() as *const _,
                        v_h.as_ptr() as *const _,
                        o_h.as_mut_ptr() as *mut _,
                        slen as i64,
                        slen as i64,
                        head_dim as i64,
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

    // Single Tensor construction at the very end
    Tensor::from_vec(out_flat, &[total_tokens, num_heads, head_dim], q.device())
}

// ── F32 cross-attention (Q and KV may have different lengths) ────────────

/// F32 cross-attention using oneDNN: Q attends to full KV cache.
/// Q: [seq_q, num_heads, head_dim], K: [seq_kv, num_kv_heads, head_dim],
/// V: [seq_kv, num_kv_heads, head_dim].
/// `position_offset` is the KV position of Q's first token (for causal mask).
/// Returns: [seq_q, num_heads, head_dim]
#[allow(clippy::too_many_arguments)]
pub fn cross_attention_f32_onednn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_q: usize,
    seq_kv: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
    causal: bool,
    position_offset: usize,
) -> Result<Tensor> {
    use crate::ops::buf_tensor::CpuTensorF32;
    use rayon::prelude::*;

    crate::onednn::init();

    let gqa_ratio = num_heads / num_kv_heads;

    let q_cont = q.contiguous()?;
    let k_cont = k.contiguous()?;
    let v_cont = v.contiguous()?;
    let q_all = CpuTensorF32::from_tensor(&q_cont)?;
    let k_all = CpuTensorF32::from_tensor(&k_cont)?;
    let v_all = CpuTensorF32::from_tensor(&v_cont)?;

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

    scores_buf
        .par_chunks_mut(head_scores)
        .zip(head_out.par_chunks_mut(head_q_block))
        .enumerate()
        .for_each(|(h, (s_h, o_h))| {
            let q_h = &q_heads[h * head_q_block..(h + 1) * head_q_block];
            let k_h = &k_heads[h * head_kv_block..(h + 1) * head_kv_block];
            let v_h = &v_heads[h * head_kv_block..(h + 1) * head_kv_block];

            unsafe {
                crate::onednn::ffi::onednn_f32_linear(
                    q_h.as_ptr() as *const _,
                    k_h.as_ptr() as *const _,
                    s_h.as_mut_ptr() as *mut _,
                    seq_q as i64,
                    head_dim as i64,
                    seq_kv as i64,
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
                for val in s_h.iter_mut() {
                    *val *= softmax_scale;
                }
            }

            crate::ops::softmax::softmax_f32_inplace(s_h, seq_q, seq_kv);

            unsafe {
                crate::onednn::ffi::onednn_f32_matmul(
                    s_h.as_ptr() as *const _,
                    v_h.as_ptr() as *const _,
                    o_h.as_mut_ptr() as *mut _,
                    seq_q as i64,
                    seq_kv as i64,
                    head_dim as i64,
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

// ── Reference SDPA ──────────────────────────────────────────────────────

/// Reference F32 SDPA (fallback when oneDNN is unavailable).
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn matmul_sdpa_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_lens: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let gqa_ratio = num_heads / num_kv_heads;
    let mut outputs = Vec::with_capacity(seq_lens.len());
    let mut offset = 0usize;

    for &slen in seq_lens {
        let q_seq = q.narrow(0, offset, slen)?;
        let k_seq = k.narrow(0, offset, slen)?;
        let v_seq = v.narrow(0, offset, slen)?;

        let k_exp = if gqa_ratio > 1 {
            k_seq
                .unsqueeze(2)?
                .expand((slen, num_kv_heads, gqa_ratio, head_dim))?
                .reshape((slen, num_heads, head_dim))?
        } else {
            k_seq.clone()
        };
        let v_exp = if gqa_ratio > 1 {
            v_seq
                .unsqueeze(2)?
                .expand((slen, num_kv_heads, gqa_ratio, head_dim))?
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
            let causal_mask =
                Tensor::from_vec(mask_data, (slen, slen), q.device())?.to_dtype(scores.dtype())?;
            scores.broadcast_add(&causal_mask)?
        } else {
            scores
        };

        let attn_weights = prelude_core::tensor::softmax(&scores, scores.rank() - 1)?;
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
    use prelude_core::tensor::{Device, Tensor};

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
    fn test_f32_sdpa_reference_vs_onednn() {
        let seq_len = 32;
        let num_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = deterministic_f32_tensor(&[seq_len, num_heads, head_dim], 1);
        let k = deterministic_f32_tensor(&[seq_len, num_kv_heads, head_dim], 2);
        let v = deterministic_f32_tensor(&[seq_len, num_kv_heads, head_dim], 3);

        let ref_out = matmul_sdpa_reference(
            &q,
            &k,
            &v,
            &[seq_len],
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            true,
        )
        .unwrap();

        {
            let onednn_out = matmul_sdpa_onednn(
                &q,
                &k,
                &v,
                &[seq_len],
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                true,
            )
            .unwrap();

            let c: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();
            let o: Vec<f32> = onednn_out.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(c.len(), o.len());

            let max_diff = c
                .iter()
                .zip(o.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "correctness: max_diff = {max_diff:.2e} ({} elements)",
                c.len()
            );
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

        let ref_out = matmul_sdpa_reference(
            &q,
            &k,
            &v,
            &seq_lens,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            true,
        )
        .unwrap();

        {
            let onednn_out = matmul_sdpa_onednn(
                &q,
                &k,
                &v,
                &seq_lens,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                true,
            )
            .unwrap();

            let c: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();
            let o: Vec<f32> = onednn_out.flatten_all().unwrap().to_vec1().unwrap();
            let max_diff = c
                .iter()
                .zip(o.iter())
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
            let _ = matmul_sdpa_reference(
                &q,
                &k,
                &v,
                &[seq_len],
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                true,
            );
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = matmul_sdpa_reference(
                &q,
                &k,
                &v,
                &[seq_len],
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                true,
            );
        }
        let ref_us = t0.elapsed().as_micros() / iters as u128;

        {
            for _ in 0..warmup {
                let _ = matmul_sdpa_onednn(
                    &q,
                    &k,
                    &v,
                    &[seq_len],
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    true,
                );
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = matmul_sdpa_onednn(
                    &q,
                    &k,
                    &v,
                    &[seq_len],
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    true,
                );
            }
            let onednn_us = t0.elapsed().as_micros() / iters as u128;

            eprintln!();
            eprintln!(
                "=== F32 SDPA Benchmark (seq={seq_len}, H={num_heads}, Hkv={num_kv_heads}, D={head_dim}) ==="
            );
            eprintln!("reference: {ref_us} us/iter");
            eprintln!("oneDNN:  {onednn_us} us/iter");
            if onednn_us > 0 {
                let speedup = ref_us as f64 / onednn_us as f64;
                eprintln!("speedup: {speedup:.2}x");
            }
        }
    }
}
