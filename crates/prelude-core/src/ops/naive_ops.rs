//! Built-in naive Ops — uses crate::tensor for basic operations.
//!
//! Always available. Device crates (prelude-cpu, prelude-cuda) register
//! optimized implementations that override this.

use std::sync::{Arc, LazyLock};
use crate::tensor::{Module, Result, Tensor};
use crate::ops::traits::*;

struct NaiveOps;

pub fn naive_ops() -> &'static Ops {
    static OPS: LazyLock<Ops> = LazyLock::new(|| {
        let n = Arc::new(NaiveOps);
        Ops {
            attn: n.clone(), kv_cache: n.clone(), gemm: n.clone(),
            norm: n.clone(), act: n.clone(), conv: n.clone(),
            comm: n.clone(), fused: n.clone(), session: n,
        }
    });
    &OPS
}

impl AttentionOps for NaiveOps {
    fn name(&self) -> &str { "naive" }

    /// Naive varlen attention via matmul SDPA.
    /// Processes each sequence separately (no fused varlen kernel).
    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, p: &VarlenParams) -> Result<Tensor> {
        use crate::tensor::DType;
        let cu_q: Vec<u32> = p.cu_seqlens_q.to_vec1()?;
        let cu_k: Vec<u32> = p.cu_seqlens_k.to_vec1()?;
        let n_seqs = cu_q.len() - 1;
        let head_dim = q.dim(2)?;
        let mut outputs = Vec::with_capacity(n_seqs);

        for i in 0..n_seqs {
            let q_start = cu_q[i] as usize;
            let q_len = cu_q[i + 1] as usize - q_start;
            let k_start = cu_k[i] as usize;
            let k_len = cu_k[i + 1] as usize - k_start;

            // q_i: [q_len, H, D], k_i: [k_len, H_kv, D], v_i: [k_len, H_kv, D]
            let q_i = q.narrow(0, q_start, q_len)?;
            let k_i = k.narrow(0, k_start, k_len)?;
            let v_i = v.narrow(0, k_start, k_len)?;

            // Transpose to [H, L, D] for batched matmul
            let q_t = q_i.transpose(0, 1)?.to_dtype(DType::F32)?;
            let k_t = k_i.transpose(0, 1)?.to_dtype(DType::F32)?;
            let v_t = v_i.transpose(0, 1)?.to_dtype(DType::F32)?;

            // GQA: expand kv heads
            let n_q_heads = q_t.dim(0)?;
            let n_kv_heads = k_t.dim(0)?;
            let (k_exp, v_exp) = if n_q_heads != n_kv_heads {
                let rep = n_q_heads / n_kv_heads;
                (
                    k_t.unsqueeze(1)?.expand((n_kv_heads, rep, k_len, head_dim))?.reshape((n_q_heads, k_len, head_dim))?,
                    v_t.unsqueeze(1)?.expand((n_kv_heads, rep, k_len, head_dim))?.reshape((n_q_heads, k_len, head_dim))?,
                )
            } else {
                (k_t, v_t)
            };

            // scores = Q @ K^T * scale
            let scores = q_t.matmul(&k_exp.transpose(1, 2)?)?.affine(p.scale as f64, 0.0)?;

            // Causal mask: add -1e9 to future positions
            let masked = if matches!(p.mask, MaskType::Causal) && q_len > 1 {
                let mut mask_data = vec![0.0f32; q_len * k_len];
                for qi in 0..q_len {
                    let max_ki = qi + (k_len - q_len) + 1;
                    for ki in max_ki..k_len {
                        mask_data[qi * k_len + ki] = -1e9;
                    }
                }
                let mask = Tensor::from_vec(mask_data, (q_len, k_len), scores.device())?;
                let mask = mask.unsqueeze(0)?.broadcast_as(scores.dims())?;
                (&scores + &mask)?
            } else {
                scores
            };

            // softmax + attn @ V
            let last_dim = masked.dims().len() - 1;
            let attn_w = self.softmax(&masked, last_dim)?;
            let out = attn_w.matmul(&v_exp)?; // [H, q_len, D]
            let out = out.transpose(0, 1)?.to_dtype(q.dtype())?; // [q_len, H, D]
            outputs.push(out);
        }

        Tensor::cat(&outputs, 0)
    }

    fn paged_attention(&self, _q: &Tensor, _kc: &Tensor, _vc: &Tensor, _p: &PagedParams) -> Result<Tensor> {
        crate::tensor::bail!("paged_attention requires prelude-cuda")
    }
}

impl KvCacheOps for NaiveOps {
    fn reshape_and_cache(&self, _k: &Tensor, _v: &Tensor, _kc: &Tensor, _vc: &Tensor, _sm: &Tensor) -> Result<()> {
        Ok(())
    }
}

impl GemmOps for NaiveOps {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { a.matmul(b) }
    fn quantized_matmul(&self, _a: &Tensor, _b: &Tensor, _sa: Option<&Tensor>, _sb: Option<&Tensor>, _q: QuantScheme) -> Result<Tensor> {
        crate::tensor::bail!("quantized_matmul requires prelude-cpu or prelude-cuda")
    }
    fn moe_gemm(&self, _input: &Tensor, _weights: &Tensor, _topk_weights: &Option<Tensor>, _sorted_token_ids: &Tensor, _sorted_expert_ids: &Tensor, _topk: usize, _is_prefill: bool) -> Result<Tensor> {
        crate::tensor::bail!("moe_gemm requires prelude-cuda")
    }
}

impl NormOps for NaiveOps {
    fn rms_norm(&self, x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
        crate::nn_ops::CandleRmsNorm::new(w.clone(), eps as f64).forward(x)
    }
    fn layer_norm(&self, _x: &Tensor, _w: &Tensor, _b: Option<&Tensor>, _eps: f32) -> Result<Tensor> {
        crate::tensor::bail!("layer_norm: not implemented in naive ops")
    }
    fn group_norm(&self, _x: &Tensor, _w: &Tensor, _b: Option<&Tensor>, _g: usize, _eps: f32) -> Result<Tensor> {
        crate::tensor::bail!("group_norm: not implemented in naive ops")
    }
}

impl ActivationOps for NaiveOps {
    fn silu(&self, x: &Tensor) -> Result<Tensor> { crate::nn_ops::Activation::Silu.forward(x) }
    fn gelu(&self, x: &Tensor) -> Result<Tensor> { x.gelu_erf() }
    fn gelu_approximate(&self, x: &Tensor) -> Result<Tensor> { x.gelu() }
    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        let max = x.max_keepdim(dim)?;
        let exp = x.broadcast_sub(&max)?.exp()?;
        let sum = exp.sum_keepdim(dim)?;
        exp.broadcast_div(&sum)
    }
}

impl ConvOps for NaiveOps {
    fn conv1d(&self, _i: &Tensor, _w: &Tensor, _b: Option<&Tensor>, _s: usize, _p: usize) -> Result<Tensor> {
        crate::tensor::bail!("conv1d: not implemented in naive ops")
    }
    fn conv_transpose1d(&self, _i: &Tensor, _w: &Tensor, _b: Option<&Tensor>, _s: usize, _p: usize, _op: usize) -> Result<Tensor> {
        crate::tensor::bail!("conv_transpose1d: not implemented in naive ops")
    }
    fn conv2d(&self, _i: &Tensor, _w: &Tensor, _b: Option<&Tensor>, _s: [usize; 2], _p: [usize; 2]) -> Result<Tensor> {
        crate::tensor::bail!("conv2d: not implemented in naive ops")
    }
}

impl CommOps for NaiveOps {
    fn world_size(&self) -> usize { 1 }
    fn rank(&self) -> usize { 0 }
    fn all_reduce_sum(&self, x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
    fn all_gather(&self, x: &Tensor, _dim: usize) -> Result<Tensor> { Ok(x.clone()) }
    fn reduce_scatter(&self, x: &Tensor, _dim: usize) -> Result<Tensor> { Ok(x.clone()) }
    fn all_to_all(&self, x: &Tensor, _in_splits: &[usize], _out_splits: &[usize]) -> Result<Tensor> { Ok(x.clone()) }
}

impl FusedOps for NaiveOps {}

impl OpsSession for NaiveOps {
    fn begin_forward(&self) {}
    fn end_forward(&self) {}
}
