//! Built-in naive Ops — uses crate::tensor for basic operations.
//!
//! Always available. Device crates (prelude-cpu, prelude-cuda) register
//! optimized implementations that override this.

use std::sync::{Arc, LazyLock};
use crate::tensor::{DType, Device, Module, Shape, Result, Tensor};
use crate::ops::traits::*;

struct NaiveOps;

pub fn naive_ops() -> &'static Ops {
    static OPS: LazyLock<Ops> = LazyLock::new(|| {
        let n = Arc::new(NaiveOps);
        Ops {
            attn: n.clone(), kv_cache: n.clone(), gemm: n.clone(),
            norm: n.clone(), act: n.clone(), conv: n.clone(),
            comm: n.clone(), fused: n.clone(), session: n.clone(),
            tensor: n,
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
    fn grouped_gemm(&self, _input: &Tensor, _weights: &Tensor, _sorted_token_ids: &Tensor, _sorted_expert_ids: &Tensor, _num_tokens_per_expert: &Tensor) -> Result<Tensor> {
        crate::tensor::bail!("grouped_gemm requires prelude-cuda")
    }
}

impl NormOps for NaiveOps {
    fn rms_norm(&self, x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
        use crate::tensor::{DType, D};
        let xs_f32 = x.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let eps_t = Tensor::new(&[eps], x.device())?.broadcast_as(variance.shape())?;
        let inv_rms = (variance + eps_t)?.sqrt()?.recip()?;
        let normed = xs_f32.broadcast_mul(&inv_rms)?;
        let weight = w.to_dtype(DType::F32)?;
        normed.broadcast_mul(&weight)?.to_dtype(x.dtype())
    }
    fn layer_norm(&self, _x: &Tensor, _w: &Tensor, _b: Option<&Tensor>, _eps: f32) -> Result<Tensor> {
        crate::tensor::bail!("layer_norm: not implemented in naive ops")
    }
    fn group_norm(&self, _x: &Tensor, _w: &Tensor, _b: Option<&Tensor>, _g: usize, _eps: f32) -> Result<Tensor> {
        crate::tensor::bail!("group_norm: not implemented in naive ops")
    }
}

impl ActivationOps for NaiveOps {
    fn silu(&self, x: &Tensor) -> Result<Tensor> { x.silu() }
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

// ── TensorOps (calls candle_core::Tensor directly via .inner()) ─────
//
// Fallback implementation. Device crates override with optimized versions.
// IMPORTANT: uses x.inner() to call candle directly, NOT our Tensor methods,
// because Tensor methods will route through current_ops().tensor (→ infinite recursion).

fn w(r: candle_core::Result<candle_core::Tensor>) -> Result<Tensor> {
    Ok(Tensor::from_candle(r?))
}

impl TensorOps for NaiveOps {
    fn unary(&self, x: &Tensor, op: UnaryOp) -> Result<Tensor> {
        let i = x.inner();
        w(match op {
            UnaryOp::Exp => i.exp(),
            UnaryOp::Log => i.log(),
            UnaryOp::Sin => i.sin(),
            UnaryOp::Cos => i.cos(),
            UnaryOp::Abs => i.abs(),
            UnaryOp::Neg => i.neg(),
            UnaryOp::Sqr => i.sqr(),
            UnaryOp::Sqrt => i.sqrt(),
            UnaryOp::Recip => i.recip(),
            UnaryOp::Tanh => i.tanh(),
            UnaryOp::Relu => i.relu(),
            UnaryOp::Ceil => i.ceil(),
            UnaryOp::Floor => i.floor(),
            UnaryOp::Round => i.round_to(0),
            UnaryOp::Sign => i.sign(),
        })
    }

    fn binary(&self, a: &Tensor, b: &Tensor, op: BinaryOp) -> Result<Tensor> {
        let (ai, bi) = (a.inner(), b.inner());
        w(match op {
            BinaryOp::Add => ai.broadcast_add(bi),
            BinaryOp::Sub => ai.broadcast_sub(bi),
            BinaryOp::Mul => ai.broadcast_mul(bi),
            BinaryOp::Div => ai.broadcast_div(bi),
            BinaryOp::Min => ai.minimum(bi),
            BinaryOp::Max => ai.maximum(bi),
        })
    }

    fn compare(&self, a: &Tensor, b: &Tensor, op: CompareOp) -> Result<Tensor> {
        let (ai, bi) = (a.inner(), b.inner());
        w(match op {
            CompareOp::Eq => ai.eq(bi),
            CompareOp::Ne => ai.ne(bi),
            CompareOp::Lt => ai.lt(bi),
            CompareOp::Gt => ai.gt(bi),
            CompareOp::Ge => ai.ge(bi),
            CompareOp::Le => ai.le(bi),
        })
    }

    fn reduce(&self, x: &Tensor, dim: usize, keepdim: bool, op: ReduceOp) -> Result<Tensor> {
        let i = x.inner();
        w(if keepdim {
            match op {
                ReduceOp::Sum => i.sum_keepdim(dim),
                ReduceOp::Max => i.max_keepdim(dim),
                ReduceOp::Min => i.min_keepdim(dim),
                ReduceOp::ArgMax => i.argmax_keepdim(dim),
                ReduceOp::ArgMin => i.argmin_keepdim(dim),
            }
        } else {
            match op {
                ReduceOp::Sum => i.sum(dim),
                ReduceOp::Max => i.max(dim),
                ReduceOp::Min => i.min(dim),
                ReduceOp::ArgMax => i.argmax(dim),
                ReduceOp::ArgMin => i.argmin(dim),
            }
        })
    }

    fn cast(&self, x: &Tensor, dtype: DType) -> Result<Tensor> { w(x.inner().to_dtype(dtype.into())) }
    fn contiguous(&self, x: &Tensor) -> Result<Tensor> { w(x.inner().contiguous()) }
    fn to_device(&self, x: &Tensor, device: &Device) -> Result<Tensor> { w(x.inner().to_device(device)) }

    fn index_select(&self, x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
        w(x.inner().index_select(indices.inner(), dim))
    }
    fn gather(&self, x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
        w(x.inner().gather(indices.inner(), dim))
    }
    fn scatter_add(&self, x: &Tensor, indices: &Tensor, src: &Tensor, dim: usize) -> Result<Tensor> {
        w(x.inner().scatter_add(indices.inner(), src.inner(), dim))
    }
    fn where_cond(&self, cond: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Result<Tensor> {
        w(cond.inner().where_cond(on_true.inner(), on_false.inner()))
    }
    fn affine(&self, x: &Tensor, mul: f64, add: f64) -> Result<Tensor> {
        w(x.inner().affine(mul, add))
    }
    fn zeros(&self, shape: &Shape, dtype: DType, device: &Device) -> Result<Tensor> {
        let cs: candle_core::Shape = shape.clone().into();
        w(candle_core::Tensor::zeros(cs, dtype.into(), device))
    }
    fn sort_last_dim(&self, x: &Tensor, asc: bool) -> Result<(Tensor, Tensor)> {
        let (a, b) = x.inner().sort_last_dim(asc)?;
        Ok((Tensor::from_candle(a), Tensor::from_candle(b)))
    }
    fn cat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        let inner: Vec<&candle_core::Tensor> = tensors.iter().map(|t| t.inner()).collect();
        w(candle_core::Tensor::cat(&inner, dim))
    }
}
