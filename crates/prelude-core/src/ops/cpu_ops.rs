//! CPU Ops — implements all 9 Ops traits using pure Rust AVX-512 kernels and oneDNN.

use std::sync::Arc;
use crate::tensor::{DType, Result, Tensor};
use crate::ops::traits::*;

/// CPU implementation of the Ops bundle.
pub struct CpuOps;

impl CpuOps {
    pub fn create() -> Ops {
        let cpu = Arc::new(CpuOps);
        Ops {
            attn: cpu.clone(),
            kv_cache: cpu.clone(),
            gemm: cpu.clone(),
            norm: cpu.clone(),
            act: cpu.clone(),
            conv: cpu.clone(),
            comm: cpu.clone(),
            fused: cpu.clone(),
            session: cpu,
        }
    }
}

/// Returns a lazily-initialized static `&Ops` for CPU-only paths that don't
/// receive an `Ops` from the engine (e.g. `forward_with_cache` for local
/// generation without the server).
pub fn cpu_ops() -> &'static Ops {
    use std::sync::LazyLock;
    static OPS: LazyLock<Ops> = LazyLock::new(CpuOps::create);
    &OPS
}

// ── AttentionOps ────────────────────────────────────────────────────

impl AttentionOps for CpuOps {
    fn name(&self) -> &str { "cpu" }

    fn varlen_attention(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        params: &VarlenParams,
    ) -> Result<Tensor> {
        match &params.mask {
            MaskType::Bidirectional => {
                crate::models::common::attn::cpu::varlen_bidirectional(
                    q, k, v, params.cu_seqlens_q, params.scale,
                )
            }
            _ => {
                // Causal, SlidingWindow (CPU ignores window), Custom
                crate::models::common::attn::cpu::varlen_causal(
                    q, k, v, params.cu_seqlens_q, params.cu_seqlens_k, params.scale,
                )
            }
        }
    }

    fn paged_attention(
        &self,
        _q: &Tensor,
        _key_cache: &Tensor, _value_cache: &Tensor,
        _params: &PagedParams,
    ) -> Result<Tensor> {
        crate::tensor::bail!("paged_attention is not supported on CPU")
    }
}

// ── KvCacheOps ──────────────────────────────────────────────────────

impl KvCacheOps for CpuOps {
    fn reshape_and_cache(
        &self,
        _key: &Tensor, _value: &Tensor,
        _key_cache: &Tensor, _value_cache: &Tensor,
        _slot_mapping: &Tensor,
    ) -> Result<()> {
        crate::tensor::bail!("paged KV cache not supported on CPU")
    }
}

// ── GemmOps ─────────────────────────────────────────────────────────

impl GemmOps for CpuOps {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.matmul(b)
    }

    fn quantized_matmul(
        &self,
        _a: &Tensor, _b: &Tensor,
        _scale_a: Option<&Tensor>, _scale_b: Option<&Tensor>,
        _quant: QuantScheme,
    ) -> Result<Tensor> {
        crate::tensor::bail!("quantized_matmul not supported on CPU (use LinearBackend)")
    }

    fn grouped_gemm(
        &self,
        _input: &Tensor, _weights: &Tensor,
        _sorted_token_ids: &Tensor, _sorted_expert_ids: &Tensor,
        _num_tokens_per_expert: &Tensor,
    ) -> Result<Tensor> {
        crate::tensor::bail!("grouped_gemm not supported on CPU")
    }
}

// ── NormOps ─────────────────────────────────────────────────────────

impl NormOps for CpuOps {
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        crate::ops::cpu::cpu_rmsnorm(x, weight, eps as f64)
    }

    fn layer_norm(
        &self,
        x: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        // Manual layer norm: y = (x - mean) / sqrt(var + eps) * weight + bias
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean = x_f32.mean_keepdim(crate::tensor::D::Minus1)?;
        let centered = x_f32.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(crate::tensor::D::Minus1)?;
        let normed = centered.broadcast_div(&(var + eps as f64)?.sqrt()?)?;
        let normed = normed.to_dtype(x.dtype())?;
        let result = normed.broadcast_mul(weight)?;
        match bias {
            Some(b) => result.broadcast_add(b),
            None => Ok(result),
        }
    }

    fn group_norm(
        &self,
        _x: &Tensor, _weight: &Tensor, _bias: Option<&Tensor>,
        _num_groups: usize, _eps: f32,
    ) -> Result<Tensor> {
        crate::tensor::bail!("group_norm not yet implemented on CPU")
    }
}

// ── ActivationOps ───────────────────────────────────────────────────

impl ActivationOps for CpuOps {
    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        crate::nn_ops::ops::silu(x)
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        x.gelu_erf()
    }

    fn gelu_approximate(&self, x: &Tensor) -> Result<Tensor> {
        x.gelu()
    }

    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        crate::nn_ops::ops::softmax(x, dim)
    }
}

// ── ConvOps ─────────────────────────────────────────────────────────

impl ConvOps for CpuOps {
    fn conv1d(
        &self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
        stride: usize, padding: usize,
    ) -> Result<Tensor> {
        let out = input.conv1d(weight, padding, stride, 1, 1)?;
        match bias {
            Some(b) => out.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?),
            None => Ok(out),
        }
    }

    fn conv_transpose1d(
        &self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
        stride: usize, padding: usize, output_padding: usize,
    ) -> Result<Tensor> {
        let out = input.conv_transpose1d(weight, padding, output_padding, stride, 1, 1)?;
        match bias {
            Some(b) => out.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?),
            None => Ok(out),
        }
    }

    fn conv2d(
        &self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
        stride: [usize; 2], padding: [usize; 2],
    ) -> Result<Tensor> {
        let out = input.conv2d(weight, padding[0], stride[0], 1, 1)?;
        match bias {
            Some(b) => out.broadcast_add(&b.reshape((1, b.dim(0)?, 1, 1))?),
            None => Ok(out),
        }
    }
}

// ── CommOps ─────────────────────────────────────────────────────────

impl CommOps for CpuOps {
    fn world_size(&self) -> usize { 1 }
    fn rank(&self) -> usize { 0 }

    fn all_reduce_sum(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }

    fn all_gather(&self, x: &Tensor, _dim: usize) -> Result<Tensor> {
        Ok(x.clone())
    }

    fn reduce_scatter(&self, x: &Tensor, _dim: usize) -> Result<Tensor> {
        Ok(x.clone())
    }

    fn all_to_all(
        &self, x: &Tensor,
        _input_splits: &[usize], _output_splits: &[usize],
    ) -> Result<Tensor> {
        Ok(x.clone())
    }
}

// ── FusedOps ────────────────────────────────────────────────────────

impl FusedOps for CpuOps {
    fn fused_add_rmsnorm(
        &self, residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> {
        Some(crate::ops::cpu::cpu_fused_add_rmsnorm(x, residual, weight, eps as f64))
    }

    fn fused_silu_mul(&self, gate: &Tensor, up: &Tensor) -> Option<Result<Tensor>> {
        let combined = match Tensor::cat(&[gate, up], gate.dims().len() - 1) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        Some(crate::ops::cpu::cpu_silu_and_mul(&combined))
    }
}

// ── OpsSession ──────────────────────────────────────────────────────

impl OpsSession for CpuOps {}
