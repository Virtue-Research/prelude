//! CpuOps — implements `Ops` with AVX-512 kernels and oneDNN where available.
//!
//! Only overrides methods with optimized implementations.
//! Everything else inherits from `default_impl()` → bare_ops (CubeCL/Device backend).

use prelude_core::ops::traits::*;
use prelude_core::tensor::{DType, Result, Tensor};

pub struct CpuOps;

/// Static CpuOps instance for registration.
pub fn cpu_ops() -> &'static dyn Ops {
    use std::sync::LazyLock;
    static OPS: LazyLock<CpuOps> = LazyLock::new(|| CpuOps);
    &*OPS
}

impl Ops for CpuOps {
    fn default_impl(&self) -> &dyn Ops {
        prelude_core::ops::bare_ops()
    }

    // ── Normalization (AVX-512 optimized) ──────────────────────────

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        crate::ops::cpu_rmsnorm(x, weight, eps as f64)
    }

    fn layer_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean = x_f32.mean_keepdim(prelude_core::tensor::D::Minus1)?;
        let centered = x_f32.broadcast_sub(&mean)?;
        let var = centered
            .sqr()?
            .mean_keepdim(prelude_core::tensor::D::Minus1)?;
        let normed = centered.broadcast_div(&(var + eps as f64)?.sqrt()?)?;
        let normed = normed.to_dtype(x.dtype())?;
        let result = normed.broadcast_mul(weight)?;
        match bias {
            Some(b) => result.broadcast_add(b),
            None => Ok(result),
        }
    }

    // ── Attention (CPU matmul-based) ───────────────────────────────

    fn attn_name(&self) -> &str {
        "cpu"
    }

    fn varlen_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        params: &VarlenParams,
    ) -> Result<Tensor> {
        match &params.mask {
            MaskType::Bidirectional => {
                crate::attn_cpu::varlen_bidirectional(q, k, v, params.cu_seqlens_q, params.scale)
            }
            _ => crate::attn_cpu::varlen_causal(
                q,
                k,
                v,
                params.cu_seqlens_q,
                params.cu_seqlens_k,
                params.scale,
            ),
        }
    }

    // ── Fused ops (AVX-512 optimized) ─────────────────────────────

    fn fused_add_rmsnorm(
        &self,
        residual: &Tensor,
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> {
        Some(crate::ops::cpu_fused_add_rmsnorm(
            x, residual, weight, eps as f64,
        ))
    }

    fn fused_silu_mul(&self, gate: &Tensor, up: &Tensor) -> Option<Result<Tensor>> {
        let combined = match Tensor::cat(&[gate, up], gate.dims().len() - 1) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        Some(crate::ops::cpu_silu_and_mul(&combined))
    }
}
