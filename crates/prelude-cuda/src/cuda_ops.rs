//! CudaOps — implements all 9 Ops traits for CUDA devices.
//!
//! This is the CUDA device implementation of the Ops bundle.
//! It wraps the GPU kernel functions in `crate::ops::*` and attention backends
//! in `crate::attn::*`, exposing them through the Ops trait interface.

use std::sync::Arc;
use candle_core::{DType, Result, Tensor};
use prelude_core::ops::traits::*;

/// Factory: create the full Ops bundle for CUDA devices.
///
/// Called by the composition root (prelude-server) at startup.
/// Also registers the GPU GEMM dispatch with candle-core.
pub fn create_cuda_ops() -> Ops {
    // Register GPU GEMM dispatch so Tensor::matmul() routes through CUTLASS/DeepGEMM.
    crate::ops::gemm::register_gpu_gemm();

    let cuda = Arc::new(CudaOps);
    let attn: Arc<dyn AttentionOps> = select_attention_backend();
    Ops {
        attn,
        kv_cache: cuda.clone(),
        gemm: cuda.clone(),
        norm: cuda.clone(),
        act: cuda.clone(),
        conv: cuda.clone(),
        comm: cuda.clone(),
        fused: cuda.clone(),
        session: Arc::new(CudaSession),
    }
}

pub struct CudaOps;

// ── GemmOps ─────────────────────────────────────────────────────────

impl GemmOps for CudaOps {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Uses registered CUTLASS/DeepGEMM dispatch via candle-core
        a.matmul(b)
    }

    fn quantized_matmul(
        &self,
        _a: &Tensor, _b: &Tensor,
        _scale_a: Option<&Tensor>, _scale_b: Option<&Tensor>,
        _quant: QuantScheme,
    ) -> Result<Tensor> {
        candle_core::bail!("quantized_matmul not yet implemented in CudaOps")
    }

    fn grouped_gemm(
        &self,
        _input: &Tensor, _weights: &Tensor,
        _sorted_token_ids: &Tensor, _sorted_expert_ids: &Tensor,
        _num_tokens_per_expert: &Tensor,
    ) -> Result<Tensor> {
        candle_core::bail!("grouped_gemm not yet implemented in CudaOps")
    }
}

// ── KvCacheOps ──────────────────────────────────────────────────────

impl KvCacheOps for CudaOps {
    fn reshape_and_cache(
        &self,
        key: &Tensor, value: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        crate::ops::kv_cache::scatter_kv_cache_flash(key, value, key_cache, value_cache, slot_mapping)
    }
}

// ── NormOps ─────────────────────────────────────────────────────────

impl NormOps for CudaOps {
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        crate::ops::rmsnorm::fast_rmsnorm(x, weight, eps as f64)
    }

    fn layer_norm(
        &self,
        x: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        // Manual layer norm (candle_nn is not a dep here)
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean = x_f32.mean_keepdim(candle_core::D::Minus1)?;
        let centered = x_f32.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
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
        candle_core::bail!("group_norm not yet implemented on CUDA")
    }
}

// ── ActivationOps ───────────────────────────────────────────────────

impl ActivationOps for CudaOps {
    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        // SiLU: x * sigmoid(x)
        let sigmoid = (x.neg()?.exp()? + 1.0)?.recip()?;
        x.mul(&sigmoid)
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        x.gelu_erf()
    }

    fn gelu_approximate(&self, x: &Tensor) -> Result<Tensor> {
        x.gelu()
    }

    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        let max = x.max_keepdim(dim)?;
        let shifted = x.broadcast_sub(&max)?;
        let exp = shifted.exp()?;
        let sum = exp.sum_keepdim(dim)?;
        exp.broadcast_div(&sum)
    }
}

// ── ConvOps ─────────────────────────────────────────────────────────

impl ConvOps for CudaOps {
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

// ── CommOps (single-GPU for now) ────────────────────────────────────

impl CommOps for CudaOps {
    fn world_size(&self) -> usize { 1 }
    fn rank(&self) -> usize { 0 }

    fn all_reduce_sum(&self, x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
    fn all_gather(&self, x: &Tensor, _dim: usize) -> Result<Tensor> { Ok(x.clone()) }
    fn reduce_scatter(&self, x: &Tensor, _dim: usize) -> Result<Tensor> { Ok(x.clone()) }
    fn all_to_all(&self, x: &Tensor, _: &[usize], _: &[usize]) -> Result<Tensor> { Ok(x.clone()) }
}

// ── FusedOps ────────────────────────────────────────────────────────

impl FusedOps for CudaOps {
    fn fused_add_rmsnorm(
        &self, residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> {
        Some(crate::ops::rmsnorm::fused_add_rmsnorm(x, residual, weight, eps as f64))
    }

    fn fused_silu_mul(&self, gate: &Tensor, up: &Tensor) -> Option<Result<Tensor>> {
        Some(crate::ops::elementwise::fused_silu_mul(gate, up))
    }

    fn fused_add(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> {
        Some(crate::ops::elementwise::vectorized_add(a, b))
    }

    fn fused_qknorm_rope(
        &self,
        q: &Tensor, k: &Tensor,
        q_weight: &Tensor, k_weight: &Tensor,
        cos: &Tensor, sin: &Tensor,
        position_ids: &Tensor,
        eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> {
        let q_out = match crate::ops::rope::fused_qknorm_rope_varlen(
            q, q_weight, cos, sin, position_ids, eps as f64,
        ) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        let k_out = match crate::ops::rope::fused_qknorm_rope_varlen(
            k, k_weight, cos, sin, position_ids, eps as f64,
        ) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        Some(Ok((q_out, k_out)))
    }

    fn fused_knorm_rope_cache_write(
        &self,
        k: &Tensor, v: &Tensor,
        k_weight: &Tensor,
        cos: &Tensor, sin: &Tensor,
        position_ids: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &Tensor,
        eps: f32,
    ) -> Option<Result<()>> {
        if !crate::ops::kv_cache::fused_kv_cache_write_enabled() {
            return None;
        }
        // Extract dims from tensors
        let k_dims = k.dims();
        let kc_dims = key_cache.dims();
        let num_kv_heads = if k_dims.len() == 3 { k_dims[1] } else { return None };
        let head_dim = if k_dims.len() == 3 { k_dims[2] } else { return None };
        let block_size = if kc_dims.len() == 4 { kc_dims[1] } else { return None };
        Some(crate::ops::kv_cache::fused_knorm_rope_kv_cache_write_varlen(
            k, v, k_weight, cos, sin, position_ids,
            key_cache, value_cache, slot_mapping,
            num_kv_heads, head_dim, block_size,
            eps as f64,
        ))
    }

    fn fused_moe_routing(
        &self,
        gate_logits: &Tensor,
        top_k: usize,
    ) -> Option<Result<(Tensor, Tensor, Tensor, Tensor)>> {
        Some(crate::ops::moe::fused_moe_routing(gate_logits, top_k, true))
    }
}

// ── OpsSession ──────────────────────────────────────────────────────

pub struct CudaSession;

impl OpsSession for CudaSession {
    fn begin_forward(&self) {
        #[cfg(feature = "flashinfer")]
        crate::attn::flashinfer::begin_forward();
    }

    fn end_forward(&self) {
        #[cfg(feature = "flashinfer")]
        crate::attn::flashinfer::end_forward();
    }

    fn precompute_paged_plan(
        &self,
        _block_tables: &Tensor,
        _cu_seqlens_k: &Tensor,
        _block_size: usize,
    ) -> Result<()> {
        #[cfg(feature = "flashinfer")]
        crate::attn::flashinfer::precompute_paged_plan(
            _block_tables, _cu_seqlens_k, _block_size,
        )?;
        Ok(())
    }
}

// ── Attention backend selection ─────────────────────────────────────

/// Select the best GPU attention backend based on compiled features.
fn select_attention_backend() -> Arc<dyn AttentionOps + 'static> {
    #[cfg(feature = "flash-attn-v4")]
    {
        tracing::info!(backend = "flash-attn-v4", "attention backend selected");
        return Arc::new(FlashAttnV4Ops);
    }

    #[cfg(feature = "flashinfer")]
    {
        tracing::info!(backend = "flashinfer", "attention backend selected");
        return Arc::new(FlashInferOps);
    }

    #[cfg(feature = "flash-attn-v3")]
    {
        tracing::info!(backend = "flash-attn-v3", "attention backend selected");
        return Arc::new(FlashAttnV3Ops);
    }

    #[cfg(feature = "flash-attn")]
    {
        tracing::info!(backend = "flash-attn-v2", "attention backend selected");
        return Arc::new(FlashAttnV2Ops);
    }

    #[allow(unreachable_code)]
    {
        tracing::warn!("no GPU attention backend compiled, falling back to CPU attn ops");
        Arc::new(prelude_core::ops::CpuOps)
    }
}

// ── Shared dispatch helper ──────────────────────────────────────────

/// Dispatch varlen_attention based on MaskType for backends that have
/// separate causal/windowed/bidirectional entry points.
macro_rules! dispatch_varlen {
    ($mod:path, $q:expr, $k:expr, $v:expr, $p:expr) => {
        match &$p.mask {
            MaskType::Causal => {
                $mod::varlen_causal(
                    $q, $k, $v, $p.cu_seqlens_q, $p.cu_seqlens_k,
                    $p.max_seqlen_q, $p.max_seqlen_k, $p.scale,
                )
            }
            MaskType::Bidirectional => {
                $mod::varlen_bidirectional(
                    $q, $k, $v, $p.cu_seqlens_q, $p.cu_seqlens_k,
                    $p.max_seqlen_q, $p.max_seqlen_k, $p.scale,
                )
            }
            MaskType::SlidingWindow { left, right } => {
                $mod::varlen_windowed(
                    $q, $k, $v, $p.cu_seqlens_q, $p.cu_seqlens_k,
                    $p.max_seqlen_q, $p.max_seqlen_k, $p.scale,
                    Some(*left), Some(*right),
                )
            }
            MaskType::Custom(_) => {
                candle_core::bail!("custom mask not yet supported by {}", stringify!($mod))
            }
        }
    };
}

// ── FA4 ─────────────────────────────────────────────────────────────

#[cfg(feature = "flash-attn-v4")]
struct FlashAttnV4Ops;

#[cfg(feature = "flash-attn-v4")]
impl AttentionOps for FlashAttnV4Ops {
    fn name(&self) -> &str { "flash-attn-v4" }

    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> {
        dispatch_varlen!(crate::attn::flash_v4, q, k, v, params)
    }

    fn paged_attention(&self, q: &Tensor, key_cache: &Tensor, value_cache: &Tensor, params: &PagedParams) -> Result<Tensor> {
        let seqused_k = cu_seqlens_to_lens(params.cu_seqlens_k)?;
        crate::attn::flash_v4::varlen_paged(
            q, key_cache, value_cache, params.block_tables,
            params.cu_seqlens_q, &seqused_k, params.max_seqlen_q, params.max_seqlen_k,
            params.scale,
        )
    }
}

// ── FlashInfer ──────────────────────────────────────────────────────

#[cfg(feature = "flashinfer")]
struct FlashInferOps;

#[cfg(feature = "flashinfer")]
impl AttentionOps for FlashInferOps {
    fn name(&self) -> &str { "flashinfer" }

    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> {
        dispatch_varlen!(crate::attn::flashinfer, q, k, v, params)
    }

    fn paged_attention(&self, q: &Tensor, key_cache: &Tensor, value_cache: &Tensor, params: &PagedParams) -> Result<Tensor> {
        crate::attn::flashinfer::varlen_paged(
            q, key_cache, value_cache, params.block_tables,
            params.cu_seqlens_q, params.cu_seqlens_k, params.max_seqlen_q, params.max_seqlen_k,
            params.scale,
        )
    }
}

// ── FA3 ─────────────────────────────────────────────────────────────

#[cfg(feature = "flash-attn-v3")]
struct FlashAttnV3Ops;

#[cfg(feature = "flash-attn-v3")]
impl AttentionOps for FlashAttnV3Ops {
    fn name(&self) -> &str { "flash-attn-v3" }

    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> {
        dispatch_varlen!(crate::attn::flash_v3, q, k, v, params)
    }

    fn paged_attention(&self, q: &Tensor, key_cache: &Tensor, value_cache: &Tensor, params: &PagedParams) -> Result<Tensor> {
        crate::attn::flash_v3::varlen_paged(
            q, key_cache, value_cache, params.block_tables,
            params.cu_seqlens_q, params.cu_seqlens_k, params.max_seqlen_q, params.max_seqlen_k,
            params.scale,
        )
    }
}

// ── FA2 ─────────────────────────────────────────────────────────────

#[cfg(feature = "flash-attn")]
struct FlashAttnV2Ops;

#[cfg(feature = "flash-attn")]
impl AttentionOps for FlashAttnV2Ops {
    fn name(&self) -> &str { "flash-attn-v2" }

    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> {
        dispatch_varlen!(crate::attn::flash_v2, q, k, v, params)
    }

    fn paged_attention(&self, q: &Tensor, key_cache: &Tensor, value_cache: &Tensor, params: &PagedParams) -> Result<Tensor> {
        // FA2 paged: decode-only (Q=1) via vLLM paged_attention kernel.
        let context_lens = cu_seqlens_to_lens(params.cu_seqlens_k)?;
        crate::attn::paged::decode_attention(
            q, key_cache, value_cache, params.block_tables,
            &context_lens, params.max_seqlen_k, params.scale,
        )
    }

    fn supports_paged_prefill(&self) -> bool { false }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn cu_seqlens_to_lens(cu_seqlens: &Tensor) -> Result<Tensor> {
    let n = cu_seqlens.dim(0)? - 1;
    let hi = cu_seqlens.narrow(0, 1, n)?;
    let lo = cu_seqlens.narrow(0, 0, n)?;
    hi.sub(&lo)
}
