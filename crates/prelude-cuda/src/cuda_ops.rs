//! CudaOps — implements all 9 Ops traits for CUDA devices.
//!
//! This is the CUDA device implementation of the Ops bundle.
//! It wraps the GPU kernel functions in `crate::ops::*` and attention backends
//! in `crate::attn::*`, exposing them through the Ops trait interface.

use std::sync::Arc;
use prelude_core::tensor::{bail, DType, Result, Tensor};
use prelude_core::ops::traits::*;

/// Return a static reference to the CUDA Ops bundle (created once on first call).
///
/// Called by the composition root (prelude-server) at startup.
/// Also registers the GPU GEMM dispatch with candle-core.
pub fn cuda_ops() -> &'static Ops {
    use std::sync::LazyLock;
    static OPS: LazyLock<Ops> = LazyLock::new(create_cuda_ops);
    &OPS
}

/// Factory: create the full Ops bundle for CUDA devices.
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

/// Unwrap &Tensor to &candle_core::Tensor for internal CUDA functions.
fn i(t: &Tensor) -> &candle_core::Tensor { t.inner() }

/// Wrap a candle_core::Tensor result into our Tensor.
fn w(r: candle_core::Result<candle_core::Tensor>) -> Result<Tensor> {
    r.map(Tensor::from_candle)
}

/// Wrap a candle_core::Tensor pair result into our Tensor pair.
fn w2(r: candle_core::Result<(candle_core::Tensor, candle_core::Tensor)>) -> Result<(Tensor, Tensor)> {
    r.map(|(a, b)| (Tensor::from_candle(a), Tensor::from_candle(b)))
}

/// Wrap a 4-tuple result.
fn w4(r: candle_core::Result<(candle_core::Tensor, candle_core::Tensor, candle_core::Tensor, candle_core::Tensor)>) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    r.map(|(a, b, c, d)| (Tensor::from_candle(a), Tensor::from_candle(b), Tensor::from_candle(c), Tensor::from_candle(d)))
}

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
        bail!("quantized_matmul not yet implemented in CudaOps")
    }

    fn moe_gemm(
        &self,
        input: &Tensor, weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor, sorted_expert_ids: &Tensor,
        topk: usize, is_prefill: bool,
    ) -> Result<Tensor> {
        let topk_inner = topk_weights.as_ref().map(|t| t.inner().clone());
        w(crate::ops::moe::moe_gemm_wmma(
            i(input), i(weights), &topk_inner,
            i(sorted_token_ids), i(sorted_expert_ids),
            topk, is_prefill,
        ))
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
        crate::ops::kv_cache::scatter_kv_cache_flash(i(key), i(value), i(key_cache), i(value_cache), i(slot_mapping))
    }
}

// ── NormOps ─────────────────────────────────────────────────────────

impl NormOps for CudaOps {
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        w(crate::ops::rmsnorm::fast_rmsnorm(i(x), i(weight), eps as f64))
    }

    fn layer_norm(
        &self,
        x: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean = x_f32.mean_keepdim(prelude_core::tensor::D::Minus1)?;
        let centered = x_f32.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(prelude_core::tensor::D::Minus1)?;
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
        bail!("group_norm not yet implemented on CUDA")
    }
}

// ── ActivationOps ───────────────────────────────────────────────────

impl ActivationOps for CudaOps {
    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        x.silu()
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        x.gelu_erf()
    }

    fn gelu_approximate(&self, x: &Tensor) -> Result<Tensor> {
        x.gelu()
    }

    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        x.softmax(dim)
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
        if stride[0] != stride[1] || padding[0] != padding[1] {
            bail!(
                "conv2d: asymmetric stride/padding not supported (stride={stride:?}, padding={padding:?})"
            );
        }
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
        Some(w2(crate::ops::rmsnorm::fused_add_rmsnorm(i(x), i(residual), i(weight), eps as f64)))
    }

    fn fused_silu_mul(&self, gate: &Tensor, up: &Tensor) -> Option<Result<Tensor>> {
        Some(w(crate::ops::elementwise::fused_silu_mul(i(gate), i(up))))
    }

    fn fused_add(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> {
        Some(w(crate::ops::elementwise::vectorized_add(i(a), i(b))))
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
            i(q), i(q_weight), i(cos), i(sin), i(position_ids), eps as f64,
        ) {
            Ok(t) => Tensor::from_candle(t),
            Err(e) => return Some(Err(e)),
        };
        let k_out = match crate::ops::rope::fused_qknorm_rope_varlen(
            i(k), i(k_weight), i(cos), i(sin), i(position_ids), eps as f64,
        ) {
            Ok(t) => Tensor::from_candle(t),
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
        let k_dims = k.dims();
        let kc_dims = key_cache.dims();
        let num_kv_heads = if k_dims.len() == 3 { k_dims[1] } else { return None };
        let head_dim = if k_dims.len() == 3 { k_dims[2] } else { return None };
        let block_size = if kc_dims.len() == 4 { kc_dims[1] } else { return None };
        Some(crate::ops::kv_cache::fused_knorm_rope_kv_cache_write_varlen(
            i(k), i(v), i(k_weight), i(cos), i(sin), i(position_ids),
            i(key_cache), i(value_cache), i(slot_mapping),
            num_kv_heads, head_dim, block_size,
            eps as f64,
        ))
    }

    fn fused_moe_routing(
        &self,
        gate_logits: &Tensor,
        top_k: usize,
    ) -> Option<Result<(Tensor, Tensor, Tensor, Tensor)>> {
        Some(w4(crate::ops::moe::fused_moe_routing(i(gate_logits), top_k, true)))
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
            i(_block_tables), i(_cu_seqlens_k), _block_size,
        )?;
        Ok(())
    }

    fn gpu_free_memory(&self) -> Option<usize> {
        unsafe extern "C" {
            fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        }
        let mut free = 0usize;
        let mut total = 0usize;
        let ret = unsafe { cudaMemGetInfo(&mut free, &mut total) };
        if ret == 0 { Some(free) } else { None }
    }
}

// ── Attention backend selection ─────────────────────────────────────

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

    #[allow(unreachable_code)]
    panic!("no GPU attention backend compiled — enable flash-attn-v4 or flashinfer")
}

// ── Shared dispatch helper ──────────────────────────────────────────

macro_rules! dispatch_varlen {
    ($mod:path, $q:expr, $k:expr, $v:expr, $p:expr, dispatch) => {
        match &$p.mask {
            MaskType::Causal => {
                w($mod::varlen_causal(
                    i($q), i($k), i($v), i($p.cu_seqlens_q), i($p.cu_seqlens_k),
                    $p.max_seqlen_q, $p.max_seqlen_k, $p.scale,
                ))
            }
            MaskType::Bidirectional => {
                w($mod::varlen_bidirectional(
                    i($q), i($k), i($v), i($p.cu_seqlens_q), i($p.cu_seqlens_k),
                    $p.max_seqlen_q, $p.max_seqlen_k, $p.scale,
                ))
            }
            MaskType::SlidingWindow { left, right } => {
                w($mod::varlen_windowed(
                    i($q), i($k), i($v), i($p.cu_seqlens_q), i($p.cu_seqlens_k),
                    $p.max_seqlen_q, $p.max_seqlen_k, $p.scale,
                    Some(*left), Some(*right),
                ))
            }
            MaskType::Custom(_) => {
                bail!("custom mask not yet supported by {}", stringify!($mod))
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
        dispatch_varlen!(crate::attn::flash_v4, q, k, v, params, dispatch)
    }

    fn paged_attention(&self, q: &Tensor, key_cache: &Tensor, value_cache: &Tensor, params: &PagedParams) -> Result<Tensor> {
        let seqused_k = cu_seqlens_to_lens(params.cu_seqlens_k)?;
        w(crate::attn::flash_v4::varlen_paged(
            i(q), i(key_cache), i(value_cache), i(params.block_tables),
            i(params.cu_seqlens_q), seqused_k.inner(), params.max_seqlen_q, params.max_seqlen_k,
            params.scale,
        ))
    }
}

// ── FlashInfer ──────────────────────────────────────────────────────

#[cfg(feature = "flashinfer")]
struct FlashInferOps;

#[cfg(feature = "flashinfer")]
impl AttentionOps for FlashInferOps {
    fn name(&self) -> &str { "flashinfer" }

    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> {
        dispatch_varlen!(crate::attn::flashinfer, q, k, v, params, dispatch)
    }

    fn paged_attention(&self, q: &Tensor, key_cache: &Tensor, value_cache: &Tensor, params: &PagedParams) -> Result<Tensor> {
        w(crate::attn::flashinfer::varlen_paged(
            i(q), i(key_cache), i(value_cache), i(params.block_tables),
            i(params.cu_seqlens_q), i(params.cu_seqlens_k), params.max_seqlen_q, params.max_seqlen_k,
            params.scale,
        ))
    }
}

/// Convert cu_seqlens → per-sequence lengths (for FA4 paged API).
fn cu_seqlens_to_lens(cu_seqlens: &Tensor) -> Result<Tensor> {
    let v: Vec<u32> = cu_seqlens.to_vec1()?;
    let lens: Vec<u32> = v.windows(2).map(|w| w[1] - w[0]).collect();
    Tensor::from_vec(lens, (v.len() - 1,), cu_seqlens.device())
}
