//! CudaOps — single `impl Ops` for CUDA devices.
//!
//! Overrides tensor primitives with CUDA kernels, plus norm, fused, attention, etc.
//! Methods not overridden inherit defaults from `trait Ops`.

use prelude_core::tensor::{bail, DType, Device, Shape, Result, Tensor};
use prelude_core::ops::traits::*;

pub struct CudaOps;

/// Static CudaOps instance.
pub fn cuda_ops() -> &'static dyn Ops {
    use std::sync::LazyLock;
    static OPS: LazyLock<CudaOps> = LazyLock::new(|| {
        // Register GPU GEMM dispatch on first access.
        crate::ops::gemm::register_gpu_gemm();
        CudaOps
    });
    &*OPS
}

// ── Helpers ────────────────────────────────────────────────────────

use crate::device::{self as cs, CuResultExt as CsResultExt, DevicePtr};
use crate::tensor_ops_kernels as tk;

fn extract(t: &Tensor) -> Result<(std::sync::RwLockReadGuard<'_, prelude_core::tensor::Storage>, &prelude_core::tensor::Layout)> {
    let guard = t.storage_rw().read().map_err(|_| prelude_core::tensor::Error::Msg("lock poisoned".into()))?;
    let layout = t.our_layout();
    Ok((guard, layout))
}

fn cu_seqlens_to_lens(cu_seqlens: &Tensor) -> Result<Tensor> {
    let v: Vec<u32> = cu_seqlens.to_vec1()?;
    let lens: Vec<u32> = v.windows(2).map(|w| w[1] - w[0]).collect();
    Tensor::from_vec(lens, (v.len() - 1,), cu_seqlens.device())
}

// ── The single impl ───────────────────────────────────────────────

impl Ops for CudaOps {
    fn default_impl(&self) -> &dyn Ops { self }

    // ── Tensor primitives (CUDA kernels) ──────────────────────────

    fn unary(&self, x: &Tensor, op: UnaryOp) -> Result<Tensor> {
        use UnaryOp::*;
        let (guard, layout) = extract(x)?;
        let cuda = cs::as_cuda(&guard, "unary")?;
        let kernel = match op {
            Exp => "uexp", Log => "ulog", Sin => "usin", Cos => "ucos",
            Abs => "uabs", Neg => "uneg", Sqr => "usqr", Sqrt => "usqrt",
            Recip => "urecip", Tanh => "utanh", Relu => "urelu",
            Gelu => "ugelu", GeluErf => "ugelu_erf", Silu => "usilu",
            Ceil => "uceil", Floor => "ufloor", Round => "uround", Sign => "usign",
        };
        let result = tk::launch_unary(cuda, layout, kernel)?;
        let shape = layout.shape().clone();
        drop(guard);
        Ok(cs::tensor_from_device(result, shape))
    }

    fn binary(&self, a: &Tensor, b: &Tensor, op: BinaryOp) -> Result<Tensor> {
        use BinaryOp::*;
        let (ga, la) = extract(a)?;
        let (gb, lb) = extract(b)?;
        let ca = cs::as_cuda(&ga, "binary lhs")?;
        let cb = cs::as_cuda(&gb, "binary rhs")?;
        let kernel = match op { Add => "badd", Sub => "bsub", Mul => "bmul", Div => "bdiv", Min => "bminimum", Max => "bmaximum" };
        let out_shape = la.shape().broadcast_shape_binary_op(lb.shape(), "binary")?;
        let result = tk::launch_binary(ca, &la.broadcast_as(&out_shape)?, cb, &lb.broadcast_as(&out_shape)?, &out_shape, kernel)?;
        drop(ga); drop(gb);
        Ok(cs::tensor_from_device(result, out_shape))
    }

    fn compare(&self, a: &Tensor, b: &Tensor, op: CompareOp) -> Result<Tensor> {
        use CompareOp::*;
        let (ga, la) = extract(a)?;
        let (gb, lb) = extract(b)?;
        let ca = cs::as_cuda(&ga, "compare lhs")?;
        let cb = cs::as_cuda(&gb, "compare rhs")?;
        let kernel = match op { Eq => "eq", Ne => "ne", Lt => "lt", Gt => "gt", Ge => "ge", Le => "le" };
        let out_shape = la.shape().broadcast_shape_binary_op(lb.shape(), "compare")?;
        let result = tk::launch_compare(ca, &la.broadcast_as(&out_shape)?, cb, &lb.broadcast_as(&out_shape)?, &out_shape, kernel)?;
        drop(ga); drop(gb);
        Ok(cs::tensor_from_device(result, out_shape))
    }

    fn reduce(&self, x: &Tensor, dim: usize, keepdim: bool, op: ReduceOp) -> Result<Tensor> {
        use ReduceOp::*;
        let (guard, layout) = extract(x)?;
        let cuda = cs::as_cuda(&guard, "reduce")?;
        let kernel = match op { Sum => "fast_sum", Max => "fast_max", Min => "fast_min", ArgMax => "fast_argmax", ArgMin => "fast_argmin" };
        let (result, out_shape) = tk::launch_reduce(cuda, layout, kernel, dim)?;
        drop(guard);
        if keepdim {
            let mut dims = x.dims().to_vec();
            dims[dim] = 1;
            cs::tensor_from_device(result, out_shape).reshape(dims.as_slice())
        } else {
            Ok(cs::tensor_from_device(result, out_shape))
        }
    }

    fn cast(&self, x: &Tensor, dtype: DType) -> Result<Tensor> {
        if x.dtype() == dtype { return Ok(x.clone()); }
        let (guard, layout) = extract(x)?;
        let cuda = cs::as_cuda(&guard, "cast")?;
        let result = tk::launch_cast(cuda, layout, dtype)?;
        let shape = layout.shape().clone();
        drop(guard);
        Ok(cs::tensor_from_device(result, shape))
    }

    fn contiguous(&self, x: &Tensor) -> Result<Tensor> {
        let (guard, layout) = extract(x)?;
        let cuda = cs::as_cuda(&guard, "contiguous")?;
        let result = tk::launch_contiguous(cuda, layout)?;
        let shape = layout.shape().clone();
        drop(guard);
        Ok(cs::tensor_from_device(result, shape))
    }

    fn affine(&self, x: &Tensor, mul: f64, add: f64) -> Result<Tensor> {
        let (guard, layout) = extract(x)?;
        let cuda = cs::as_cuda(&guard, "affine")?;
        let result = tk::launch_affine(cuda, layout, mul, add)?;
        let shape = layout.shape().clone();
        drop(guard);
        Ok(cs::tensor_from_device(result, shape))
    }

    fn where_cond(&self, cond: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Result<Tensor> {
        // TODO: fix CUDA where_cond kernel for non-contiguous/broadcast layouts
        let c = cond.to_device(&Device::Cpu)?;
        let t = on_true.to_device(&Device::Cpu)?;
        let f = on_false.to_device(&Device::Cpu)?;
        prelude_core::ops::device_backend::device_ops().where_cond(&c, &t, &f)?.to_device(cond.device())
    }

    fn zeros(&self, shape: &Shape, dtype: DType, device: &Device) -> Result<Tensor> {
        let stream = cs::cuda_stream(device.ordinal())?;
        let result = tk::launch_zeros(&stream, dtype, shape.elem_count())?;
        Ok(cs::tensor_from_device(result, shape.clone()))
    }

    fn to_device(&self, x: &Tensor, device: &Device) -> Result<Tensor> {
        if x.device() == device { return Ok(x.clone()); }
        let shape = x.shape().clone();
        let layout = x.our_layout().clone();
        if device.is_cpu() {
            let guard = x.storage_rw().read().map_err(|_| prelude_core::tensor::Error::Msg("lock".into()))?;
            let cuda = cs::as_cuda(&guard, "to_device CPU")?;
            let cpu = cuda.to_cpu(&layout)?;
            drop(guard);
            Ok(Tensor::from_storage_layout(
                std::sync::Arc::new(std::sync::RwLock::new(prelude_core::tensor::Storage::Device(
                    prelude_core::tensor::DeviceStorage::from_cpu(cpu),
                ))),
                prelude_core::tensor::Layout::contiguous(shape), x.dtype(), Device::Cpu,
            ))
        } else {
            let ordinal = device.ordinal();
            let stream = cs::cuda_stream(ordinal)?;
            let guard = x.storage_rw().read().map_err(|_| prelude_core::tensor::Error::Msg("lock".into()))?;
            match &*guard {
                prelude_core::tensor::Storage::Device(dev) if dev.downcast_ref::<prelude_core::tensor::CpuStorage>().is_some() => {
                    let cpu = dev.downcast_ref::<prelude_core::tensor::CpuStorage>().unwrap();
                    let result = cs::CudaStorage::from_cpu(&stream, cpu, &layout)?;
                    drop(guard);
                    Ok(cs::tensor_from_device(result, shape))
                }
                prelude_core::tensor::Storage::Device(_) => {
                    let cuda = cs::as_cuda(&guard, "to_device GPU")?;
                    let cpu = cuda.to_cpu(&layout)?;
                    let cl = prelude_core::tensor::Layout::contiguous(shape.clone());
                    let result = cs::CudaStorage::from_cpu(&stream, &cpu, &cl)?;
                    drop(guard);
                    Ok(cs::tensor_from_device(result, shape))
                }
                _ => bail!("to_device: unsupported storage type"),
            }
        }
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Standard matmul: C[...,M,N] = A[...,M,K] @ B[...,K,N]
        // CUTLASS/DeepGEMM use TN: y[M,N] = x[M,K] @ W[N,K]^T
        // Convert: transpose B's last two dims → [...,N,K], then call TN dispatch.
        let a_dims = a.dims().to_vec();
        let b_dims = b.dims().to_vec();
        let k = *a_dims.last().unwrap();
        let n = *b_dims.last().unwrap();
        let m = if a_dims.len() >= 2 { a_dims[a_dims.len() - 2] } else { 1 };
        let batch: usize = a_dims[..a_dims.len().saturating_sub(2)].iter().product::<usize>().max(1);

        let a_cont = if a.is_contiguous() { a.clone() } else { a.contiguous()? };
        let b_t = b.t()?.contiguous()?; // [..., N, K]

        let y_flat = crate::ops::gemm::gpu_matmul_nt_batched(&a_cont, &b_t, m, n, k, batch)?;

        let mut out_dims = a_dims[..a_dims.len()-1].to_vec();
        out_dims.push(n);
        y_flat.reshape(out_dims.as_slice())
    }

    fn index_select(&self, x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
        let (gx, lx) = extract(x)?;
        let (gi, li) = extract(indices)?;
        let cx = cs::as_cuda(&gx, "index_select x")?;
        let ci = cs::as_cuda(&gi, "index_select ids")?;
        let (result, out_shape) = tk::launch_index_select(cx, lx, ci, li, dim)?;
        drop(gx); drop(gi);
        Ok(cs::tensor_from_device(result, out_shape))
    }

    fn gather(&self, x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
        let (gx, lx) = extract(x)?;
        let (gi, li) = extract(indices)?;
        let cx = cs::as_cuda(&gx, "gather x")?;
        let ci = cs::as_cuda(&gi, "gather ids")?;
        let (result, out_shape) = tk::launch_gather(cx, lx, ci, li, dim)?;
        drop(gx); drop(gi);
        Ok(cs::tensor_from_device(result, out_shape))
    }

    fn scatter_add(&self, x: &Tensor, indices: &Tensor, src: &Tensor, dim: usize) -> Result<Tensor> {
        let (gx, lx) = extract(x)?;
        let (gi, li) = extract(indices)?;
        let (gs, ls) = extract(src)?;
        let cx = cs::as_cuda(&gx, "scatter_add dst")?;
        let ci = cs::as_cuda(&gi, "scatter_add ids")?;
        let cs_s = cs::as_cuda(&gs, "scatter_add src")?;
        let result = tk::launch_scatter_add(cx, lx, ci, li, cs_s, ls, dim)?;
        let shape = lx.shape().clone();
        drop(gx); drop(gi); drop(gs);
        Ok(cs::tensor_from_device(result, shape))
    }

    fn index_add(&self, x: &Tensor, indices: &Tensor, src: &Tensor, dim: usize) -> Result<Tensor> {
        let (gx, lx) = extract(x)?;
        let (gi, li) = extract(indices)?;
        let (gs, ls) = extract(src)?;
        let cx = cs::as_cuda(&gx, "index_add dst")?;
        let ci = cs::as_cuda(&gi, "index_add ids")?;
        let cs_s = cs::as_cuda(&gs, "index_add src")?;
        let result = tk::launch_index_add(cx, lx, ci, li, cs_s, ls, dim)?;
        let shape = lx.shape().clone();
        drop(gx); drop(gi); drop(gs);
        Ok(cs::tensor_from_device(result, shape))
    }

    fn sort_last_dim(&self, x: &Tensor, asc: bool) -> Result<(Tensor, Tensor)> {
        // TODO: fix CUDA sort kernel crash
        let x_cpu = x.to_device(&Device::Cpu)?;
        let (vals, idxs) = prelude_core::ops::device_backend::device_ops().sort_last_dim(&x_cpu, asc)?;
        Ok((vals.to_device(x.device())?, idxs.to_device(x.device())?))
    }

    fn cat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() { bail!("cat: empty tensor list"); }
        let mut out_dims = tensors[0].dims().to_vec();
        for t in &tensors[1..] { out_dims[dim] += t.dims()[dim]; }
        let out_shape = Shape::from(out_dims);
        let guards: Vec<_> = tensors.iter()
            .map(|t| t.storage_rw().read().map_err(|_| prelude_core::tensor::Error::Msg("lock".into())))
            .collect::<Result<Vec<_>>>()?;
        let layouts: Vec<_> = tensors.iter().map(|t| t.our_layout()).collect();
        let cudas: Vec<_> = guards.iter().map(|g| cs::as_cuda(g, "cat")).collect::<Result<Vec<_>>>()?;
        let pairs: Vec<_> = cudas.iter().zip(layouts.iter()).map(|(c, l)| (*c, *l)).collect();
        let result = tk::launch_cat(&pairs, dim, &out_shape)?;
        drop(guards);
        Ok(cs::tensor_from_device(result, out_shape))
    }

    // data_ptr / data_ptr_mut: use trait defaults (not needed for CUDA tensor ops)

    // ── Normalization (CUDA kernels) ──────────────────────────────

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        if x.dtype() == DType::BF16 {
            return crate::ops::rmsnorm::fast_rmsnorm(x, weight, eps as f64);
        }
        // Non-BF16: use composed fallback
        prelude_core::ops::traits::norm::rms_norm(x, weight, eps)
    }

    // layer_norm: uses trait default (composed via tensor methods)

    // ── GEMM ──────────────────────────────────────────────────────

    fn grouped_gemm(&self, input: &Tensor, weights: &Tensor, sorted_token_ids: &Tensor, sorted_expert_ids: &Tensor, _num_tokens_per_expert: &Tensor) -> Result<Tensor> {
        let num_assignments = sorted_token_ids.elem_count();
        let num_tokens = input.dims()[0];
        let topk = if num_tokens > 0 { num_assignments / num_tokens } else { 1 };
        crate::ops::moe::moe_gemm_wmma(input, weights, &None, sorted_token_ids, sorted_expert_ids, topk, num_tokens > 1)
    }

    // ── KV cache ──────────────────────────────────────────────────

    fn reshape_and_cache(&self, key: &Tensor, value: &Tensor, key_cache: &Tensor, value_cache: &Tensor, slot_mapping: &Tensor) -> Result<()> {
        crate::ops::kv_cache::scatter_kv_cache_flash(key, value, key_cache, value_cache, slot_mapping)
    }

    // ── Attention ─────────────────────────────────────────────────
    // Runtime dispatch: try FA4 first (has runtime kernel registry with SM detection),
    // fall back to FlashInfer, then composed CPU fallback.

    fn attn_name(&self) -> &str { "cuda" }

    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> {
        if let Some(r) = try_fa4_varlen(q, k, v, params) { return r; }
        if let Some(r) = try_flashinfer_varlen(q, k, v, params) { return r; }
        // Ultimate fallback: composed matmul SDPA
        prelude_core::ops::traits::attention::varlen_attention(q, k, v, params)
    }

    fn paged_attention(&self, q: &Tensor, key_cache: &Tensor, value_cache: &Tensor, params: &PagedParams) -> Result<Tensor> {
        // Decode (Q=1): FlashInfer's dedicated decode kernel is much faster
        // than FA4's varlen prefill kernel for the single-token case.
        if params.max_seqlen_q == 1 {
            if let Some(r) = try_flashinfer_paged(q, key_cache, value_cache, params) { return r; }
        }
        // Prefill (Q>1): try FA4 first, fall back to FlashInfer prefill.
        let seqused_k = cu_seqlens_to_lens(params.cu_seqlens_k)?;
        if let Some(r) = crate::attn::flash_v4::try_varlen_paged(
            q, key_cache, value_cache, params.block_tables,
            params.cu_seqlens_q, &seqused_k, params.max_seqlen_q, params.max_seqlen_k, params.scale,
        ) { return r; }
        if let Some(r) = try_flashinfer_paged(q, key_cache, value_cache, params) { return r; }
        bail!("paged_attention: no kernel available for this configuration")
    }

    fn paged_block_size_hint(&self, head_dim: usize) -> usize {
        // FlashInfer is the paged fallback, use its alignment requirements
        if head_dim == 256 { 64 } else { 128 }
    }

    // ── Fused ops (CUDA kernels) ─────────────────────────────────

    fn fused_add_rmsnorm(&self, residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32) -> Option<Result<(Tensor, Tensor)>> {
        if x.dtype() != DType::BF16 { return None; }
        Some(crate::ops::rmsnorm::fused_add_rmsnorm(x, residual, weight, eps as f64))
    }

    fn fused_silu_mul(&self, gate: &Tensor, up: &Tensor) -> Option<Result<Tensor>> {
        if gate.dtype() != DType::BF16 { return None; }
        Some(crate::ops::elementwise::fused_silu_mul(gate, up))
    }

    fn fused_add(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> {
        if a.dtype() == DType::BF16 && b.dtype() == DType::BF16 {
            Some(crate::ops::elementwise::vectorized_add(a, b))
        } else { None }
    }

    fn fused_qknorm_rope(&self, q: &Tensor, k: &Tensor, q_weight: &Tensor, k_weight: &Tensor, cos: &Tensor, sin: &Tensor, position_ids: &Tensor, eps: f32) -> Option<Result<(Tensor, Tensor)>> {
        if q.dtype() != DType::BF16 { return None; }
        // CUDA kernel supports head_dim <= 256 (8 elements per lane, vals[8])
        let head_dim = q.dims().last().copied().unwrap_or(0);
        if head_dim > 256 { return None; }
        let q_out = match crate::ops::rope::fused_qknorm_rope_varlen(q, q_weight, cos, sin, position_ids, eps as f64) {
            Ok(t) => t, Err(e) => return Some(Err(e)),
        };
        let k_out = match crate::ops::rope::fused_qknorm_rope_varlen(k, k_weight, cos, sin, position_ids, eps as f64) {
            Ok(t) => t, Err(e) => return Some(Err(e)),
        };
        Some(Ok((q_out, k_out)))
    }

    fn fused_knorm_rope_cache_write(&self, k: &Tensor, v: &Tensor, k_weight: &Tensor, cos: &Tensor, sin: &Tensor, position_ids: &Tensor, key_cache: &Tensor, value_cache: &Tensor, slot_mapping: &Tensor, eps: f32) -> Option<Result<()>> {
        if k.dtype() != DType::BF16 { return None; }
        if !crate::ops::kv_cache::fused_kv_cache_write_enabled() { return None; }
        let k_dims = k.dims();
        let kc_dims = key_cache.dims();
        let num_kv_heads = if k_dims.len() == 3 { k_dims[1] } else { return None };
        let head_dim = if k_dims.len() == 3 { k_dims[2] } else { return None };
        let block_size = if kc_dims.len() == 4 { kc_dims[1] } else { return None };
        Some(crate::ops::kv_cache::fused_knorm_rope_kv_cache_write_varlen(
            k, v, k_weight, cos, sin, position_ids, key_cache, value_cache, slot_mapping,
            num_kv_heads, head_dim, block_size, eps as f64,
        ))
    }

    fn fused_moe_routing(&self, gate_logits: &Tensor, top_k: usize) -> Option<Result<(Tensor, Tensor, Tensor, Tensor)>> {
        if gate_logits.dtype() != DType::BF16 { return None; }
        Some(crate::ops::moe::fused_moe_routing(gate_logits, top_k, true))
    }

    fn fused_moe_gemm(&self, input: &Tensor, weights: &Tensor, topk_weights: &Tensor, sorted_token_ids: &Tensor, sorted_expert_ids: &Tensor, topk: usize, is_prefill: bool) -> Option<Result<Tensor>> {
        if !matches!(input.dtype(), DType::BF16 | DType::F16) { return None; }
        Some(crate::ops::moe::moe_gemm_wmma(input, weights, &Some(topk_weights.clone()), sorted_token_ids, sorted_expert_ids, topk, is_prefill))
    }

    // ── Session ───────────────────────────────────────────────────

    fn begin_forward(&self) {
        crate::attn::flashinfer::begin_forward();
    }
    fn end_forward(&self) {
        crate::attn::flashinfer::end_forward();
    }
    fn precompute_paged_plan(&self, q_shape: (usize, usize, usize), key_cache: &Tensor, cu_seqlens_q: &Tensor, block_tables: &Tensor, cu_seqlens_k: &Tensor, softmax_scale: f32) -> Result<()> {
        crate::attn::flashinfer::precompute_paged_plan(q_shape, key_cache, cu_seqlens_q, block_tables, cu_seqlens_k, softmax_scale)?;
        Ok(())
    }
    fn gpu_free_memory(&self) -> Option<usize> {
        unsafe extern "C" { fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32; }
        let (mut free, mut total) = (0usize, 0usize);
        if unsafe { cudaMemGetInfo(&mut free, &mut total) } == 0 { Some(free) } else { None }
    }
}

// ── Attention dispatch helpers ─────────────────────────────────────

/// Try FA4 for varlen attention. Returns None if FA4 can't handle this request.
fn try_fa4_varlen(q: &Tensor, k: &Tensor, v: &Tensor, p: &VarlenParams) -> Option<Result<Tensor>> {
    use crate::attn::flash_v4;
    match &p.mask {
        MaskType::Causal => flash_v4::try_varlen_causal(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale),
        MaskType::Bidirectional => flash_v4::try_varlen_bidirectional(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale),
        MaskType::SlidingWindow { left, right } => flash_v4::try_varlen_windowed(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale, Some(*left), Some(*right)),
        MaskType::Custom(_) => None, // FA4 doesn't support custom masks
    }
}

/// Try FlashInfer for varlen attention. Returns None if FlashInfer can't handle this.
fn try_flashinfer_varlen(q: &Tensor, k: &Tensor, v: &Tensor, p: &VarlenParams) -> Option<Result<Tensor>> {
    use crate::attn::flashinfer;
    // FlashInfer supports BF16/FP16 on SM80+; head_dim up to 256 (FA2/FA3 limit)
    if !matches!(q.dtype(), DType::BF16 | DType::F16) { return None; }
    let head_dim = q.dims().last().copied().unwrap_or(0);
    if head_dim > 256 { return None; }
    let result = match &p.mask {
        MaskType::Causal => flashinfer::varlen_causal(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale),
        MaskType::Bidirectional => flashinfer::varlen_bidirectional(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale),
        MaskType::SlidingWindow { left, right } => flashinfer::varlen_windowed(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale, Some(*left), Some(*right)),
        MaskType::Custom(_) => return None,
    };
    // If FlashInfer fails due to missing kernel variant, return None to fall through to SDPA
    match &result {
        Err(e) if e.to_string().contains("no FA3") || e.to_string().contains("no variant") => {
            tracing::debug!("FlashInfer fallback: {e}");
            None
        }
        _ => Some(result),
    }
}

/// Try FlashInfer for paged attention.
fn try_flashinfer_paged(q: &Tensor, key_cache: &Tensor, value_cache: &Tensor, p: &PagedParams) -> Option<Result<Tensor>> {
    if !matches!(q.dtype(), DType::BF16 | DType::F16) { return None; }
    Some(crate::attn::flashinfer::varlen_paged(
        q, key_cache, value_cache, p.block_tables,
        p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale,
    ))
}
