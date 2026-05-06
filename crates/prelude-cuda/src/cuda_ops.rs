//! CudaOps — single `impl Ops` for CUDA devices.
//!
//! Overrides tensor primitives with CUDA kernels, plus norm, fused, attention, etc.
//! Methods not overridden inherit defaults from `trait Ops`.

use prelude_core::ops::traits::*;
use prelude_core::tensor::{DType, DeviceExt, Result, Tensor};

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

/// Detect the major compute capability of the **current** CUDA device,
/// cached per-device for the process lifetime. Used to gate kernels
/// whose bindings are architecture-specific.
///
/// Per-device cache (not process-wide) because the calling thread's
/// current device is what determines kernel availability. With a
/// process-wide cache, an init call on one GPU could mask a later
/// call from a different-current-device thread.
pub(crate) fn detect_sm_major() -> i32 {
    use std::collections::HashMap;
    use std::sync::Mutex;
    unsafe extern "C" {
        fn cudaGetDevice(dev: *mut i32) -> i32;
        fn cudaDeviceGetAttribute(v: *mut i32, attr: i32, dev: i32) -> i32;
    }
    const CUDA_DEV_ATTR_MAJOR: i32 = 75;

    let mut dev = 0i32;
    if unsafe { cudaGetDevice(&mut dev) } != 0 {
        return 0;
    }

    static CACHE: Mutex<Option<HashMap<i32, i32>>> = Mutex::new(None);
    let mut guard = match CACHE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(&major) = map.get(&dev) {
        return major;
    }

    let mut major = 0i32;
    if unsafe { cudaDeviceGetAttribute(&mut major, CUDA_DEV_ATTR_MAJOR, dev) } != 0 {
        return 0;
    }
    map.insert(dev, major);
    major
}

fn cu_seqlens_to_lens(cu_seqlens: &Tensor) -> Result<Tensor> {
    let n = cu_seqlens.dim(0)? - 1;
    let hi = cu_seqlens.narrow(0, 1, n)?;
    let lo = cu_seqlens.narrow(0, 0, n)?;
    hi.sub(&lo)
}

// ── The single impl ───────────────────────────────────────────────
// Basic tensor ops (matmul, unary, binary, etc.) are handled by candle-core natively.
// CudaOps only overrides fused/inference-specific ops.

impl Ops for CudaOps {
    fn default_impl(&self) -> &dyn Ops {
        self
    }

    // ── Normalization (CUDA kernels) ──────────────────────────────

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        // The fast kernel supports BF16 input with BF16 or F32 weights. The F32
        // weight variant keeps HF parity for models with small `+1` residual
        // norm weights while avoiding the composed multi-kernel path.
        if x.dtype() == DType::BF16 && matches!(weight.dtype(), DType::BF16 | DType::F32) {
            return crate::ops::rmsnorm::fast_rmsnorm(x, weight, eps as f64);
        }
        prelude_core::ops::traits::norm::rms_norm(x, weight, eps)
    }

    // layer_norm: uses trait default (composed via tensor methods)

    // ── GEMM ──────────────────────────────────────────────────────

    fn grouped_gemm(
        &self,
        input: &Tensor,
        weights: &Tensor,
        sorted_token_ids: &Tensor,
        sorted_expert_ids: &Tensor,
        _num_tokens_per_expert: &Tensor,
    ) -> Result<Tensor> {
        let num_assignments = sorted_token_ids.elem_count();
        let num_tokens = input.dims()[0];
        let topk = if num_tokens > 0 {
            num_assignments / num_tokens
        } else {
            1
        };
        // Dispatch order, fastest-first:
        //   1. DeepGEMM `m_grouped_bf16_gemm` (SM90+, BF16, MoE-tuned).
        //      Pads each expert slice to 128 rows; near-zero overhead at
        //      typical prefill batch sizes where M_e ≥ 128.
        //   2. CUTLASS SM100 PtrArray grouped GEMM (Blackwell-only) —
        //      handles odd M_e but with more per-call overhead.
        //   3. WMMA SM75-era kernel (universal fallback).
        // Each helper returns `Ok(None)` on unsupported arch/dtype/shape
        // so we fall through to the next path cleanly.
        if let Some(out) = crate::ops::moe::try_grouped_gemm_deepgemm(
            input,
            weights,
            sorted_token_ids,
            sorted_expert_ids,
            topk,
        )? {
            return Ok(out);
        }
        if let Some(out) = crate::ops::moe::try_grouped_gemm_sm100(
            input,
            weights,
            sorted_token_ids,
            sorted_expert_ids,
            topk,
        )? {
            return Ok(out);
        }
        crate::ops::moe::moe_gemm_wmma(
            input,
            weights,
            &None,
            sorted_token_ids,
            sorted_expert_ids,
            topk,
            num_tokens > 1,
        )
    }

    fn moe_sort_experts(&self, expert_ids: &Tensor) -> Option<Result<(Tensor, Tensor)>> {
        Some(crate::ops::moe::moe_sort_experts_gpu(expert_ids))
    }

    fn rmsnorm_gated(
        &self,
        x: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Option<Result<Tensor>> {
        if x.dtype() == DType::BF16 && weight.dtype() == DType::F32 {
            return Some(crate::ops::rmsnorm::fast_rmsnorm_gated(
                x, gate, weight, eps as f64,
            ));
        }
        None
    }

    fn swap_moe_gate_up(&self, w1: &Tensor, inter: usize) -> Option<Result<()>> {
        Some(crate::ops::moe::swap_gate_up_inplace(w1, inter))
    }

    fn cutlass_fused_moe(
        &self,
        input: &Tensor,
        experts_per_tok: &Tensor,
        topk_weights: &Tensor,
        w1: &Tensor,
        w2: &Tensor,
    ) -> Option<Result<Tensor>> {
        if !input.device().is_cuda()
            || input.dtype() != DType::BF16
            || w1.dtype() != DType::BF16
            || w2.dtype() != DType::BF16
            || topk_weights.dtype() != DType::F32
            || experts_per_tok.dtype() != DType::U32
        {
            return None;
        }
        let device_id = input.device().ordinal() as i32;
        if !crate::ops::moe::cutlass_fused_moe_available(device_id) {
            return None;
        }
        Some(crate::ops::moe::cutlass_fused_moe_forward(
            input,
            experts_per_tok,
            topk_weights,
            w1,
            w2,
        ))
    }

    fn shared_expert_gate(
        &self,
        input: &Tensor,
        shared_out: &Tensor,
        gate_weight: &Tensor,
    ) -> Option<Result<Tensor>> {
        if input.dtype() != DType::BF16
            || shared_out.dtype() != DType::BF16
            || gate_weight.dtype() != DType::BF16
        {
            return None;
        }
        Some(crate::ops::moe::shared_expert_gate(
            input,
            shared_out,
            gate_weight,
        ))
    }

    // ── KV cache ──────────────────────────────────────────────────

    fn reshape_and_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        crate::ops::kv_cache::scatter_kv_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
        )
    }

    // ── Attention ─────────────────────────────────────────────────
    // Runtime dispatch: try FA4 first (has runtime kernel registry with SM detection),
    // fall back to FlashInfer, then composed CPU fallback.

    fn attn_name(&self) -> &str {
        "cuda"
    }

    fn varlen_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        params: &VarlenParams,
    ) -> Result<Tensor> {
        // All three backends are stride-aware on Q/K/V (requires only stride(-1) == 1):
        // FA4 reads strides from DLTensor, FlashInfer reads q.stride(0)/stride(1) in
        // its C++ wrapper, and the composed SDPA fallback materializes via to_dtype.
        if let Some(r) = try_fa4_varlen(q, k, v, params) {
            return r;
        }
        if let Some(r) = try_flashinfer_varlen(q, k, v, params) {
            return r;
        }
        prelude_core::ops::traits::attention::varlen_attention(q, k, v, params)
    }

    fn paged_attention(
        &self,
        q: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        params: &PagedParams,
    ) -> Result<Tensor> {
        // FA4 paged: handles both prefill (Q>1) and decode (Q=1).
        let seqused_k = cu_seqlens_to_lens(params.cu_seqlens_k)?;
        if let Some(r) = crate::attn::flash_v4::try_varlen_paged(
            q,
            key_cache,
            value_cache,
            params.block_tables,
            params.cu_seqlens_q,
            &seqused_k,
            params.max_seqlen_q,
            params.max_seqlen_k,
            params.scale,
        ) {
            return r;
        }
        // FlashInfer fallback (SM80 without FA4, or non-BF16).
        if let Some(r) = try_flashinfer_paged(q, key_cache, value_cache, params) {
            return r;
        }
        prelude_core::bail!("paged_attention: no CUDA backend accepted this shape")
    }

    fn paged_block_size_hint(&self, head_dim: usize) -> usize {
        // Prefer block_size == FA4 tile_n so `paged_non_tma=false` and the
        // compiled TMA paged kernels are eligible. Fall through to the
        // FlashInfer-compatible value for head_dim combinations FA4 doesn't
        // cover.
        match head_dim {
            128 => 128,
            256 => crate::attn::fa4_tile_n(head_dim, head_dim), // 80 for head_dim=256
            _ => 128,
        }
    }

    // ── Fused ops (CUDA kernels) ─────────────────────────────────

    fn fused_add_rmsnorm(
        &self,
        residual: &Tensor,
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> {
        if x.dtype() != DType::BF16 {
            return None;
        }
        // FlashInfer's fused kernel applies weight in BF16; fall through to the
        // composed path when weight is F32 so the multiply happens in F32 (HF
        // parity for models with small `+1` residual norm weights).
        if weight.dtype() != DType::BF16 {
            return None;
        }
        // FlashInfer in-place: after call, x = rmsnorm(x + residual), residual = x + residual.
        // Safe because residual flows linearly through layers (never branched).
        Some(
            crate::attn::flashinfer::fi_fused_add_rmsnorm(x, residual, weight, eps as f64)
                .map(|()| (residual.clone(), x.clone())),
        )
    }

    fn fused_silu_mul(&self, gate: &Tensor, up: &Tensor) -> Option<Result<Tensor>> {
        if gate.dtype() != DType::BF16 {
            return None;
        }
        Some(crate::ops::elementwise::fused_silu_mul(gate, up))
    }

    fn silu_mul_concat(&self, gate_up: &Tensor) -> Option<Result<Tensor>> {
        if gate_up.dtype() != DType::BF16 {
            return None;
        }
        Some(crate::attn::flashinfer::silu_and_mul(gate_up))
    }

    fn fused_add(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> {
        if a.dtype() == DType::BF16 && b.dtype() == DType::BF16 {
            Some(crate::ops::elementwise::vectorized_add(a, b))
        } else {
            None
        }
    }

    fn fused_qknorm_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        position_ids: &Tensor,
        eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> {
        if q.dtype() != DType::BF16 {
            return None;
        }
        // CUDA kernel supports head_dim <= 256 (8 elements per lane, vals[8])
        let head_dim = q.dims().last().copied().unwrap_or(0);
        if head_dim > 256 {
            return None;
        }
        let q_out = match crate::ops::rope::fused_qknorm_rope_varlen(
            q,
            q_weight,
            cos,
            sin,
            position_ids,
            eps as f64,
        ) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        let k_out = match crate::ops::rope::fused_qknorm_rope_varlen(
            k,
            k_weight,
            cos,
            sin,
            position_ids,
            eps as f64,
        ) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        Some(Ok((q_out, k_out)))
    }

    fn fused_q_norm_rope(
        &self,
        q: &Tensor,
        q_weight: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        position_ids: &Tensor,
        eps: f32,
    ) -> Option<Result<Tensor>> {
        if q.dtype() != DType::BF16 {
            return None;
        }
        let head_dim = q.dims().last().copied().unwrap_or(0);
        if head_dim > 256 {
            return None;
        }
        Some(crate::ops::rope::fused_qknorm_rope_varlen(
            q,
            q_weight,
            cos,
            sin,
            position_ids,
            eps as f64,
        ))
    }

    fn fused_knorm_rope_cache_write(
        &self,
        k: &Tensor,
        v: &Tensor,
        k_weight: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        position_ids: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
        eps: f32,
    ) -> Option<Result<()>> {
        if k.dtype() != DType::BF16 {
            return None;
        }
        if !crate::ops::kv_cache::fused_kv_cache_write_enabled() {
            return None;
        }
        let k_dims = k.dims();
        let kc_dims = key_cache.dims();
        let num_kv_heads = if k_dims.len() == 3 {
            k_dims[1]
        } else {
            return None;
        };
        let head_dim = if k_dims.len() == 3 {
            k_dims[2]
        } else {
            return None;
        };
        let block_size = if kc_dims.len() == 4 {
            kc_dims[1]
        } else {
            return None;
        };
        Some(
            crate::ops::kv_cache::fused_knorm_rope_kv_cache_write_varlen(
                k,
                v,
                k_weight,
                cos,
                sin,
                position_ids,
                key_cache,
                value_cache,
                slot_mapping,
                num_kv_heads,
                head_dim,
                block_size,
                eps as f64,
            ),
        )
    }

    fn fused_moe_routing(
        &self,
        gate_logits: &Tensor,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Option<Result<(Tensor, Tensor, Tensor, Tensor)>> {
        if gate_logits.dtype() != DType::BF16 {
            return None;
        }
        Some(crate::ops::moe::fused_moe_routing(
            gate_logits,
            top_k,
            norm_topk_prob,
        ))
    }

    fn fused_moe_gemm(
        &self,
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Tensor,
        sorted_token_ids: &Tensor,
        sorted_expert_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Option<Result<Tensor>> {
        if !matches!(input.dtype(), DType::BF16 | DType::F16) {
            return None;
        }
        Some(crate::ops::moe::moe_gemm_wmma(
            input,
            weights,
            &Some(topk_weights.clone()),
            sorted_token_ids,
            sorted_expert_ids,
            topk,
            is_prefill,
        ))
    }

    fn kda_decode_batched(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        a_raw: &Tensor,
        b_raw: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        pool_state: &Tensor,
        slot_ids: &Tensor,
    ) -> Option<Result<Tensor>> {
        // The wrapper returns Ok(None) when no compiled variant matches the
        // GPU arch or model dims — lift that into `None` at the Ops layer so
        // the caller can fall back cleanly.
        match crate::attn::kda_decode::try_decode(
            q, k, v, a_raw, b_raw, a_log, dt_bias, pool_state, slot_ids,
        ) {
            Ok(Some(out)) => Some(Ok(out)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    fn kda_decode_available_for(
        &self,
        batch: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> bool {
        crate::attn::kda_decode::variant_available(
            batch,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        )
    }

    fn causal_conv1d_fn(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        initial_states: Option<&Tensor>,
        silu_activation: bool,
    ) -> Option<Result<Tensor>> {
        match crate::attn::causal_conv1d::try_fwd(x, weight, bias, initial_states, silu_activation)
        {
            Ok(Some(t)) => Some(Ok(t)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    fn causal_conv1d_fn_channellast(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        initial_states: Option<&Tensor>,
        silu_activation: bool,
    ) -> Option<Result<Tensor>> {
        match crate::attn::causal_conv1d::try_fwd_channellast(
            x,
            weight,
            bias,
            initial_states,
            silu_activation,
        ) {
            Ok(Some(t)) => Some(Ok(t)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    fn causal_conv1d_update(
        &self,
        x: &Tensor,
        conv_state: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        silu_activation: bool,
        conv_state_indices: Option<&Tensor>,
    ) -> Option<Result<Tensor>> {
        match crate::attn::causal_conv1d::try_update(
            x,
            conv_state,
            weight,
            bias,
            silu_activation,
            conv_state_indices,
        ) {
            Ok(Some(t)) => Some(Ok(t)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    fn gdn_post_conv(
        &self,
        mixed_qkv: &Tensor,
        a_raw: &Tensor,
        b_raw: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        num_k_heads: usize,
        num_v_heads: usize,
        head_dim: usize,
    ) -> Option<Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>> {
        Some(crate::ops::gdn_post_conv::gdn_post_conv(
            mixed_qkv,
            a_raw,
            b_raw,
            a_log,
            dt_bias,
            num_k_heads,
            num_v_heads,
            head_dim,
        ))
    }

    fn gather_log_softmax(&self, logits: &Tensor, target_ids: &Tensor) -> Option<Result<Tensor>> {
        Some(crate::ops::gather_log_softmax::gather_log_softmax(
            logits, target_ids,
        ))
    }

    fn gdn_prefill_varlen(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        cu_seqlens: &Tensor,
        initial_state: Option<&Tensor>,
        scale: f32,
    ) -> Option<Result<(Tensor, Tensor)>> {
        match crate::attn::gdn_prefill::try_prefill(
            q,
            k,
            v,
            alpha,
            beta,
            cu_seqlens,
            initial_state,
            scale,
        ) {
            Ok(Some(pair)) => Some(Ok(pair)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    // ── Sampling ──────────────────────────────────────────────────

    fn sample_from_logits(&self, logits: &Tensor, deterministic: bool) -> Option<Result<Tensor>> {
        Some(crate::ops::sampling::sample_from_logits(
            logits,
            deterministic,
        ))
    }

    fn top_k_top_p_sample(
        &self,
        logits: &Tensor,
        top_k: i64,
        top_p: f64,
        deterministic: bool,
    ) -> Option<Result<Tensor>> {
        Some(crate::ops::sampling::top_k_top_p_sample(
            logits,
            top_k,
            top_p,
            deterministic,
        ))
    }

    // ── Session ───────────────────────────────────────────────────

    fn begin_forward(&self) {
        crate::attn::flashinfer::begin_forward();
    }
    fn end_forward(&self) {
        crate::attn::flashinfer::end_forward();
    }
    fn precompute_paged_plan(
        &self,
        q_shape: (usize, usize, usize),
        key_cache: &Tensor,
        cu_seqlens_q: &Tensor,
        block_tables: &Tensor,
        cu_seqlens_k: &Tensor,
        softmax_scale: f32,
    ) -> Result<()> {
        crate::attn::flashinfer::precompute_paged_plan(
            q_shape,
            key_cache,
            cu_seqlens_q,
            block_tables,
            cu_seqlens_k,
            softmax_scale,
        )?;
        Ok(())
    }
    fn gpu_free_memory(&self) -> Option<usize> {
        unsafe extern "C" {
            fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        }
        let (mut free, mut total) = (0usize, 0usize);
        if unsafe { cudaMemGetInfo(&mut free, &mut total) } == 0 {
            Some(free)
        } else {
            None
        }
    }
    fn gpu_total_memory(&self) -> Option<usize> {
        unsafe extern "C" {
            fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        }
        let (mut free, mut total) = (0usize, 0usize);
        if unsafe { cudaMemGetInfo(&mut free, &mut total) } == 0 {
            Some(total)
        } else {
            None
        }
    }
}

// ── Attention dispatch helpers ─────────────────────────────────────

/// Try FA4 for varlen attention. Returns None if FA4 can't handle this request.
fn try_fa4_varlen(q: &Tensor, k: &Tensor, v: &Tensor, p: &VarlenParams) -> Option<Result<Tensor>> {
    use crate::attn::flash_v4;
    match &p.mask {
        MaskType::Causal => flash_v4::try_varlen_causal(
            q,
            k,
            v,
            p.cu_seqlens_q,
            p.cu_seqlens_k,
            p.max_seqlen_q,
            p.max_seqlen_k,
            p.scale,
        ),
        MaskType::Bidirectional => flash_v4::try_varlen_bidirectional(
            q,
            k,
            v,
            p.cu_seqlens_q,
            p.cu_seqlens_k,
            p.max_seqlen_q,
            p.max_seqlen_k,
            p.scale,
        ),
        MaskType::SlidingWindow { left, right } => flash_v4::try_varlen_windowed(
            q,
            k,
            v,
            p.cu_seqlens_q,
            p.cu_seqlens_k,
            p.max_seqlen_q,
            p.max_seqlen_k,
            p.scale,
            Some(*left),
            Some(*right),
        ),
        MaskType::Custom(_) => None, // FA4 doesn't support custom masks
    }
}

/// Try FlashInfer for varlen attention. Returns None if FlashInfer can't handle this.
fn try_flashinfer_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    p: &VarlenParams,
) -> Option<Result<Tensor>> {
    use crate::attn::flashinfer;
    // FlashInfer supports BF16/FP16 on SM80+. The vendored FA2 ragged
    // prefill kernels are not reliable for head_dim > 256 on SM10x
    // (hdim512 can abort inside FlashInfer's scheduler). Let the composed
    // fallback handle those shapes.
    if !matches!(q.dtype(), DType::BF16 | DType::F16) {
        return None;
    }
    let head_dim = q.dims().last().copied().unwrap_or(0);
    if head_dim > 256 {
        return None;
    }
    let result = match &p.mask {
        MaskType::Causal => flashinfer::varlen_causal(
            q,
            k,
            v,
            p.cu_seqlens_q,
            p.cu_seqlens_k,
            p.max_seqlen_q,
            p.max_seqlen_k,
            p.scale,
        ),
        MaskType::Bidirectional => flashinfer::varlen_bidirectional(
            q,
            k,
            v,
            p.cu_seqlens_q,
            p.cu_seqlens_k,
            p.max_seqlen_q,
            p.max_seqlen_k,
            p.scale,
        ),
        MaskType::SlidingWindow { left, right } => {
            // FlashInfer's ragged windowed path is causal-left-window only in
            // this binding. Bidirectional sliding attention (EmbeddingGemma)
            // needs both sides of the window; let the composed SDPA fallback
            // handle it unless FA4 already took the request above.
            if *right != 0 {
                return None;
            }
            flashinfer::varlen_windowed(
                q,
                k,
                v,
                p.cu_seqlens_q,
                p.cu_seqlens_k,
                p.max_seqlen_q,
                p.max_seqlen_k,
                p.scale,
                Some(*left),
                Some(*right),
            )
        }
        MaskType::Custom(_) => return None,
    };
    // If FlashInfer fails due to missing kernel variant, return None to fall through to SDPA
    match &result {
        Err(e)
            if e.to_string().contains("no FA3")
                || e.to_string().contains("no variant")
                || e.to_string().contains("no prefill variant") =>
        {
            tracing::debug!("FlashInfer fallback: {e}");
            None
        }
        _ => Some(result),
    }
}

/// Try FlashInfer for paged attention.
fn try_flashinfer_paged(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    p: &PagedParams,
) -> Option<Result<Tensor>> {
    if !matches!(q.dtype(), DType::BF16 | DType::F16) {
        return None;
    }
    let head_dim = q.dims().last().copied().unwrap_or(0);
    if head_dim > 256 {
        return None;
    }
    Some(crate::attn::flashinfer::varlen_paged(
        q,
        key_cache,
        value_cache,
        p.block_tables,
        p.cu_seqlens_q,
        p.cu_seqlens_k,
        p.max_seqlen_q,
        p.max_seqlen_k,
        p.scale,
    ))
}
