//! Unified ops trait — the single contract between models and device backends.
//!
//! `default_impl()` points to the fallback backend. All methods have defaults:
//! - Primitives delegate to `default_impl()`
//! - Higher-level ops compose from tensor methods (no ops parameter needed)
//! - Fused ops return `None` (no kernel available)
//!
//! Device crates override only what they have optimized kernels for.
//! Everything else auto-inherits.
//!
//! Users call `tensor.exp()`, `tensor.matmul()` etc. — not `ops.exp()`.
//! The trait only exposes methods that models need to call on ops directly.

use crate::tensor::{DType, Tensor, Result};

// ── Op enums ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Exp, Log, Sin, Cos, Abs, Neg, Sqr, Sqrt, Recip, Tanh,
    Relu, Ceil, Floor, Round, Sign,
    Gelu, GeluErf, Silu,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp { Add, Sub, Mul, Div, Min, Max }

#[derive(Debug, Clone, Copy)]
pub enum CompareOp { Eq, Ne, Lt, Gt, Ge, Le }

#[derive(Debug, Clone, Copy)]
pub enum ReduceOp { Sum, Max, Min, ArgMax, ArgMin }

#[derive(Debug, Clone, Copy)]
pub enum QuantScheme {
    Fp8E4m3,
    W4A16 { group_size: usize },
    W4A4 { group_size: usize },
    Int8,
}

// ── Attention params ───────────────────────────────────────────────

pub enum MaskType {
    Causal,
    Bidirectional,
    SlidingWindow { left: usize, right: usize },
    Custom(Tensor),
}

pub struct VarlenParams<'a> {
    pub cu_seqlens_q: &'a Tensor,
    pub cu_seqlens_k: &'a Tensor,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub scale: f32,
    pub mask: MaskType,
    pub softcap: Option<f32>,
}

pub struct PagedParams<'a> {
    pub block_tables: &'a Tensor,
    pub cu_seqlens_q: &'a Tensor,
    pub cu_seqlens_k: &'a Tensor,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub scale: f32,
    pub mask: MaskType,
    pub softcap: Option<f32>,
}

// ── KV cache ───────────────────────────────────────────────────────

pub struct CacheSlotSpec {
    pub slot_size: usize,
    pub dtype: DType,
}

// ── Communication ──────────────────────────────────────────────────

pub enum RemoteTarget {
    Rank(usize),
    Group { name: String, rank: usize },
}

// ── The trait ──────────────────────────────────────────────────────

pub trait Ops: Send + Sync {
    /// Fallback implementation. Terminal backends (CUDA, etc.) return self.
    /// Device crates return the underlying backend.
    fn default_impl(&self) -> &dyn Ops;

    // ════════════════════════════════════════════════════════════════
    // Low-level pointers (needed by CUDA interop)
    // ════════════════════════════════════════════════════════════════

    unsafe fn data_ptr(&self, x: &Tensor) -> Result<*const u8> { unsafe { self.default_impl().data_ptr(x) }}
    unsafe fn data_ptr_mut(&self, x: &Tensor) -> Result<*mut u8> { unsafe { self.default_impl().data_ptr_mut(x) }}

    // ════════════════════════════════════════════════════════════════
    // Normalization — models call ops.rms_norm() directly
    // ════════════════════════════════════════════════════════════════

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> { super::norm::rms_norm(x, weight, eps) }
    fn layer_norm(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> { super::norm::layer_norm(x, weight, bias, eps) }
    fn group_norm(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, num_groups: usize, eps: f32) -> Result<Tensor> { super::norm::group_norm(x, weight, bias, num_groups, eps) }

    // ════════════════════════════════════════════════════════════════
    // Activation — models call ops.silu(), ops.softmax() etc.
    // ════════════════════════════════════════════════════════════════

    fn silu(&self, x: &Tensor) -> Result<Tensor> { candle_nn::ops::silu(x) }
    fn gelu(&self, x: &Tensor) -> Result<Tensor> { x.gelu() }
    fn gelu_approximate(&self, x: &Tensor) -> Result<Tensor> { x.gelu_erf() }
    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> { candle_nn::ops::softmax(x, dim) }
    fn sigmoid(&self, x: &Tensor) -> Result<Tensor> { candle_nn::ops::sigmoid(x) }
    fn log_softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> { candle_nn::ops::log_softmax(x, dim) }

    // ════════════════════════════════════════════════════════════════
    // Convolution
    // ════════════════════════════════════════════════════════════════

    fn conv1d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> Result<Tensor> { super::conv::conv1d(input, weight, bias, stride, padding) }
    fn conv_transpose1d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize, output_padding: usize) -> Result<Tensor> { super::conv::conv_transpose1d(input, weight, bias, stride, padding, output_padding) }
    fn conv2d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: [usize; 2], padding: [usize; 2]) -> Result<Tensor> { super::conv::conv2d(input, weight, bias, stride, padding) }

    // ════════════════════════════════════════════════════════════════
    // GEMM (quantized / grouped)
    // ════════════════════════════════════════════════════════════════

    fn quantized_matmul(&self, _a: &Tensor, _b: &Tensor, _sa: Option<&Tensor>, _sb: Option<&Tensor>, _q: QuantScheme) -> Result<Tensor> { crate::bail!("quantized_matmul: requires device backend") }
    fn grouped_gemm(&self, _input: &Tensor, _weights: &Tensor, _st: &Tensor, _se: &Tensor, _nt: &Tensor) -> Result<Tensor> { crate::bail!("grouped_gemm: requires device backend") }

    // ════════════════════════════════════════════════════════════════
    // Attention
    // ════════════════════════════════════════════════════════════════

    fn attn_name(&self) -> &str { "default" }
    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> { super::attention::varlen_attention(q, k, v, params) }
    fn paged_attention(&self, _q: &Tensor, _kc: &Tensor, _vc: &Tensor, _p: &PagedParams) -> Result<Tensor> { crate::bail!("paged_attention: requires device backend") }

    // ════════════════════════════════════════════════════════════════
    // KV cache
    // ════════════════════════════════════════════════════════════════

    fn cache_slot_spec(&self, head_dim: usize, dtype: DType) -> CacheSlotSpec { CacheSlotSpec { slot_size: head_dim, dtype } }
    fn paged_block_size_hint(&self, _head_dim: usize) -> usize { 16 }
    fn reshape_and_cache(&self, _key: &Tensor, _value: &Tensor, _kc: &Tensor, _vc: &Tensor, _sm: &Tensor) -> Result<()> { crate::bail!("reshape_and_cache: requires device backend") }

    // ════════════════════════════════════════════════════════════════
    // Communication (single-device defaults)
    // ════════════════════════════════════════════════════════════════

    fn world_size(&self) -> usize { 1 }
    fn rank(&self) -> usize { 0 }
    fn all_reduce_sum(&self, x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
    fn all_gather(&self, x: &Tensor, _dim: usize) -> Result<Tensor> { Ok(x.clone()) }
    fn reduce_scatter(&self, x: &Tensor, _dim: usize) -> Result<Tensor> { Ok(x.clone()) }
    fn all_to_all(&self, x: &Tensor, _input_splits: &[usize], _output_splits: &[usize]) -> Result<Tensor> { Ok(x.clone()) }
    fn send(&self, _x: &Tensor, _dst: RemoteTarget) -> Result<()> { crate::bail!("send: not supported") }
    fn recv(&self, _src: RemoteTarget) -> Result<Tensor> { crate::bail!("recv: not supported") }

    /// Fused MoE dispatch: quantize to FP8 + send to expert owners.
    /// None = not supported, fallback to all_to_all.
    fn ep_dispatch_fused(
        &self, _x: &Tensor, _topk_ids: &Tensor, _num_experts: usize, _use_fp8: bool,
    ) -> Option<Result<(Tensor, Tensor)>> { None }

    /// Fused MoE combine: receive + weighted accumulate expert outputs.
    fn ep_combine_fused(
        &self, _x: &Tensor, _topk_weights: &Tensor, _topk_ids: &Tensor,
    ) -> Option<Result<Tensor>> { None }

    // ════════════════════════════════════════════════════════════════
    // Fused ops (optional — return None = no fused kernel)
    // ════════════════════════════════════════════════════════════════

    fn fused_add_rmsnorm(&self, _residual: &Tensor, _x: &Tensor, _weight: &Tensor, _eps: f32) -> Option<Result<(Tensor, Tensor)>> { None }
    fn fused_silu_mul(&self, _gate: &Tensor, _up: &Tensor) -> Option<Result<Tensor>> { None }
    fn fused_gelu_mul(&self, _gate: &Tensor, _up: &Tensor) -> Option<Result<Tensor>> { None }
    /// `silu(gate) * up` on a concatenated `[tokens, 2*dim]` tensor.
    /// Splits the last dimension internally. Returns `[tokens, dim]`.
    /// Avoids the narrow+contiguous copy that separate gate/up tensors need.
    fn silu_mul_concat(&self, _gate_up: &Tensor) -> Option<Result<Tensor>> { None }
    fn fused_qknorm_rope(&self, _q: &Tensor, _k: &Tensor, _qw: &Tensor, _kw: &Tensor, _cos: &Tensor, _sin: &Tensor, _pos: &Tensor, _eps: f32) -> Option<Result<(Tensor, Tensor)>> { None }
    fn fused_knorm_rope_cache_write(&self, _k: &Tensor, _v: &Tensor, _kw: &Tensor, _cos: &Tensor, _sin: &Tensor, _pos: &Tensor, _kc: &Tensor, _vc: &Tensor, _sm: &Tensor, _eps: f32) -> Option<Result<()>> { None }
    fn fused_add(&self, _a: &Tensor, _b: &Tensor) -> Option<Result<Tensor>> { None }
    fn fused_moe_routing(&self, _logits: &Tensor, _top_k: usize) -> Option<Result<(Tensor, Tensor, Tensor, Tensor)>> { None }
    fn fused_moe_gemm(&self, _input: &Tensor, _weights: &Tensor, _tw: &Tensor, _st: &Tensor, _se: &Tensor, _topk: usize, _prefill: bool) -> Option<Result<Tensor>> { None }

    /// Fused Adaptive Layer Norm (AdaLN-Zero) for diffusion transformers.
    /// Computes: normed = layer_norm(x) * (1 + scale) + shift, gated = normed * gate.
    fn fused_adaln_zero(
        &self, _x: &Tensor, _weight: &Tensor, _bias: Option<&Tensor>,
        _scale: &Tensor, _shift: &Tensor, _gate: &Tensor, _eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> { None }

    /// Fused scale + shift (continuous AdaLN variant, no gate).
    /// Computes: layer_norm(x) * (1 + scale) + shift.
    fn fused_scale_shift(
        &self, _x: &Tensor, _weight: &Tensor, _bias: Option<&Tensor>,
        _scale: &Tensor, _shift: &Tensor, _eps: f32,
    ) -> Option<Result<Tensor>> { None }

    /// Fused multi-LoRA matmul: y = base_weight @ x + scale * (lora_b @ lora_a @ x).
    /// adapter_indices: [batch] mapping each token to its adapter (-1 = no LoRA).
    fn fused_lora_matmul(
        &self, _x: &Tensor, _base_weight: &Tensor,
        _lora_a: &Tensor, _lora_b: &Tensor,
        _adapter_indices: &Tensor, _lora_scale: f32,
    ) -> Option<Result<Tensor>> { None }

    // ════════════════════════════════════════════════════════════════
    // Convenience: try fused → fallback to composed
    // ════════════════════════════════════════════════════════════════

    fn add_rmsnorm(&self, residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
        if let Some(r) = self.fused_add_rmsnorm(residual, x, weight, eps) { return r; }
        let sum = (residual + x)?;
        let normed = self.rms_norm(&sum, weight, eps)?;
        Ok((sum, normed))
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        if let Some(r) = self.fused_silu_mul(gate, up) { return r; }
        gate.silu()?.broadcast_mul(up)
    }

    fn gelu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        if let Some(r) = self.fused_gelu_mul(gate, up) { return r; }
        gate.gelu()?.broadcast_mul(up)
    }

    fn add_or_fused(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if let Some(r) = self.fused_add(a, b) { return r; }
        a.broadcast_add(b)
    }

    fn qknorm_rope_and_cache(
        &self, q: &Tensor, k: &Tensor, v: &Tensor,
        q_weight: &Tensor, k_weight: &Tensor,
        cos: &Tensor, sin: &Tensor, position_ids: &Tensor, eps: f32,
        kv_cache: Option<(&Tensor, &Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        if let Some(r) = self.fused_qknorm_rope(q, k, q_weight, k_weight, cos, sin, position_ids, eps) {
            let (q_out, k_out) = r?;
            if let Some((kc, vc, sm)) = kv_cache { self.reshape_and_cache(&k_out, v, kc, vc, sm)?; }
            return Ok((q_out, k_out));
        }
        let q_normed = self.rms_norm(q, q_weight, eps)?;
        let k_normed = self.rms_norm(k, k_weight, eps)?;
        let q_cos = cos.index_select(position_ids, 0)?;
        let q_sin = sin.index_select(position_ids, 0)?;
        let (total, hq, d) = q_normed.dims3()?;
        let hk = k_normed.dim(1)?;
        let q_out = rope_thd(&q_normed.reshape((1, total, hq, d))?, &q_cos, &q_sin)?.reshape((total, hq, d))?;
        let k_out = rope_thd(&k_normed.reshape((1, total, hk, d))?, &q_cos, &q_sin)?.reshape((total, hk, d))?;
        if let Some((kc, vc, sm)) = kv_cache { self.reshape_and_cache(&k_out, v, kc, vc, sm)?; }
        Ok((q_out, k_out))
    }

    // ════════════════════════════════════════════════════════════════
    // Session lifecycle
    // ════════════════════════════════════════════════════════════════

    fn begin_forward(&self) {}
    fn end_forward(&self) {}
    fn precompute_paged_plan(&self, _q_shape: (usize, usize, usize), _kc: &Tensor, _csq: &Tensor, _bt: &Tensor, _csk: &Tensor, _scale: f32) -> Result<()> { Ok(()) }
    fn gpu_free_memory(&self) -> Option<usize> { None }
}

/// Re-export candle-nn rope_thd for use in default implementations.
pub use candle_nn::rotary_emb::rope_thd;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    /// Minimal Ops impl — all defaults inherited.
    struct StubOps;
    impl Ops for StubOps {
        fn default_impl(&self) -> &dyn Ops { self }
    }

    #[test]
    fn fused_stubs_return_none() {
        let ops = StubOps;
        let t = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();

        // Existing fused ops
        assert!(ops.fused_add_rmsnorm(&t, &t, &t, 1e-5).is_none());
        assert!(ops.fused_silu_mul(&t, &t).is_none());
        assert!(ops.fused_gelu_mul(&t, &t).is_none());
        assert!(ops.fused_add(&t, &t).is_none());
        assert!(ops.fused_moe_routing(&t, 2).is_none());

        // New fused ops from design
        assert!(ops.fused_adaln_zero(&t, &t, None, &t, &t, &t, 1e-5).is_none());
        assert!(ops.fused_scale_shift(&t, &t, None, &t, &t, 1e-5).is_none());
        assert!(ops.fused_lora_matmul(&t, &t, &t, &t, &t, 1.0).is_none());

        // EP comm fused ops from design
        assert!(ops.ep_dispatch_fused(&t, &t, 8, false).is_none());
        assert!(ops.ep_combine_fused(&t, &t, &t).is_none());
    }

    #[test]
    fn comm_defaults_single_device() {
        let ops = StubOps;
        assert_eq!(ops.world_size(), 1);
        assert_eq!(ops.rank(), 0);
    }
}
