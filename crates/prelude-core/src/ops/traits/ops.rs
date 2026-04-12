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

    /// Fused Gated DeltaNet (KDA) batched single-token decode.
    ///
    /// Runs the full delta rule step for an entire decode batch in one CUDA
    /// kernel call. Each request's recurrent state lives in the pool at
    /// `pool_state[slot_ids[i]]` and is updated in place. The kernel does
    /// its own L2 norm on q/k and its own softplus-based gate computation,
    /// so inputs are passed raw.
    ///
    /// Shapes (one token per request, `N` requests in the batch):
    /// - `q`, `k`: `[N, H, K]` BF16
    /// - `v`: `[N, HV, V]` BF16
    /// - `a_raw`: `[N, HV]` BF16 (from `in_proj_a`)
    /// - `b_raw`: `[N, HV]` BF16 (from `in_proj_b`)
    /// - `a_log`: `[HV]` F32 (layer param)
    /// - `dt_bias`: `[HV]` (any float dtype) — scalar per v-head
    /// - `pool_state`: `[pool_size, HV, V, K]` F32 (read AND written)
    /// - `slot_ids`: `[N]` U32 on the same device
    ///
    /// Returns `[N, HV, V]` BF16 on success.
    ///
    /// Returns `None` if no backend kernel is available for this GPU
    /// architecture or `(H, HV, K, V)` combination, so the caller can
    /// fall back to a sequential delta-rule loop.
    #[allow(clippy::too_many_arguments)]
    fn kda_decode_batched(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _a_raw: &Tensor,
        _b_raw: &Tensor,
        _a_log: &Tensor,
        _dt_bias: &Tensor,
        _pool_state: &Tensor,
        _slot_ids: &Tensor,
    ) -> Option<Result<Tensor>> { None }

    /// Fused short causal 1D convolution (Dao-AILab causal-conv1d).
    ///
    /// Runs the same kernel Mamba / Mamba-2 / Qwen3-next / Qwen3.5
    /// DeltaNet rely on for the depthwise conv that precedes the
    /// recurrent state update. The whole sequence is processed in one
    /// CUDA launch per call (instead of our `broadcast * weight + sum`
    /// per-token fallback loop).
    ///
    /// Layout is `[batch, dim, seqlen]` (channel-before-time) — note
    /// that this is NOT the `[batch, seqlen, dim]` our Qwen3.5 model
    /// uses elsewhere. The caller transposes.
    ///
    /// Shapes:
    /// - `x`: `[batch, dim, seqlen]` BF16/F16/F32
    /// - `weight`: `[dim, width]` depthwise filter, same dtype class as `x`
    /// - `bias`: optional `[dim]`
    /// - `initial_states`: optional `[batch, dim, width - 1]` — the
    ///   left-context state saved from a previous chunk of the same
    ///   stream. Pass `None` for a fresh start (kernel treats as zeros).
    /// - `silu_activation`: if true, fuse a SiLU at the tail.
    ///
    /// Returns the output `[batch, dim, seqlen]` in the same dtype as
    /// `x`. The caller is responsible for saving the last `width - 1`
    /// positions of the raw input `x` as the new conv_state for the
    /// next chunk.
    ///
    /// Returns `None` when the backend can't serve the call (non-CUDA,
    /// unsupported dtype, or `width > 4` — upstream's specialization
    /// matrix).
    #[allow(clippy::too_many_arguments)]
    fn causal_conv1d_fn(
        &self,
        _x: &Tensor,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        _initial_states: Option<&Tensor>,
        _silu_activation: bool,
    ) -> Option<Result<Tensor>> { None }

    /// Single-token decode step for causal conv1d. Reads the tail of
    /// `conv_state`, computes the output at the new token, and updates
    /// `conv_state` in place by shifting left and appending the new
    /// input.
    ///
    /// Shapes:
    /// - `x`: `[batch, dim]` BF16/F16/F32 (single token)
    /// - `conv_state`: `[batch, dim, width - 1]` **updated in place**
    /// - `weight`: `[dim, width]` depthwise filter
    /// - `bias`: optional `[dim]`
    ///
    /// Returns the output `[batch, dim]`.
    ///
    /// Returns `None` when the backend can't serve the call.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv1d_update(
        &self,
        _x: &Tensor,
        _conv_state: &Tensor,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        _silu_activation: bool,
    ) -> Option<Result<Tensor>> { None }

    /// Fused post-conv1d prep for Qwen3.5 / Qwen3-next DeltaNet.
    ///
    /// Collapses ~20 candle ops per layer (QKV narrow, L2 norm Q/K,
    /// softplus + A_log gate computation, sigmoid beta) into a single
    /// CUDA kernel launch.
    ///
    /// Inputs (all on the same device):
    /// - `mixed_qkv`: `[L, 2*HK*D + HV*D]` BF16, the output of the
    ///   causal_conv1d kernel with channel layout `[Q | K | V]`
    /// - `a_raw`, `b_raw`: `[L, HV]` BF16 — raw scalar gate inputs
    ///   from the in_proj_a / in_proj_b projections
    /// - `a_log`: `[HV]` F32 — the learned log-decay parameter
    /// - `dt_bias`: `[HV]` F32 — the learned delta-t bias
    ///
    /// Returns `(q, k, v, alpha, beta)`:
    /// - `q, k`: `[L, HK, D]` BF16 L2-normalised over the last dim
    /// - `v`: `[L, HV, D]` BF16 (raw slice out of `mixed_qkv`)
    /// - `alpha`: `[L, HV]` F32, `= exp(-exp(A_log) * softplus(a + dt_bias))`
    /// - `beta`: `[L, HV]` F32, `= sigmoid(b_raw)`
    ///
    /// Returns `None` when the backend can't serve the call (non-CUDA
    /// device, unsupported head_dim, or PTX not loaded).
    #[allow(clippy::too_many_arguments)]
    fn gdn_post_conv(
        &self,
        _mixed_qkv: &Tensor,
        _a_raw: &Tensor,
        _b_raw: &Tensor,
        _a_log: &Tensor,
        _dt_bias: &Tensor,
        _num_k_heads: usize,
        _num_v_heads: usize,
        _head_dim: usize,
    ) -> Option<Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>> { None }

    /// Fused Gated DeltaNet (GDN) varlen prefill — matches FLA's
    /// `chunk_gated_delta_rule` semantics (Qwen3-next / Qwen3.5 family).
    ///
    /// Drives FlashInfer's SM90 `gdn_prefill` CUTLASS warp-specialized TMA
    /// kernel. The kernel consumes **scalar-per-head linear-space decay**:
    /// unlike the KDA kernel, there's no chunk cumsum, no `RCP_LN2`
    /// rescaling, and no `safe_gate` clamp — the caller just computes
    /// `alpha = exp(-exp(A_log) * softplus(a + dt_bias))` and hands it in.
    ///
    /// Shapes (`T = total_seqlen` across all requests, packed as batch-1;
    /// `num_sab_heads = max(num_q_heads, num_v_heads)`):
    /// - `q`: `[T, num_q_heads, head_dim]` BF16
    /// - `k`: `[T, num_k_heads, head_dim]` BF16
    /// - `v`: `[T, num_v_heads, head_dim]` BF16
    /// - `alpha`: `[T, num_sab_heads]` F32 (already = exp(per-step g))
    /// - `beta`:  `[T, num_sab_heads]` F32 (already = sigmoid(raw))
    /// - `cu_seqlens`: `[num_seqs+1]` **I64** (note: not I32 like KDA)
    /// - `initial_state`: optional `[num_seqs, num_sab_heads, head_dim, head_dim]` F32
    ///
    /// Supports both:
    /// - **GQA** (`num_q > num_v`, with `num_k == num_v`)
    /// - **GVA** (`num_v > num_q`, with `num_q == num_k`) — Qwen3.5's layout
    ///
    /// Returns `(output, final_state)`:
    /// - `output`: `[T, num_sab_heads, head_dim]` BF16
    /// - `final_state`: `[num_seqs, num_sab_heads, head_dim, head_dim]` F32
    ///
    /// Returns `None` when the backend can't serve this call (non-CUDA,
    /// non-SM90 arch, head_dim != 128, or kernel wasn't AOT-compiled).
    #[allow(clippy::too_many_arguments)]
    fn gdn_prefill_varlen(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _alpha: &Tensor,
        _beta: &Tensor,
        _cu_seqlens: &Tensor,
        _initial_state: Option<&Tensor>,
        _scale: f32,
    ) -> Option<Result<(Tensor, Tensor)>> { None }

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
