use crate::tensor::{Result, Tensor};

/// All methods default to `None` — devices only override what they support.
pub trait FusedOps: Send + Sync {
    /// Fused residual add + RMSNorm. Returns (sum, normed).
    fn fused_add_rmsnorm(
        &self,
        _residual: &Tensor,
        _x: &Tensor,
        _weight: &Tensor,
        _eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> {
        None
    }

    /// Fused SiLU(gate) * up.
    fn fused_silu_mul(
        &self,
        _gate: &Tensor,
        _up: &Tensor,
    ) -> Option<Result<Tensor>> {
        None
    }

    /// Fused QK-norm + RoPE. Returns (q_normed_roped, k_normed_roped).
    fn fused_qknorm_rope(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _q_weight: &Tensor,
        _k_weight: &Tensor,
        _cos: &Tensor,
        _sin: &Tensor,
        _position_ids: &Tensor,
        _eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> {
        None
    }

    /// Fused K-norm + RoPE + KV cache write.
    fn fused_knorm_rope_cache_write(
        &self,
        _k: &Tensor,
        _v: &Tensor,
        _k_weight: &Tensor,
        _cos: &Tensor,
        _sin: &Tensor,
        _position_ids: &Tensor,
        _key_cache: &Tensor,
        _value_cache: &Tensor,
        _slot_mapping: &Tensor,
        _eps: f32,
    ) -> Option<Result<()>> {
        None
    }

    /// Fused Adaptive Layer Norm (AdaLN-Zero) for diffusion models.
    fn fused_adaln_zero(
        &self,
        _x: &Tensor,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        _scale: &Tensor,
        _shift: &Tensor,
        _gate: &Tensor,
        _eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> {
        None
    }

    /// Fused scale + shift (continuous AdaLN variant).
    fn fused_scale_shift(
        &self,
        _x: &Tensor,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        _scale: &Tensor,
        _shift: &Tensor,
        _eps: f32,
    ) -> Option<Result<Tensor>> {
        None
    }

    /// Fused multi-LoRA matmul.
    fn fused_lora_matmul(
        &self,
        _x: &Tensor,
        _base_weight: &Tensor,
        _lora_a: &Tensor,
        _lora_b: &Tensor,
        _adapter_indices: &Tensor,
        _lora_scale: f32,
    ) -> Option<Result<Tensor>> {
        None
    }

    /// Fused vectorized BF16 add.
    fn fused_add(
        &self,
        _a: &Tensor,
        _b: &Tensor,
    ) -> Option<Result<Tensor>> {
        None
    }

    /// Fused MoE routing: softmax + top-k + weight normalization + sort.
    fn fused_moe_routing(
        &self,
        _gate_logits: &Tensor,
        _top_k: usize,
    ) -> Option<Result<(Tensor, Tensor, Tensor, Tensor)>> {
        None
    }
}
