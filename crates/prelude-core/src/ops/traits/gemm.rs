use crate::tensor::{Result, Tensor};

/// Quantization scheme for quantized_matmul dispatch.
pub enum QuantScheme {
    Fp8E4m3,
    W4A16 { group_size: usize },
    W4A4 { group_size: usize },
    Int8,
}

pub trait GemmOps: Send + Sync {
    /// Matrix multiply. Dispatch: DeepGEMM > CUTLASS > CK > CPU BLAS.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Quantized matmul with explicit scaling.
    fn quantized_matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        scale_a: Option<&Tensor>,
        scale_b: Option<&Tensor>,
        quant: QuantScheme,
    ) -> Result<Tensor>;

    /// MoE expert-parallel GEMM.
    ///
    /// Each token is dispatched to `topk` experts. The kernel reads tokens in
    /// `sorted_token_ids` order, applies the per-expert weight slice from
    /// `weights[expert_id]`, and optionally multiplies by `topk_weights`.
    ///
    /// - `input`:            [num_tokens, K]
    /// - `weights`:          [num_experts, N, K]
    /// - `topk_weights`:     if Some, [num_tokens, topk] F32
    /// - `sorted_token_ids`: [num_tokens * topk] U32
    /// - `sorted_expert_ids`: [num_tokens * topk] U32
    /// - `topk`:             number of experts per token
    /// - `is_prefill`:       true for prefill (batch), false for decode (M=1)
    fn moe_gemm(
        &self,
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        sorted_expert_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor>;
}
