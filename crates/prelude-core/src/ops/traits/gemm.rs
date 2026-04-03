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

    /// Grouped GEMM for MoE.
    fn grouped_gemm(
        &self,
        input: &Tensor,
        weights: &Tensor,
        sorted_token_ids: &Tensor,
        sorted_expert_ids: &Tensor,
        num_tokens_per_expert: &Tensor,
    ) -> Result<Tensor>;
}
