use crate::tensor::{Result, Tensor};

/// Attention mask type.
pub enum MaskType {
    Causal,
    Bidirectional,
    SlidingWindow { left: usize, right: usize },
    /// Custom mask tensor (e.g., speculative decoding tree attention).
    /// Shape: [max_seqlen_q, max_seqlen_k], values 0.0 (attend) or -inf (mask).
    Custom(Tensor),
}

/// Parameters for variable-length contiguous attention.
pub struct VarlenParams<'a> {
    pub cu_seqlens_q: &'a Tensor,
    pub cu_seqlens_k: &'a Tensor,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub scale: f32,
    pub mask: MaskType,
    pub softcap: Option<f32>,
}

/// Parameters for paged KV cache attention.
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

/// Unified attention backend.
///
/// Two methods, not four — mask variant is encoded in `MaskType`.
/// `reshape_and_cache` is on `KvCacheOps`, not here.
pub trait AttentionOps: Send + Sync {
    /// Human-readable backend name (e.g., "flash-attn-v4", "cpu").
    fn name(&self) -> &str;

    /// Variable-length attention over contiguous Q, K, V.
    fn varlen_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        params: &VarlenParams,
    ) -> Result<Tensor>;

    /// Paged attention: Q attends to K/V in block cache.
    ///
    /// Covers decode (max_seqlen_q=1) and chunked prefill (max_seqlen_q>1).
    /// If the underlying kernel doesn't support paged prefill natively,
    /// the device implementation handles the fallback internally.
    fn paged_attention(
        &self,
        q: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        params: &PagedParams,
    ) -> Result<Tensor>;
}
