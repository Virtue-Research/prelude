use candle_core::{Result, Tensor};

/// Attention mask type.
pub enum MaskType {
    Causal,
    Bidirectional,
    SlidingWindow { left: usize, right: usize },
    Custom(Tensor),
}

/// Parameters for variable-length contiguous attention.
pub struct VarlenParams<'a> {
    pub cu_seqlens_q: &'a Tensor,
    pub cu_seqlens_k: &'a Tensor,
    pub max_seqlen_q: u32,
    pub max_seqlen_k: u32,
    pub scale: f32,
    pub mask: MaskType,
    pub softcap: Option<f32>,
}

/// Parameters for paged KV cache attention.
pub struct PagedParams<'a> {
    pub block_tables: &'a Tensor,
    pub num_seqs: u32,
    pub max_blocks_per_seq: u32,
    pub cu_seqlens_q: &'a Tensor,
    pub cu_seqlens_k: &'a Tensor,
    pub max_seqlen_q: u32,
    pub max_seqlen_k: u32,
    pub scale: f32,
    pub mask: MaskType,
    pub softcap: Option<f32>,
}

pub trait AttentionOps: Send + Sync {
    /// Variable-length attention over contiguous Q, K, V.
    fn varlen_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        params: &VarlenParams,
    ) -> Result<Tensor>;

    /// Paged attention: Q attends to K/V in block cache.
    fn paged_attention(
        &self,
        q: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        params: &PagedParams,
    ) -> Result<Tensor>;
}
