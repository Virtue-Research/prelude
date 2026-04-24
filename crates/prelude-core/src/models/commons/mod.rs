// Shared layer primitives reusable across all model architectures.

pub mod activation;
pub(crate) mod attn_utils;
pub mod embedding;
pub mod linear;

use crate::tensor::Tensor;

// ── Paged KV structs ────────────────────────────────────────────────────

/// Per-layer paged KV context for varlen attention with paged prefix.
/// When passed to `forward`, enables fused norm+RoPE+KV cache write
/// and paged attention (instead of plain flash_attn_varlen).
pub struct PagedKvContext<'a> {
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub slot_mapping: &'a Tensor,
    pub block_tables: &'a Tensor,
    pub cu_seqlens_k: &'a Tensor,
    pub max_seqlen_k: usize,
}

/// Batch-level paged KV context with per-layer cache slices.
/// Used at Model/CausalLM level; call `.layer(i)` to get per-layer context.
pub struct PagedKvBatchContext<'a> {
    pub key_caches: &'a [Tensor],
    pub value_caches: &'a [Tensor],
    pub slot_mapping: &'a Tensor,
    pub block_tables: &'a Tensor,
    pub cu_seqlens_k: &'a Tensor,
    pub max_seqlen_k: usize,
}

impl<'a> PagedKvBatchContext<'a> {
    /// Create a per-layer view for the given layer index.
    pub fn layer(&self, i: usize) -> PagedKvContext<'a> {
        PagedKvContext {
            key_cache: &self.key_caches[i],
            value_cache: &self.value_caches[i],
            slot_mapping: self.slot_mapping,
            block_tables: self.block_tables,
            cu_seqlens_k: self.cu_seqlens_k,
            max_seqlen_k: self.max_seqlen_k,
        }
    }
}

// ── BatchState (per-batch runtime state for Linear / modules) ──────────

/// Per-batch runtime state passed to `Linear.forward` and module functions.
///
/// Separate from `Ops` (per-device, static) and model weights (per-model, static).
/// Models that don't use LoRA still receive this but never inspect it — they just
/// forward it to `Linear` and module functions.
pub struct BatchState<'a> {
    /// Per-token LoRA adapter index. `[batch_size]` mapping each token to its adapter.
    /// `-1` = no LoRA for this token. `None` = LoRA not active for this batch.
    pub adapter_ids: Option<&'a Tensor>,
}

impl<'a> BatchState<'a> {
    /// Create a BatchState with no LoRA active.
    pub fn no_lora() -> Self {
        Self { adapter_ids: None }
    }
}

// ── Attention context structs ───────────────────────────────────────────

/// Layer-level attention context: packs all per-forward-call metadata that
/// threads through Attention → DecoderLayer → Model.
pub struct LayerAttnContext<'a> {
    pub ops: &'a dyn crate::ops::Ops,
    pub cu_seqlens_q: &'a Tensor,
    pub max_seqlen_q: usize,
    pub position_ids: &'a Tensor,
    pub paged_kv: Option<&'a PagedKvContext<'a>>,
}

/// Batch-level attention context: packs all per-forward-call metadata that
/// threads through CausalLM → Model (and from task callers / ModelVariant).
pub struct BatchAttnContext<'a> {
    pub ops: &'a dyn crate::ops::Ops,
    pub cu_seqlens_q: &'a Tensor,
    pub max_seqlen_q: usize,
    pub position_ids: &'a Tensor,
    pub seq_lens: &'a [usize],
    pub paged_kv: Option<&'a PagedKvBatchContext<'a>>,
    pub deltanet_pool: Option<&'a mut crate::cache::deltanet_pool::DeltaNetPool>,
    pub deltanet_slots: Option<&'a [u32]>,
}

// Re-exports for convenient use.
pub(crate) use linear::{Linear, RmsNorm};
pub(crate) use attn_utils::RotaryEmbedding;

// ── Utility functions ───────────────────────────────────────────────────

/// Extract the last token per sequence from a packed `[total_tokens, ...]` tensor.
pub(crate) fn last_token_select(hidden: &Tensor, seq_lens: &[usize]) -> crate::tensor::Result<Tensor> {
    if seq_lens.iter().all(|&l| l == 1) {
        return Ok(hidden.clone());
    }
    let batch_size = seq_lens.len();
    let mut last_indices = Vec::with_capacity(batch_size);
    let mut off = 0usize;
    for &len in seq_lens {
        last_indices.push((off + len - 1) as u32);
        off += len;
    }
    let indices = Tensor::from_vec(last_indices, (batch_size,), hidden.device())?;
    hidden.index_select(&indices, 0)
}

/// Extract the first token per sequence from a packed `[total_tokens, ...]` tensor.
pub(crate) fn first_token_select(hidden: &Tensor, seq_lens: &[usize]) -> crate::tensor::Result<Tensor> {
    let batch_size = seq_lens.len();
    let mut first_indices = Vec::with_capacity(batch_size);
    let mut off = 0usize;
    for &len in seq_lens {
        first_indices.push(off as u32);
        off += len;
    }
    let indices = Tensor::from_vec(first_indices, (batch_size,), hidden.device())?;
    hidden.index_select(&indices, 0)
}
