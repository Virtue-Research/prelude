// Shared layer primitives reusable across all model architectures.

pub mod activation;
pub(crate) mod attn_utils;
pub mod embedding;
pub mod linear;

use crate::tensor::Tensor;

/// Layer pattern for hybrid attention models that alternate full softmax
/// attention layers with recurrent/linear-attention layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridAttentionPattern {
    num_hidden_layers: usize,
    full_attention_interval: usize,
}

impl HybridAttentionPattern {
    pub fn new(num_hidden_layers: usize, full_attention_interval: usize) -> Self {
        assert!(
            full_attention_interval > 0,
            "full_attention_interval must be greater than zero"
        );
        Self {
            num_hidden_layers,
            full_attention_interval,
        }
    }

    #[inline]
    pub fn is_full_attention_layer(self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.full_attention_interval == 0
    }

    #[inline]
    pub fn is_recurrent_layer(self, layer_idx: usize) -> bool {
        !self.is_full_attention_layer(layer_idx)
    }

    pub fn full_attention_layers(self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_full_attention_layer(i))
            .count()
    }

    pub fn recurrent_layers(self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.full_attention_layers())
    }
}

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
    /// Per-request hint for hybrid DeltaNet models. `true` means the request's
    /// recurrent and convolution states are known to be zero at this step, so
    /// the model can avoid loading an all-zero row from the state pool before
    /// prefill. Decode and continuation chunks must use `false`.
    pub deltanet_state_is_zero: Option<&'a [bool]>,
    /// Pre-allocated GPU tensor of DeltaNet slot IDs `[bs]` U32.
    /// Used by batched conv1d_update (conv_state_indices) and kda_decode.
    /// Created outside model.forward() to be CUDA-graph compatible.
    pub deltanet_slots_gpu: Option<&'a Tensor>,
}

// Re-exports for convenient use.
pub(crate) use attn_utils::RotaryEmbedding;
pub(crate) use linear::{Linear, RmsNorm};

// ── Utility functions ───────────────────────────────────────────────────

/// Extract the last token per sequence from a packed `[total_tokens, ...]` tensor.
pub(crate) fn last_token_select(
    hidden: &Tensor,
    seq_lens: &[usize],
) -> crate::tensor::Result<Tensor> {
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
pub(crate) fn first_token_select(
    hidden: &Tensor,
    seq_lens: &[usize],
) -> crate::tensor::Result<Tensor> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_attention_pattern_counts_layers() {
        let pattern = HybridAttentionPattern::new(40, 4);

        assert!(pattern.is_recurrent_layer(0));
        assert!(pattern.is_recurrent_layer(2));
        assert!(pattern.is_full_attention_layer(3));
        assert_eq!(pattern.full_attention_layers(), 10);
        assert_eq!(pattern.recurrent_layers(), 30);
    }
}
