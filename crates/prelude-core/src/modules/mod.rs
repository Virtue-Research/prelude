// Shared layer primitives reusable across all model architectures.

pub(crate) mod attn_utils;
pub mod linear;
pub(crate) mod mlp;
mod moe;
pub(crate) mod norm;
pub(crate) mod transformer_block;

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

// ── Attention context structs ───────────────────────────────────────────

/// Layer-level attention context: packs all per-forward-call metadata that
/// threads through Attention → DecoderLayer → Model.
pub struct LayerAttnContext<'a> {
    pub ops: &'a crate::ops::Ops,
    pub cu_seqlens_q: &'a Tensor,
    pub max_seqlen_q: usize,
    pub position_ids: &'a Tensor,
    pub paged_kv: Option<&'a PagedKvContext<'a>>,
}

/// Batch-level attention context: packs all per-forward-call metadata that
/// threads through CausalLM → Model (and from task callers / ModelVariant).
pub struct BatchAttnContext<'a> {
    pub ops: &'a crate::ops::Ops,
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
pub(crate) use mlp::GatedMlp;
pub(crate) use attn_utils::{
    reshape_and_cache, varlen_attention, varlen_attention_bidirectional, varlen_attention_paged,
    varlen_attention_windowed, fused_qkv_projection, qknorm_rope_varlen, RotaryEmbedding,
};
pub(crate) use norm::{
    debug_disable_fused_add_rmsnorm, debug_disable_fused_qknorm_rope, fast_add, fast_rms_norm,
    fast_silu_mul, first_token_select, fused_add_rmsnorm, last_token_select,
};
pub(crate) use transformer_block::TransformerBlock;

// Debug setters re-exported for public API (server CLI, benchmarks, qwen3 re-export).
pub use norm::{
    set_debug_disable_fast_rmsnorm, set_debug_disable_flash_attn_path,
    set_debug_disable_fused_add_rmsnorm, set_debug_disable_fused_qknorm_rope,
    set_debug_disable_fused_silu_mul, set_debug_disable_vectorized_add,
};
