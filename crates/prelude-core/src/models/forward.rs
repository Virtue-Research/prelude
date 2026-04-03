use crate::tensor::Tensor;

use crate::modules::BatchAttnContext;

// ── Sub-traits ──────────────────────────────────────────────────────────

/// Models that can split forward into hidden-states + lm_head for
/// chunked prompt-logprobs extraction.
pub trait LogitsSplitModel: Send {
    /// Forward pass returning hidden states BEFORE lm_head.
    /// Returns `(total_tokens, hidden_dim)`.
    fn forward_hidden_states(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> crate::tensor::Result<Tensor>;

    /// Apply lm_head (and any post-processing like softcapping) to hidden states.
    /// Can be called on chunks: `compute_logits(chunk)` → `(chunk_size, vocab_size)`.
    fn compute_logits(&self, hidden: &Tensor) -> crate::tensor::Result<Tensor>;
}

/// Models with internal KV cache for CPU sequential decode.
pub trait KvCacheModel: Send {
    /// Forward with internal KV cache.
    /// `input_ids` is `[L]` flat token IDs, `position_offset` is the starting
    /// position (0 for prefill, prompt_len+step for decode).
    /// Returns logits `[L, vocab_size]`.
    fn forward_with_cache(
        &mut self,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> crate::tensor::Result<Tensor>;
}

/// Classifier model metadata.
pub trait ClassifierModel {
    fn num_labels(&self) -> usize;
    fn get_label(&self, class_idx: usize) -> Option<String>;

    /// Convenience: returns (num_labels, sample_label_for_class_0).
    fn classifier_info(&self) -> (usize, Option<String>) {
        (self.num_labels(), self.get_label(0))
    }
}

/// Embedding model metadata.
pub trait EmbeddingModel {
    fn embedding_dim(&self) -> usize;
}

// ── Core trait ──────────────────────────────────────────────────────────

/// Trait implemented by all model architectures for uniform dispatch.
///
/// Required methods:
/// - `forward()` — the main forward pass
/// - `clear_kv_cache()` — reset KV cache between requests
///
/// Optional capabilities are accessed via `as_xxx()` accessors, which return
/// `None` by default. Models that support a capability implement the
/// corresponding sub-trait and override the accessor.
pub trait ModelForward: Send {
    /// Run the model forward pass.
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> crate::tensor::Result<Tensor>;

    /// Clear all KV caches.
    fn clear_kv_cache(&mut self);

    // ── Capability accessors ────────────────────────────────────────

    /// Access hidden-states / logits splitting (for prompt logprobs).
    fn as_logits_model(&self) -> Option<&dyn LogitsSplitModel> { None }
    /// Mutable access for `forward_hidden_states` which takes `&mut self`.
    fn as_logits_model_mut(&mut self) -> Option<&mut dyn LogitsSplitModel> { None }

    /// Access CPU KV-cache decode capability.
    fn as_kv_cache_model(&mut self) -> Option<&mut dyn KvCacheModel> { None }

    /// Access classifier-specific metadata.
    fn as_classifier(&self) -> Option<&dyn ClassifierModel> { None }

    /// Access embedding-specific metadata.
    fn as_embedding(&self) -> Option<&dyn EmbeddingModel> { None }

    /// Direct generation: prefill + decode loop handled internally (e.g. by llama.cpp FFI).
    /// Returns (generated_token_ids, last_logits_f32). Default: not supported.
    fn generate_direct(
        &mut self,
        _prompt_tokens: &[u32],
        _max_new: usize,
    ) -> crate::tensor::Result<Option<(Vec<u32>, Vec<f32>)>> {
        Ok(None)
    }
}
