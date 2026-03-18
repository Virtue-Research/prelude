use candle_core::Tensor;

use crate::models::layers::BatchAttnContext;

/// Trait implemented by all model architectures for uniform dispatch.
///
/// Required methods:
/// - `forward()` — the main forward pass
/// - `clear_kv_cache()` — reset KV cache between requests
#[allow(dead_code)]
pub trait ModelForward: Send {
    /// Run the model forward pass.
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> candle_core::Result<Tensor>;

    /// Clear all KV caches.
    fn clear_kv_cache(&mut self);

    /// Forward pass with internal KV cache for CPU decode.
    ///
    /// `input_ids` is `[L]` (flat token IDs). `position_offset` is the starting
    /// position index for the tokens (0 for prefill, prompt_len+step for decode).
    /// Returns logits `[L, vocab_size]`.
    ///
    /// Default: not supported (GPU models use paged KV).
    fn forward_with_cache(
        &mut self,
        _input_ids: &Tensor,
        _position_offset: usize,
    ) -> candle_core::Result<Tensor> {
        candle_core::bail!("forward_with_cache not supported for this model")
    }

    /// Whether this model supports `forward_with_cache` for CPU KV-cached decode.
    fn supports_kv_cache(&self) -> bool {
        false
    }

    // ── Task-specific queries ──────────────────────────────────────────

    /// Whether this is a classifier model.
    fn is_classifier(&self) -> bool {
        false
    }

    /// Whether this is an embedding model.
    fn is_embedding(&self) -> bool {
        false
    }

    /// Embedding hidden size (embedding models only).
    fn embedding_dim(&self) -> Option<usize> {
        None
    }

    /// Returns (num_labels, sample_label) for classifier models.
    fn classifier_info(&self) -> Option<(usize, Option<String>)> {
        None
    }

    /// Get the label string for a class index (classifier models only).
    fn get_label(&self, _class_idx: usize) -> Option<String> {
        None
    }

    /// Get the number of labels (classifier models only).
    fn num_labels(&self) -> Option<usize> {
        None
    }
}
