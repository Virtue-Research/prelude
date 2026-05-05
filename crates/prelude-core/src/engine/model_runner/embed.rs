use super::super::*;
use super::prefill::PrefillForwardResult;
use super::prefill_output::materialize_prefill_output_rows;

fn apply_embedding_postprocess(normalization: EmbeddingNormalization, values: &mut [f32]) {
    match normalization {
        EmbeddingNormalization::None => {}
        EmbeddingNormalization::L2 => normalize_embedding_l2(values),
    }
}

fn normalize_embedding_l2(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values {
            *value /= norm;
        }
    }
}

/// Raw GPU output for embed batches (before CPU post-processing).
pub(crate) struct RawEmbedOutput {
    pub items: Vec<PreTokenizedEmbedItem>,
    pub forward_result: Option<PrefillForwardResult>,
    pub dimensions: usize,
    pub normalization: EmbeddingNormalization,
}

impl Engine {
    pub(crate) fn embed_sync(&self, request: &EmbedRequest) -> Result<EmbedResult, EngineError> {
        let (token_ids, total_tokens) = tokenize_batch_inputs(&self.tokenizer, &request.inputs)?;
        let item = PreTokenizedEmbedItem {
            request_idx: 0,
            request: request.clone(),
            token_ids,
            total_tokens,
        };
        let mut results = self.embed_batch_pretokenized(vec![item])?;
        results.pop().ok_or_else(|| {
            EngineError::Internal("embed_sync produced no result for single request".into())
        })
    }

    /// Combined GPU forward + CPU post-processing (used by sync/direct path).
    pub fn embed_batch_pretokenized(
        &self,
        items: Vec<PreTokenizedEmbedItem>,
    ) -> Result<Vec<EmbedResult>, EngineError> {
        self.ensure_task_supported(TaskKind::Embed)?;
        let raw = self.embed_forward_only(items)?;
        embed_postprocess(raw)
    }

    /// GPU-only: runs prefill_pipeline and fetches model metadata.
    /// Does NOT do to_dtype/to_vec2/result construction — those are CPU work.
    pub(crate) fn embed_forward_only(
        &self,
        items: Vec<PreTokenizedEmbedItem>,
    ) -> Result<RawEmbedOutput, EngineError> {
        self.ensure_task_supported(TaskKind::Embed)?;

        let token_groups: Vec<&[Vec<u32>]> =
            items.iter().map(|item| item.token_ids.as_slice()).collect();
        let forward_result = self.prefill_pipeline(&token_groups)?;

        let model = self
            .executor
            .model
            .lock()
            .map_err(|e| EngineError::Internal(format!("model lock poisoned: {e}")))?;
        let dimensions = model.as_embedding().map(|e| e.embedding_dim()).unwrap_or(0);
        drop(model);

        Ok(RawEmbedOutput {
            items,
            forward_result,
            dimensions,
            normalization: self.embedding_semantics.normalization,
        })
    }
}

/// CPU post-processing for embed: to_dtype → to_vec2 → result construction.
/// Standalone function — no &Engine needed, can run on any thread.
pub(crate) fn embed_postprocess(raw: RawEmbedOutput) -> Result<Vec<EmbedResult>, EngineError> {
    let RawEmbedOutput {
        items,
        forward_result,
        dimensions,
        normalization,
    } = raw;

    let Some(mut rows) = materialize_prefill_output_rows(forward_result)? else {
        return Ok(items
            .into_iter()
            .map(|item| EmbedResult {
                model: item.request.model.clone(),
                data: vec![],
                prompt_tokens: 0,
                dimensions,
            })
            .collect());
    };

    let mut results = Vec::with_capacity(items.len());

    for (item_idx, item) in items.into_iter().enumerate() {
        let num_seqs = rows.item_seq_counts[item_idx];
        let mut item_embeddings = Vec::with_capacity(num_seqs);

        for local_idx in 0..num_seqs {
            let mut embedding = rows.next_row()?;
            apply_embedding_postprocess(normalization, &mut embedding);
            item_embeddings.push(EmbeddingData {
                index: local_idx as u32,
                embedding,
            });
        }

        results.push(EmbedResult {
            model: item.request.model.clone(),
            data: item_embeddings,
            prompt_tokens: item.total_tokens,
            dimensions,
        });
    }

    Ok(results)
}
