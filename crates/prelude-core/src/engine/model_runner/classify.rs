use super::super::*;
use super::prefill::PrefillForwardResult;
use super::prefill_output::materialize_prefill_output_rows;

/// Raw GPU output for classify batches (before CPU post-processing).
///
/// Contains the raw model output tensor and metadata needed for result
/// construction. The GPU worker produces this; CPU post-processing consumes it.
pub(crate) struct RawClassifyOutput {
    pub items: Vec<PreTokenizedClassifyItem>,
    pub forward_result: Option<PrefillForwardResult>,
    pub num_labels: usize,
    /// Pre-fetched label map to avoid model lock during post-processing.
    pub label_map: Vec<Option<String>>,
}

impl Engine {
    pub(crate) fn classify_sync(
        &self,
        request: &ClassifyRequest,
    ) -> Result<ClassifyResult, EngineError> {
        let (token_ids, total_tokens) = tokenize_batch_inputs(&self.tokenizer, &request.inputs)?;
        let item = PreTokenizedClassifyItem {
            request_idx: 0,
            request: request.clone(),
            token_ids,
            total_tokens,
        };
        let mut results = self.classify_batch_pretokenized(vec![item])?;
        results.pop().ok_or_else(|| {
            EngineError::Internal("classify_sync produced no result for single request".into())
        })
    }

    /// Combined GPU forward + CPU post-processing (used by sync/direct path).
    pub fn classify_batch_pretokenized(
        &self,
        items: Vec<PreTokenizedClassifyItem>,
    ) -> Result<Vec<ClassifyResult>, EngineError> {
        self.ensure_task_supported(TaskKind::Classify)?;
        let raw = self.classify_forward_only(items)?;
        classify_postprocess(raw)
    }

    /// GPU-only: runs prefill_pipeline and fetches model metadata.
    /// Does NOT do to_dtype/to_vec2/result construction — those are CPU work.
    pub(crate) fn classify_forward_only(
        &self,
        items: Vec<PreTokenizedClassifyItem>,
    ) -> Result<RawClassifyOutput, EngineError> {
        self.ensure_task_supported(TaskKind::Classify)?;

        let token_groups: Vec<&[Vec<u32>]> =
            items.iter().map(|item| item.token_ids.as_slice()).collect();
        let forward_result = self.prefill_pipeline(&token_groups)?;

        let model = self
            .executor
            .model
            .lock()
            .map_err(|e| EngineError::Internal(format!("model lock poisoned: {e}")))?;
        let classifier = model
            .as_classifier()
            .expect("classify called on model without ClassifierModel");
        let num_labels = classifier.num_labels();
        let label_map = (0..num_labels).map(|i| classifier.get_label(i)).collect();
        drop(model);

        Ok(RawClassifyOutput {
            items,
            forward_result,
            num_labels,
            label_map,
        })
    }
}

/// CPU post-processing for classify: to_dtype → to_vec2 → result construction.
/// Standalone function — no &Engine needed, can run on any thread.
pub(crate) fn classify_postprocess(
    raw: RawClassifyOutput,
) -> Result<Vec<ClassifyResult>, EngineError> {
    let RawClassifyOutput {
        items,
        forward_result,
        num_labels,
        label_map,
    } = raw;

    let Some(mut rows) = materialize_prefill_output_rows(forward_result)? else {
        return Ok(items
            .into_iter()
            .map(|item| ClassifyResult {
                model: item.request.model.clone(),
                results: vec![],
                prompt_tokens: 0,
            })
            .collect());
    };

    let mut results = Vec::with_capacity(items.len());

    for (item_idx, item) in items.into_iter().enumerate() {
        let num_seqs = rows.item_seq_counts[item_idx];
        let mut item_results = Vec::with_capacity(num_seqs);

        for local_idx in 0..num_seqs {
            let probs = rows.next_row()?;
            let (max_idx, _) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            let label = label_map
                .get(max_idx)
                .cloned()
                .flatten()
                .or_else(|| Some(format!("LABEL_{}", max_idx)));

            item_results.push(ClassificationResult {
                index: local_idx as u32,
                label,
                probs,
                num_classes: num_labels as u32,
            });
        }

        results.push(ClassifyResult {
            model: item.request.model.clone(),
            results: item_results,
            prompt_tokens: item.total_tokens,
        });
    }

    Ok(results)
}
