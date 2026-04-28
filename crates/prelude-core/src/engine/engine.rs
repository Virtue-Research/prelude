use std::sync::Mutex;

use async_trait::async_trait;
use crate::tensor::{DType, Device};
use chrono::Utc;
use fastokens::Tokenizer;

use super::types::{
    CommonModelConfig, EmbeddingSemantics, ModelDescriptor, RuntimeCaps, TaskKind,
};
use super::{
    ClassifyRequest, ClassifyResult, EmbedRequest, EmbedResult, EngineError, GenerateRequest,
    GenerateResult, InferenceEngine, ModelInfo, StreamEvent,
};

/// Type-erased model — all architectures implement `ModelForward`.
/// Adding a new model requires only implementing the trait; no match arms here.
pub(crate) type ModelVariant = Box<dyn crate::models::ModelForward>;

/// Owns the model weights, device, dtype, and model configuration.
/// Extracted from Engine to separate model execution concerns from cache management.
pub struct ModelExecutor {
    pub model: Mutex<ModelVariant>,
    pub device: Device,
    pub(crate) dtype: DType,
    pub config: CommonModelConfig,
    pub(crate) runtime_caps: RuntimeCaps,
    pub ops: &'static dyn crate::ops::Ops,
}

pub struct Engine {
    pub executor: ModelExecutor,
    pub cache: crate::cache::manager::CacheManager,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) model_id: String,
    pub(crate) embedding_semantics: EmbeddingSemantics,
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) descriptor: ModelDescriptor,
    pub engine_config: crate::config::EngineConfig,
}

impl Engine {
    pub(crate) fn ensure_task_supported(&self, requested: TaskKind) -> Result<(), EngineError> {
        let actual = self.descriptor.task;
        if actual == requested {
            return Ok(());
        }

        Err(EngineError::Unavailable(format!(
            "{} not supported: model task is {}, use {}",
            requested.as_str(),
            actual.as_str(),
            actual.endpoint_hint(),
        )))
    }

    // ── Runtime accessors ───────────────────────────────────────────────

    pub fn tokenize(&self, input: &crate::types::PromptInput) -> Result<Vec<u32>, EngineError> {
        super::tokenizer::tokenize_prompt_input(&self.tokenizer, input)
    }

    pub fn tokenize_and_validate(
        &self,
        input: &crate::types::PromptInput,
    ) -> Result<Vec<u32>, EngineError> {
        if matches!(input, crate::types::PromptInput::Text(text) if text.trim().is_empty()) {
            return Err(EngineError::InvalidRequest("prompt is empty".into()));
        }
        let tokens = self.tokenize(input)?;
        if tokens.is_empty() {
            return Err(EngineError::InvalidRequest(
                "prompt tokenized to 0 tokens".into(),
            ));
        }
        Ok(tokens)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn device(&self) -> &crate::tensor::Device {
        &self.executor.device
    }

    pub fn model_descriptor(&self) -> ModelDescriptor {
        self.descriptor
    }

    pub fn runtime_caps(&self) -> RuntimeCaps {
        self.executor.runtime_caps
    }

    /// Returns the maximum context length (max_position_embeddings) for this model.
    pub fn max_context_len(&self) -> usize {
        self.executor.config.max_position_embeddings
    }

    #[inline]
    pub(crate) fn sync_timing_enabled(&self) -> bool {
        self.engine_config.runtime.sync_timing
    }

    #[inline]
    pub(crate) fn maybe_sync_device(&self) {
        if self.sync_timing_enabled() && self.executor.device.is_cuda() {
            // GPU synchronization is handled by the device crate (prelude-cuda)
            // via cudarc; no-op here since we don't own the CUDA context.
        }
    }

    /// Access the engine configuration.
    pub fn engine_config(&self) -> &crate::config::EngineConfig {
        &self.engine_config
    }

    /// Classify model metadata: (num_labels, label_map).
    pub fn classify_metadata(&self) -> Result<(usize, Vec<Option<String>>), EngineError> {
        self.ensure_task_supported(TaskKind::Classify)?;
        let model = self.executor.model.lock()
            .map_err(|e| EngineError::Internal(format!("model lock: {e}")))?;
        let classifier = model.as_classifier()
            .ok_or_else(|| EngineError::Unavailable("model has no ClassifierModel".into()))?;
        let num_labels = classifier.num_labels();
        let label_map = (0..num_labels).map(|i| classifier.get_label(i)).collect();
        Ok((num_labels, label_map))
    }

    /// Embedding model metadata: (dimensions, normalization).
    pub fn embed_metadata(&self) -> Result<(usize, super::types::EmbeddingNormalization), EngineError> {
        self.ensure_task_supported(TaskKind::Embed)?;
        let model = self.executor.model.lock()
            .map_err(|e| EngineError::Internal(format!("model lock: {e}")))?;
        let dimensions = model.as_embedding()
            .map(|e| e.embedding_dim())
            .unwrap_or(0);
        Ok((dimensions, self.embedding_semantics.normalization))
    }

    // ── Executor-facing forward pass ────────────────────────────────────

    /// Run a forward pass for the given batch and return raw model output.
    ///
    /// This is the method that `Executor` implementations call. It dispatches
    /// to the appropriate internal pipeline based on the batch type:
    ///
    /// - **Prefill**: packs tokens → varlen forward → returns last-token logits
    /// - **OneShot**: same forward as Prefill, for classify/embed (no decode loop)
    /// - **Decode**: paged attention decode (not yet connected)
    pub fn forward_batch(
        &self,
        batch: super::executor::ForwardBatch,
    ) -> Result<super::executor::ModelOutput, EngineError> {
        use super::executor::{ForwardBatch, ModelOutput};

        match batch {
            ForwardBatch::Mixed { requests } => {
                self.forward_mixed(requests)
            }
            ForwardBatch::OneShot { token_groups, task } => {
                self.forward_oneshot(token_groups, task)
            }
            ForwardBatch::Decode { tokens, positions, block_tables, deltanet_slots } => {
                self.forward_decode(tokens, positions, block_tables, deltanet_slots)
            }
        }
    }

    /// Unified forward pass: prefill chunks (Q>1) + decode tokens (Q=1) in one batch.
    /// Returns per-request logits and prefill metadata via ModelOutput.
    fn forward_mixed(
        &self,
        requests: Vec<super::executor::StepRequest>,
    ) -> Result<super::executor::ModelOutput, EngineError> {
        use super::executor::ModelOutput;

        if requests.is_empty() {
            return Ok(ModelOutput {
                logits: crate::tensor::Tensor::zeros(
                    (0, 0), DType::F32, &Device::Cpu,
                ).map_err(|e| EngineError::Internal(e.to_string()))?,
                item_seq_counts: vec![],
                prefill_results: vec![],
            });
        }

        let mixed_results = self.batch_mixed_paged(&requests)?;

        Ok(mixed_results)
    }

    /// Forward pass for batched decode (one token per sequence, paged KV).
    fn forward_decode(
        &self,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        block_tables: Vec<Vec<u32>>,
        deltanet_slots: Option<Vec<u32>>,
    ) -> Result<super::executor::ModelOutput, EngineError> {
        use super::executor::ModelOutput;
        use super::BatchDecodeSeq;

        if tokens.is_empty() {
            return Ok(ModelOutput {
                logits: crate::tensor::Tensor::zeros(
                    (0, 0), DType::F32, &Device::Cpu,
                ).map_err(|e| EngineError::Internal(e.to_string()))?,
                item_seq_counts: vec![],
                prefill_results: vec![],
            });
        }

        let seqs: Vec<BatchDecodeSeq> = tokens.iter().enumerate().map(|(i, &token)| {
            BatchDecodeSeq {
                token,
                position: positions[i],
                context_len: positions[i] + 1,
                block_table: &block_tables[i],
                deltanet_slot: deltanet_slots.as_ref().and_then(|s| s.get(i).copied()),
            }
        }).collect();

        let logits = self.batch_decode_paged(&seqs)?;

        Ok(ModelOutput {
            logits,
            item_seq_counts: vec![],
            prefill_results: vec![],
        })
    }


    /// Forward pass for one-shot classify/embed.
    ///
    /// Uses the same varlen prefill pipeline, returns last-token output
    /// per sequence with grouping metadata.
    fn forward_oneshot(
        &self,
        token_groups: Vec<Vec<Vec<u32>>>,
        task: TaskKind,
    ) -> Result<super::executor::ModelOutput, EngineError> {
        use super::executor::ModelOutput;

        self.ensure_task_supported(task)?;

        if token_groups.is_empty() {
            return Ok(ModelOutput {
                logits: crate::tensor::Tensor::zeros(
                    (0, 0), DType::F32, &Device::Cpu,
                ).map_err(|e| EngineError::Internal(e.to_string()))?,
                item_seq_counts: vec![],
                prefill_results: vec![],
            });
        }

        // Convert Vec<Vec<Vec<u32>>> → Vec<&[Vec<u32>]> for prefill_pipeline
        let refs: Vec<&[Vec<u32>]> = token_groups.iter()
            .map(|g| g.as_slice())
            .collect();

        {
            let forward_result = self.prefill_pipeline(&refs)?
                .ok_or_else(|| EngineError::Internal("empty one-shot batch".into()))?;

            Ok(ModelOutput {
                logits: forward_result.output,
                item_seq_counts: forward_result.item_seq_counts,
                prefill_results: vec![],
            })
        }

    }
}

// ── InferenceEngine trait implementation ─────────────────────────────────

#[async_trait]
impl InferenceEngine for Engine {
    async fn model_info(&self) -> Result<ModelInfo, EngineError> {
        Ok(ModelInfo {
            id: self.model_id.clone(),
            created: Utc::now().timestamp(),
            owned_by: "prelude".to_string(),
        })
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, EngineError> {
        Ok(vec![self.model_info().await?])
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResult, EngineError> {
        let req = request;
        tokio::task::block_in_place(|| self.generate_sync(&req))
    }

    async fn generate_batch(
        &self,
        requests: Vec<GenerateRequest>,
    ) -> Result<Vec<GenerateResult>, EngineError> {
        let reqs = requests;
        tokio::task::block_in_place(|| self.generate_batch_sync(&reqs))
    }

    async fn generate_stream(
        &self,
        request: GenerateRequest,
        tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<(), EngineError> {
        tokio::task::block_in_place(|| self.generate_stream_sync(&request, tx))?;
        Ok(())
    }

    async fn cancel(&self, _request_id: &str) -> Result<bool, EngineError> {
        Ok(false)
    }

    async fn classify(&self, request: ClassifyRequest) -> Result<ClassifyResult, EngineError> {
        tokio::task::block_in_place(|| self.classify_sync(&request))
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResult, EngineError> {
        tokio::task::block_in_place(|| self.embed_sync(&request))
    }
}
