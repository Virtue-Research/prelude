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
pub(crate) struct ModelExecutor {
    pub(crate) model: Mutex<ModelVariant>,
    pub(crate) device: Device,
    #[allow(dead_code)]
    pub(crate) dtype: DType,
    pub(crate) config: CommonModelConfig,
    pub(crate) runtime_caps: RuntimeCaps,
    pub(crate) ops: &'static crate::ops::Ops,
}

pub struct Engine {
    pub(crate) executor: ModelExecutor,
    pub(crate) cache: crate::cache::manager::CacheManager,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) model_id: String,
    pub(crate) embedding_semantics: EmbeddingSemantics,
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) descriptor: ModelDescriptor,
    pub(crate) engine_config: crate::config::EngineConfig,
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
            let _ = self.executor.device.synchronize();
        }
    }

    /// Access the engine configuration.
    pub(crate) fn engine_config(&self) -> &crate::config::EngineConfig {
        &self.engine_config
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
