use async_trait::async_trait;
use thiserror::Error;

pub(crate) use crate::types::{
    ClassificationInputs, ClassificationResult, ClassifyRequest, ClassifyResult, DecodeMetrics,
    EmbedRequest, EmbedResult, EmbeddingData, FinishReason, GenerateRequest, GenerateResult,
    ModelInfo, PromptInput, StreamEvent, TokenLogprobInfo, Usage,
};

pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::sync::Mutex;
pub(crate) use std::time::Instant;

pub(crate) use candle_core::{DType, Device, Tensor};
pub(crate) use candle_nn::VarBuilder;
pub(crate) use candle_transformers::generation::{LogitsProcessor, Sampling};
pub(crate) use fastokens::Tokenizer;
pub(crate) use tracing::info;

pub(crate) use crate::constants::DEFAULT_SEED;

mod config;
mod device;
mod engine;
pub(crate) mod forward;

mod types;
pub(crate) mod planner;
mod pseudo;
pub mod scheduled;
mod tokenizer;


// ── Re-exports: plan types + engine struct ──
pub use self::engine::Engine;
pub(crate) use self::engine::{ModelExecutor, ModelVariant};
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
pub(crate) use self::types::OwnedBatchDecodeSeq;
#[cfg(feature = "cuda")]
pub(crate) use self::types::PagedKvPool;
pub use self::types::TaskOverride;
pub(crate) use self::types::{
    BatchDecodeSeq, BatchPrefillResult, CacheAllocationPlan, CacheAllocationPlanEntry,
    CommonModelConfig, DecodePlan, EmbeddingActivation, EmbeddingDenseLayerSpec,
    EmbeddingNormalization, EmbeddingPooling, EmbeddingSemantics, ExecutionKind,
    ModelDescriptor, PreTokenizedClassifyItem, PreTokenizedEmbedItem, PrefillPlan,
    PrefixReuseCandidate, PreparedGenerateBatch, GenerateBatchPlan,
    PreparedGenerateRequest, ResolvedPrefixReuse, RuntimeCaps, TaskKind, WeightsBackend,
};

// ── Re-exports: forward (task-specific execution + postprocessing) ──
// Classify/Embed: always available (stubs return errors when flash-attn-v3 absent).
pub(crate) use self::forward::{
    RawClassifyOutput, classify_postprocess,
    RawEmbedOutput, embed_postprocess,
};
#[cfg(feature = "cuda")]
pub(crate) use self::forward::{RawGenerateOutput, generate_postprocess};

// ── Re-exports: helpers (config, device, weights, tokenizer) ──
pub(crate) use self::config::*;
pub(crate) use self::device::*;
pub(crate) use self::tokenizer::tokenize_batch_inputs;
pub(crate) use crate::loading::weights::*;

pub use self::pseudo::PseudoEngine;
pub use self::scheduled::ScheduledEngine;
pub use crate::cache::manager::CacheManager;

#[derive(Debug, Clone, Error)]
pub enum EngineError {
    #[error("model backend unavailable: {0}")]
    Unavailable(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("internal error: {0}")]
    Internal(String),
}

#[async_trait]
pub trait InferenceEngine: Send + Sync + 'static {
    async fn model_info(&self) -> Result<ModelInfo, EngineError>;

    async fn list_models(&self) -> Result<Vec<ModelInfo>, EngineError>;

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResult, EngineError>;

    async fn generate_batch(
        &self,
        requests: Vec<GenerateRequest>,
    ) -> Result<Vec<GenerateResult>, EngineError> {
        let mut out = Vec::with_capacity(requests.len());
        for request in requests {
            out.push(self.generate(request).await?);
        }
        Ok(out)
    }

    /// Stream tokens as they are generated. Default implementation falls back to
    /// `generate()` and sends the entire output as a single token event.
    async fn generate_stream(
        &self,
        request: GenerateRequest,
        tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<(), EngineError> {
        let result = self.generate(request).await?;
        let _ = tx.send(StreamEvent::Started);
        let _ = tx.send(StreamEvent::Token {
            text: result.output_text,
            logprobs: None,
        });
        let _ = tx.send(StreamEvent::Finished {
            finish_reason: result.finish_reason,
            usage: result.usage,
            metrics: result.metrics,
        });
        Ok(())
    }

    async fn cancel(&self, request_id: &str) -> Result<bool, EngineError>;

    /// Classify input text(s) and return class probabilities.
    /// Default implementation returns an error indicating classification is not supported.
    async fn classify(&self, _request: ClassifyRequest) -> Result<ClassifyResult, EngineError> {
        Err(EngineError::Unavailable(
            "classification not supported by this engine".to_string(),
        ))
    }

    /// Generate embeddings for input text(s).
    /// Default implementation returns an error indicating embeddings are not supported.
    async fn embed(&self, _request: EmbedRequest) -> Result<EmbedResult, EngineError> {
        Err(EngineError::Unavailable(
            "embeddings not supported by this engine".to_string(),
        ))
    }
}
