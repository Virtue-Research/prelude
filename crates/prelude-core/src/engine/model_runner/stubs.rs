//! Stub types and methods compiled when flash attention is absent.
//!
//! These exist so that downstream code (batch.rs, gpu_queue.rs, cpu_batch_runtime.rs)
//! can reference the types unconditionally. All methods return errors at runtime.

use super::super::*;

pub(crate) struct RawClassifyOutput {
    pub(crate) _private: (),
}

pub(crate) struct RawEmbedOutput {
    pub(crate) _private: (),
}

pub(crate) fn classify_postprocess(
    _raw: RawClassifyOutput,
) -> Result<Vec<ClassifyResult>, EngineError> {
    Err(EngineError::Internal(
        "classify requires flash attention (flash-attn-v4 or flashinfer feature)".into(),
    ))
}

pub(crate) fn embed_postprocess(
    _raw: RawEmbedOutput,
) -> Result<Vec<EmbedResult>, EngineError> {
    Err(EngineError::Internal(
        "embed requires flash attention (flash-attn-v4 or flashinfer feature)".into(),
    ))
}

impl Engine {
    pub fn classify_batch_pretokenized(
        &self,
        _items: Vec<PreTokenizedClassifyItem>,
    ) -> Result<Vec<ClassifyResult>, EngineError> {
        Err(EngineError::Internal(
            "classify requires flash attention (flash-attn-v4 or flashinfer feature)".into(),
        ))
    }

    pub(crate) fn classify_forward_only(
        &self,
        _items: Vec<PreTokenizedClassifyItem>,
    ) -> Result<RawClassifyOutput, EngineError> {
        Err(EngineError::Internal(
            "classify requires flash attention (flash-attn-v4 or flashinfer feature)".into(),
        ))
    }

    pub(crate) fn classify_sync(
        &self,
        _request: &ClassifyRequest,
    ) -> Result<ClassifyResult, EngineError> {
        Err(EngineError::Internal(
            "classify requires flash attention (flash-attn-v4 or flashinfer feature)".into(),
        ))
    }

    pub fn embed_batch_pretokenized(
        &self,
        _items: Vec<PreTokenizedEmbedItem>,
    ) -> Result<Vec<EmbedResult>, EngineError> {
        Err(EngineError::Internal(
            "embed requires flash attention (flash-attn-v4 or flashinfer feature)".into(),
        ))
    }

    pub(crate) fn embed_forward_only(
        &self,
        _items: Vec<PreTokenizedEmbedItem>,
    ) -> Result<RawEmbedOutput, EngineError> {
        Err(EngineError::Internal(
            "embed requires flash attention (flash-attn-v4 or flashinfer feature)".into(),
        ))
    }

    pub(crate) fn embed_sync(
        &self,
        _request: &EmbedRequest,
    ) -> Result<EmbedResult, EngineError> {
        Err(EngineError::Internal(
            "embed requires flash attention (flash-attn-v4 or flashinfer feature)".into(),
        ))
    }

    pub(crate) fn execute_cuda_prefill_only_batch(
        &self,
        _items: Vec<PreparedGenerateRequest>,
        _prefill_plan: PrefillPlan,
    ) -> Result<Vec<GenerateResult>, EngineError> {
        Err(EngineError::Internal(
            "CUDA prefill requires flash attention (flash-attn-v4 or flashinfer feature)".into(),
        ))
    }
}
