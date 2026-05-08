use std::path::Path;

use fastokens::Tokenizer;

use crate::engine::EngineError;

pub(super) fn load_tokenizer(model_path: &Path) -> Result<Tokenizer, EngineError> {
    let tokenizer_path = model_path.join("tokenizer.json");
    load_tokenizer_file(&tokenizer_path)
}

pub(super) fn load_tokenizer_file(tokenizer_path: &Path) -> Result<Tokenizer, EngineError> {
    Tokenizer::from_file(tokenizer_path).map_err(|e| {
        EngineError::Internal(format!("failed to load {}: {e}", tokenizer_path.display()))
    })
}

/// Download tokenizer.json from HuggingFace Hub.
pub(super) fn download_tokenizer(model_id: &str) -> Result<Tokenizer, EngineError> {
    tracing::info!(
        repo = model_id,
        "downloading tokenizer from HuggingFace Hub"
    );
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| EngineError::Internal(format!("hf-hub api init: {e}")))?;
    let repo = api.model(model_id.to_string());
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| EngineError::Internal(format!("failed to download tokenizer.json: {e}")))?;
    Tokenizer::from_file(tokenizer_path.as_path())
        .map_err(|e| EngineError::Internal(format!("failed to load tokenizer: {e}")))
}
