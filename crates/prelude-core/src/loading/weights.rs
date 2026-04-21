use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use crate::loading::var_builder::VarBuilder;

use crate::engine::{candle_err, EngineError};

pub(crate) fn has_remote_file(repo: &hf_hub::api::sync::ApiRepo, filename: &str) -> bool {
    repo.get(filename).is_ok()
}

pub(crate) fn load_safetensor_filenames(
    repo: &hf_hub::api::sync::ApiRepo,
) -> Result<Vec<PathBuf>, EngineError> {
    let json_path = repo.get("model.safetensors.index.json");
    match json_path {
        Ok(index_path) => {
            let content = std::fs::read_to_string(&index_path)
                .map_err(|e| EngineError::Internal(format!("failed to read index: {e}")))?;
            let shard_names = parse_weight_map_filenames(&content)?;
            let mut files = Vec::with_capacity(shard_names.len());
            for filename in &shard_names {
                let path = repo.get(filename).map_err(|e| {
                    EngineError::Internal(format!("failed to download {filename}: {e}"))
                })?;
                files.push(path);
            }
            Ok(files)
        }
        Err(_) => {
            let path = repo.get("model.safetensors").map_err(|e| {
                EngineError::Internal(format!("failed to download model.safetensors: {e}"))
            })?;
            Ok(vec![path])
        }
    }
}

pub(crate) fn find_safetensor_files(model_path: &Path) -> Result<Vec<PathBuf>, EngineError> {
    let index_path = model_path.join("model.safetensors.index.json");
    if index_path.exists() {
        let content = std::fs::read_to_string(&index_path)
            .map_err(|e| EngineError::Internal(format!("failed to read index: {e}")))?;
        let shard_names = parse_weight_map_filenames(&content)?;
        return Ok(shard_names
            .into_iter()
            .map(|name| model_path.join(name))
            .collect());
    }

    let single = model_path.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(model_path)
        .map_err(|e| EngineError::Internal(format!("failed to read dir: {e}")))?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    files.sort();
    Ok(files)
}

pub(crate) fn load_weights(
    model_path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>, EngineError> {
    let filenames = find_safetensor_files(model_path)?;
    load_var_builder_from_filenames(&filenames, dtype, device)
}

pub(crate) fn load_var_builder_from_filenames(
    filenames: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>, EngineError> {
    if filenames.is_empty() {
        return Err(EngineError::Internal(format!(
            "no .safetensors files were resolved for model load"
        )));
    }
    tracing::info!(count = filenames.len(), "loading safetensors shards");
    unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device).map_err(candle_err) }
}

/// Parse `model.safetensors.index.json` content and return unique shard filenames
/// in insertion order. Shared by both HF Hub and local-path loading.
///
/// Rejects absolute paths and path traversal (`..`) to prevent loading files
/// outside the model directory.
fn parse_weight_map_filenames(content: &str) -> Result<Vec<String>, EngineError> {
    let index: serde_json::Value = serde_json::from_str(content)
        .map_err(|e| EngineError::Internal(format!("failed to parse index: {e}")))?;
    let mut filenames = Vec::new();
    let mut seen = std::collections::HashSet::new();
    if let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) {
        for filename in map.values().filter_map(|v| v.as_str()) {
            validate_shard_filename(filename)?;
            if seen.insert(filename.to_string()) {
                filenames.push(filename.to_string());
            }
        }
    }
    Ok(filenames)
}

/// Reject shard filenames that could escape the model directory.
fn validate_shard_filename(filename: &str) -> Result<(), EngineError> {
    let path = std::path::Path::new(filename);
    if path.is_absolute() {
        return Err(EngineError::Internal(format!(
            "shard filename must be relative, got absolute path: {filename}"
        )));
    }
    for component in path.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(EngineError::Internal(format!(
                "shard filename must not contain '..': {filename}"
            )));
        }
    }
    Ok(())
}
