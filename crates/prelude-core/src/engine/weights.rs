use std::path::{Path, PathBuf};

use crate::tensor::{DType, Device};
use super::weight_loader::VarBuilder;

use crate::engine::{tensor_err, EngineError};

pub(crate) async fn has_remote_file(repo: &hf_hub::api::tokio::ApiRepo, filename: &str) -> bool {
    repo.get(filename).await.is_ok()
}

/// Download safetensor weight files from an HF Hub repo.
///
/// Sharded models (with `model.safetensors.index.json`) are downloaded
/// concurrently via tokio tasks — all shards download in parallel,
/// bounded by network bandwidth and the hf-hub client's internal
/// semaphore (`with_max_files`). For a 4-shard Qwen3-8B this cuts
/// download time by ~3-4× vs serial. Cache-hit shards return instantly.
///
/// Single-file models (`model.safetensors` without an index) just do
/// one download, no parallelism needed.
pub(crate) async fn load_safetensor_filenames(
    repo: &hf_hub::api::tokio::ApiRepo,
) -> Result<Vec<PathBuf>, EngineError> {
    let json_path = repo.get("model.safetensors.index.json").await;
    match json_path {
        Ok(index_path) => {
            let content = std::fs::read_to_string(&index_path)
                .map_err(|e| EngineError::Internal(format!("failed to read index: {e}")))?;
            let shard_names = parse_weight_map_filenames(&content)?;
            let n = shard_names.len();
            tracing::info!("downloading {n} safetensor shards concurrently");

            let futures: Vec<_> = shard_names
                .iter()
                .map(|filename| async move {
                    let path = repo.get(filename).await.map_err(|e| {
                        EngineError::Internal(format!("failed to download {filename}: {e}"))
                    })?;
                    tracing::info!("  ✓ {filename}");
                    Ok::<PathBuf, EngineError>(path)
                })
                .collect();

            let files = futures_util::future::try_join_all(futures).await?;
            Ok(files)
        }
        Err(_) => {
            let path = repo.get("model.safetensors").await.map_err(|e| {
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
    unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device).map_err(tensor_err) }
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
