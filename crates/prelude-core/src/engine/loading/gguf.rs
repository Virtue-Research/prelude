use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use crate::cache::manager::CacheManager;
use crate::config::EngineConfig;
use crate::engine::{
    EmbeddingSemantics, Engine, EngineError, ModelDescriptor, ModelExecutor, RuntimeCaps, TaskKind,
    TaskOverride, WeightsBackend, select_device, tensor_err,
};
use crate::tensor::DType;
use crate::tensor::quantized::gguf_file::Content as GgufContent;

use super::tokenizer::{download_tokenizer, load_tokenizer};

const GGUF_BASE_MODEL_REPO_URL_KEY: &str = "general.base_model.0.repo_url";
const HF_REPO_URL_PREFIX: &str = "https://huggingface.co/";

/// Load a GGUF model from an HF Hub repo that contains .gguf files but no config.json.
/// Auto-selects the best GGUF file (prefers Q8_0 > Q4_K_M > largest).
pub(super) async fn load_hf_hub_gguf(
    repo_id: &str,
    repo: &hf_hub::api::tokio::ApiRepo,
    task_override: TaskOverride,
    engine_config: EngineConfig,
) -> Result<Engine, EngineError> {
    let info = repo
        .info()
        .await
        .map_err(|e| EngineError::Internal(format!("failed to get repo info: {e}")))?;

    let gguf_files: Vec<&str> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|f| f.ends_with(".gguf"))
        .collect();

    if gguf_files.is_empty() {
        return Err(EngineError::InvalidRequest(format!(
            "repo {repo_id} has no config.json or .gguf files"
        )));
    }

    let selected = gguf_files
        .iter()
        .find(|f| f.contains("Q8_0"))
        .or_else(|| gguf_files.iter().find(|f| f.contains("Q4_K_M")))
        .or_else(|| gguf_files.first())
        .unwrap();

    tracing::info!(file = %selected, "auto-selected GGUF file from {repo_id}");

    let gguf_path = repo
        .get(selected)
        .await
        .map_err(|e| EngineError::Internal(format!("failed to download {selected}: {e}")))?;

    let tokenizer_model_id = resolve_gguf_tokenizer_repo(repo_id, &gguf_path);

    load_gguf(&gguf_path, tokenizer_model_id, task_override, engine_config)
}

/// Try to resolve the tokenizer source for a GGUF repo.
/// 1. If this repo has tokenizer.json, use it directly.
/// 2. Otherwise, read GGUF metadata for `general.base_model.0.repo_url` to find the base model.
/// 3. Fall back to stripping `-GGUF` suffix from repo_id.
fn resolve_gguf_tokenizer_repo(repo_id: &str, gguf_path: &Path) -> String {
    if let Some(parent) = gguf_path.parent()
        && parent.join("tokenizer.json").exists()
    {
        return repo_id.to_string();
    }

    if let Ok(mut file) = std::fs::File::open(gguf_path) {
        if let Ok(ct) = GgufContent::read(&mut file)
            && let Some(repo) = gguf_base_model_repo(&ct)
        {
            tracing::info!(base_model = %repo, "resolved tokenizer from GGUF metadata");
            return repo;
        }
    }

    let fallback = stripped_gguf_repo_id(repo_id).unwrap_or_else(|| repo_id.to_string());
    tracing::info!(fallback = %fallback, "using fallback tokenizer repo (stripped -GGUF suffix)");
    fallback
}

/// Load a quantized model from a GGUF file.
pub(super) fn load_gguf(
    gguf_path: &Path,
    model_id: String,
    task_override: TaskOverride,
    engine_config: EngineConfig,
) -> Result<Engine, EngineError> {
    tracing::info!(path = %gguf_path.display(), "loading GGUF model");
    let load_start = Instant::now();

    if !matches!(task_override, TaskOverride::Auto | TaskOverride::Generate) {
        return Err(EngineError::InvalidRequest(
            "GGUF models currently support generation only".into(),
        ));
    }

    let (device, _dtype) = select_device(&engine_config.runtime)?;
    let dtype = DType::F32; // working dtype for embeddings/norms

    let mut file = std::fs::File::open(gguf_path)
        .map_err(|e| EngineError::Internal(format!("failed to open GGUF: {e}")))?;
    let ct = GgufContent::read(&mut file).map_err(tensor_err)?;

    let arch = detect_gguf_arch(&ct);
    tracing::info!(arch = %arch, "detected GGUF architecture");

    let tokenizer = if let Some(parent) = gguf_path.parent()
        && parent.join("tokenizer.json").exists()
    {
        load_tokenizer(parent)?
    } else {
        let tokenizer_repo = gguf_base_model_repo(&ct)
            .or_else(|| stripped_hf_gguf_repo_id(&model_id))
            .or_else(|| guess_tokenizer_repo_from_gguf_metadata(&ct, &arch))
            .unwrap_or_else(|| model_id.clone());
        tracing::info!(repo = %tokenizer_repo, "resolving GGUF tokenizer");
        download_tokenizer(&tokenizer_repo)?
    };

    let arch_spec =
        crate::models::registry::find_arch_spec_by_gguf_arch(&arch).ok_or_else(|| {
            EngineError::InvalidRequest(format!("unsupported GGUF architecture '{arch}'"))
        })?;
    let result = arch_spec.load_gguf(ct, &mut file, &device)?;

    let eos_token_ids = if result.eos_token_ids.is_empty() {
        return Err(EngineError::InvalidRequest(
            "GGUF metadata missing `tokenizer.ggml.eos_token_id`".into(),
        ));
    } else {
        result.eos_token_ids
    };

    let descriptor = ModelDescriptor {
        task: TaskKind::Generate,
        arch_name: arch_spec.name(),
        backend: WeightsBackend::Gguf,
    };
    let runtime_caps = RuntimeCaps {
        supports_kv_cache: true,
        ..RuntimeCaps::default()
    };

    let executor = ModelExecutor {
        model: Mutex::new(result.model),
        ops: crate::ops::select_ops(&device),
        device: device.clone(),
        dtype,
        config: result.common,
        runtime_caps,
    };

    tracing::info!(
        elapsed_ms = load_start.elapsed().as_millis() as u64,
        arch = arch_spec.name(),
        layers = executor.config.num_hidden_layers,
        vocab = executor.config.vocab_size,
        "GGUF model loaded via registry"
    );

    Ok(Engine {
        executor,
        cache: CacheManager::none(),
        tokenizer,
        model_id,
        embedding_semantics: EmbeddingSemantics::default(),
        eos_token_ids,
        descriptor,
        engine_config,
    })
}

/// Detect GGUF architecture from metadata.
fn detect_gguf_arch(ct: &GgufContent) -> String {
    ct.metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().map(|s| s.to_string()))
        .unwrap_or_else(|| "qwen3".to_string())
}

fn gguf_base_model_repo(ct: &GgufContent) -> Option<String> {
    ct.metadata
        .get(GGUF_BASE_MODEL_REPO_URL_KEY)
        .and_then(|v| v.to_string().ok())
        .and_then(|url| hf_repo_from_url(&url))
}

fn hf_repo_from_url(url: &str) -> Option<String> {
    url.strip_prefix(HF_REPO_URL_PREFIX).map(str::to_string)
}

fn stripped_gguf_repo_id(model_id: &str) -> Option<String> {
    model_id
        .strip_suffix("-GGUF")
        .or_else(|| model_id.strip_suffix("-gguf"))
        .map(str::to_string)
}

fn stripped_hf_gguf_repo_id(model_id: &str) -> Option<String> {
    stripped_gguf_repo_id(model_id).filter(|repo| repo.contains('/') && !repo.starts_with('/'))
}

/// Guess the HF repo that ships the tokenizer for this GGUF when metadata
/// does not carry `general.base_model.0.repo_url`.
fn guess_tokenizer_repo_from_gguf_metadata(ct: &GgufContent, arch: &str) -> Option<String> {
    let basename = ct
        .metadata
        .get("general.basename")
        .and_then(|v| v.to_string().ok().map(|s| s.to_string()))?;
    let size_label = ct
        .metadata
        .get("general.size_label")
        .and_then(|v| v.to_string().ok().map(|s| s.to_string()));

    let org = match arch {
        "qwen3" | "qwen2" | "qwen" => "Qwen",
        "llama" => "meta-llama",
        "gemma2" | "gemma3" => "google",
        _ => return None,
    };

    let repo = match size_label {
        Some(size) if !size.is_empty() => format!("{org}/{basename}-{size}"),
        _ => format!("{org}/{basename}"),
    };
    tracing::info!(
        repo = %repo, arch, basename,
        "guessing GGUF tokenizer repo from metadata"
    );
    Some(repo)
}
