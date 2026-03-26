//! Model loading: local path, HF Hub, safetensors, GGUF.
//!
//! All factory functions return a fully assembled `Engine`.

pub mod var_builder;
pub(crate) mod weights;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use crate::loading::var_builder::VarBuilder;
use fastokens::Tokenizer;

use crate::cache::manager::CacheManager;
use crate::config::EngineConfig;
use crate::engine::{
    candle_err, init_cpu_runtime_if_needed, load_model_config, load_safetensor_filenames,
    load_var_builder_from_filenames, load_weights, parse_model_config_for_source,
    has_remote_file, select_device,
    CommonModelConfig, EmbeddingActivation, EmbeddingDenseLayerSpec, EmbeddingNormalization,
    EmbeddingPooling, EmbeddingSemantics, Engine, EngineError, ModelDescriptor, ModelExecutor,
    ModelVariant, ResolvedModelConfig, RuntimeCaps, TaskKind, TaskOverride, WeightsBackend,
};
use crate::models::gemma3::meta::{
    Gemma3ModelBuildContext, build_gemma3_model_with_context,
};
use crate::models::registry::AuxiliaryVarBuilder;
use crate::models::qwen3_5::gguf::Qwen3_5GgufModel;
use crate::models::qwen3::gguf::Qwen3GgufModel;

#[derive(Debug, serde::Deserialize)]
struct SentenceTransformerModuleEntry {
    idx: usize,
    path: String,
    #[serde(rename = "type")]
    module_type: String,
}

#[derive(Debug, serde::Deserialize)]
struct SentenceTransformerPoolingConfig {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
}

#[derive(Debug, serde::Deserialize)]
struct SentenceTransformerDenseConfig {
    in_features: usize,
    out_features: usize,
    #[serde(default)]
    bias: Option<bool>,
    #[serde(default)]
    activation_function: Option<String>,
    #[serde(default)]
    activation: Option<String>,
}

struct LoadedEmbeddingModules {
    spec: EmbeddingSemantics,
    auxiliary: Vec<AuxiliaryVarBuilder>,
}

impl Engine {
    // Cleaned -- Reviewed by Minzhou
    pub fn from_local_path_with_task(
        model_path: impl AsRef<Path>,
        model_id: impl Into<String>,
        task_override: TaskOverride,
        engine_config: EngineConfig,
    ) -> Result<Self, EngineError> {
        let model_path = model_path.as_ref();
        let model_id = model_id.into();

        // Detect GGUF file by extension
        if model_path.extension().is_some_and(|ext| ext == "gguf") {
            return load_gguf(model_path, model_id, task_override, engine_config);
        }

        tracing::info!(path = %model_path.display(), "loading model");
        let load_start = Instant::now();

        let (device, dtype) = select_device(&engine_config.runtime)?;
        init_cpu_runtime_if_needed(&device, &engine_config.runtime);

        let resolved = load_model_config(model_path, task_override)?;
        let embedding_modules =
            load_embedding_modules_from_dir(
                model_path,
                resolved.spec.name(),
                resolved.task,
                DType::F32,
                &device,
            )?;
        let vb = load_weights(model_path, dtype, &device)?;
        let tokenizer = load_tokenizer(model_path)?;
        load_safetensor_parts(
            model_id,
            resolved,
            embedding_modules,
            vb,
            tokenizer,
            device,
            dtype,
            load_start,
            engine_config,
        )
    }

    // Cleaned -- Reviewed by Minzhou
    pub fn from_hf_hub_with_task(
        repo_id: &str,
        task_override: TaskOverride,
        engine_config: EngineConfig,
    ) -> Result<Self, EngineError> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| EngineError::Internal(format!("failed to init hf-hub api: {e}")))?;
        let repo = api.model(repo_id.to_string());

        tracing::info!(repo = repo_id, "downloading model from HuggingFace Hub");

        // Try config.json — if missing, this might be a GGUF-only repo
        let config_path = match repo.get("config.json") {
            Ok(path) => path,
            Err(_) => {
                tracing::info!("config.json not found, checking for GGUF files");
                return Self::from_hf_hub_gguf(repo_id, &repo, task_override, engine_config);
            }
        };
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
            EngineError::Internal(format!("failed to download tokenizer.json: {e}"))
        })?;

        let weight_files = load_safetensor_filenames(&repo)?;

        let (device, dtype) = select_device(&engine_config.runtime)?;
        init_cpu_runtime_if_needed(&device, &engine_config.runtime);
        let load_start = Instant::now();

        let resolved = {
            let content = std::fs::read_to_string(&config_path)
                .map_err(|e| EngineError::Internal(format!("failed to read config.json: {e}")))?;
            parse_model_config_for_source(
                &content,
                task_override,
                has_remote_file(&repo, "config_sentence_transformers.json"),
            )?
        };
        let embedding_modules =
            load_embedding_modules_from_repo(
                &repo,
                resolved.spec.name(),
                resolved.task,
                DType::F32,
                &device,
            )?;

        let vb = load_var_builder_from_filenames(&weight_files, dtype, &device)?;
        let tokenizer = load_tokenizer_file(&tokenizer_path)?;
        load_safetensor_parts(
            repo_id.to_string(),
            resolved,
            embedding_modules,
            vb,
            tokenizer,
            device,
            dtype,
            load_start,
            engine_config,
        )
    }

    /// Load a GGUF model from an HF Hub repo that contains .gguf files but no config.json.
    /// Auto-selects the best GGUF file (prefers Q8_0 > Q4_K_M > largest).
    fn from_hf_hub_gguf(
        repo_id: &str,
        repo: &hf_hub::api::sync::ApiRepo,
        task_override: TaskOverride,
        engine_config: EngineConfig,
    ) -> Result<Self, EngineError> {
        let info = repo
            .info()
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

        // Select best GGUF: prefer Q8_0 > Q4_K_M > first available
        let selected = gguf_files
            .iter()
            .find(|f| f.contains("Q8_0"))
            .or_else(|| gguf_files.iter().find(|f| f.contains("Q4_K_M")))
            .or_else(|| gguf_files.first())
            .unwrap();

        tracing::info!(file = %selected, "auto-selected GGUF file from {repo_id}");

        let gguf_path = repo
            .get(selected)
            .map_err(|e| EngineError::Internal(format!("failed to download {selected}: {e}")))?;

        // Resolve tokenizer: try this repo first, then infer base model from GGUF metadata
        let tokenizer_model_id = resolve_gguf_tokenizer_repo(repo_id, repo, &gguf_path);

        load_gguf(&gguf_path, tokenizer_model_id, task_override, engine_config)
    }
}

/// Try to resolve the tokenizer source for a GGUF repo.
/// 1. If this repo has tokenizer.json, use it directly.
/// 2. Otherwise, read GGUF metadata for `general.base_model.0.repo_url` to find the base model.
/// 3. Fall back to stripping `-GGUF` suffix from repo_id.
fn resolve_gguf_tokenizer_repo(
    repo_id: &str,
    repo: &hf_hub::api::sync::ApiRepo,
    gguf_path: &Path,
) -> String {
    // Check if tokenizer.json exists next to GGUF or in repo
    if let Some(parent) = gguf_path.parent() {
        if parent.join("tokenizer.json").exists() {
            return repo_id.to_string();
        }
    }

    // Try reading base_model from GGUF metadata
    if let Ok(mut file) = std::fs::File::open(gguf_path) {
        if let Ok(ct) = candle_core::quantized::gguf_file::Content::read(&mut file) {
            if let Some(val) = ct.metadata.get("general.base_model.0.repo_url") {
                if let Ok(url) = val.to_string() {
                    // Extract "org/model" from "https://huggingface.co/org/model"
                    if let Some(repo) = url.strip_prefix("https://huggingface.co/") {
                        tracing::info!(base_model = %repo, "resolved tokenizer from GGUF metadata");
                        return repo.to_string();
                    }
                }
            }
        }
    }

    // Fallback: strip -GGUF suffix (e.g. "unsloth/Qwen3.5-0.8B-GGUF" → "unsloth/Qwen3.5-0.8B")
    let fallback = repo_id
        .strip_suffix("-GGUF")
        .or_else(|| repo_id.strip_suffix("-gguf"))
        .unwrap_or(repo_id)
        .to_string();
    tracing::info!(fallback = %fallback, "using fallback tokenizer repo (stripped -GGUF suffix)");
    fallback
}

// ── Free functions ────────────────────────────────────────────────────────

fn load_safetensor_parts(
    model_id: String,
    resolved: ResolvedModelConfig,
    embedding_modules: Option<LoadedEmbeddingModules>,
    vb: VarBuilder<'static>,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    load_start: Instant,
    engine_config: EngineConfig,
) -> Result<Engine, EngineError> {
    // Initialize global config before building the model, so that model
    // constructors can read cache/runtime settings via global accessors.
    crate::config::init_global_config(&engine_config);

    let built = build_model_variant(
        &resolved,
        vb,
        embedding_modules.as_ref(),
        &device,
        WeightsBackend::Safetensors,
    )?;
    let eos_token_ids = resolved.eos_token_ids;
    let common_config = resolved.parsed.common;
    let deltanet_config = resolved.parsed.deltanet;

    let executor = ModelExecutor {
        model: Mutex::new(built.model),
        device,
        dtype,
        config: common_config,
        runtime_caps: built.runtime_caps,
    };

    let cache = CacheManager::new(
        &executor.config,
        deltanet_config.as_ref(),
        executor.dtype,
        &executor.device,
        &executor.runtime_caps,
        &engine_config.cache,
    )?;

    tracing::info!(
        elapsed_ms = load_start.elapsed().as_millis() as u64,
        task = ?built.descriptor.task,
        arch = built.descriptor.arch_name,
        backend = ?built.descriptor.backend,
        runtime_caps = ?executor.runtime_caps,
        is_cuda = executor.device.is_cuda(),
        dtype = ?executor.dtype,
        layers = executor.config.num_hidden_layers,
        vocab = executor.config.vocab_size,
        "model loaded"
    );

    Ok(Engine {
        executor,
        cache,
        tokenizer,
        model_id,
        embedding_semantics: embedding_modules
            .as_ref()
            .map(|modules| modules.spec.clone())
            .unwrap_or_default(),
        eos_token_ids,
        descriptor: built.descriptor,
        engine_config,
    })
}

/// Detect GGUF architecture from metadata.
fn detect_gguf_arch(ct: &candle_core::quantized::gguf_file::Content) -> String {
    ct.metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().map(|s| s.to_string()))
        .unwrap_or_else(|| "qwen3".to_string())
}

/// Load a quantized model from a GGUF file.
fn load_gguf(
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
    let ct = candle_core::quantized::gguf_file::Content::read(&mut file).map_err(candle_err)?;

    let arch = detect_gguf_arch(&ct);
    tracing::info!(arch = %arch, "detected GGUF architecture");

    // Resolve tokenizer: look next to GGUF file, else download from HF Hub
    let tokenizer = if let Some(parent) = gguf_path.parent()
        && parent.join("tokenizer.json").exists()
    {
        load_tokenizer(parent)?
    } else {
        download_tokenizer(&model_id)?
    };

    // When ggml-quants is enabled, use llama.cpp FFI for ALL architectures
    #[cfg(feature = "ggml-quants")]
    {
        let n_gpu_layers = if device.is_cuda() { -1 } else { 0 };
        return load_gguf_llama_cpp(gguf_path, model_id, engine_config, load_start, tokenizer, n_gpu_layers);
    }

    #[cfg(not(feature = "ggml-quants"))]
    match arch.as_str() {
        "qwen35" | "qwen35moe" => load_gguf_qwen35(ct, &mut file, &device, model_id, engine_config, load_start, tokenizer),
        _ => load_gguf_qwen3(ct, &mut file, &device, model_id, engine_config, load_start, tokenizer),
    }
}

/// Load ANY GGUF model via llama.cpp FFI (requires `ggml-quants` feature).
#[cfg(feature = "ggml-quants")]
fn load_gguf_llama_cpp(
    gguf_path: &Path,
    model_id: String,
    engine_config: EngineConfig,
    load_start: Instant,
    tokenizer: Tokenizer,
    n_gpu_layers: i32,
) -> Result<Engine, EngineError> {
    use crate::models::qwen3_5::gguf::LlamaGgufModel;

    let model = LlamaGgufModel::load(gguf_path, n_gpu_layers, 4096)
        .map_err(|e| EngineError::Internal(e))?;

    let cfg = model.config();
    let eos_token_ids = vec![cfg.eos_token as u32];

    let common_config = CommonModelConfig {
        vocab_size: cfg.vocab_size,
        num_hidden_layers: cfg.num_hidden_layers,
        max_position_embeddings: cfg.max_position_embeddings,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
    };

    let descriptor = ModelDescriptor {
        task: TaskKind::Generate,
        arch_name: "llama_cpp_gguf",
        backend: WeightsBackend::Gguf,
    };

    let runtime_caps = RuntimeCaps {
        supports_kv_cache: true,
        ..RuntimeCaps::default()
    };

    let device = Device::Cpu; // llama.cpp manages its own device
    let executor = ModelExecutor {
        model: Mutex::new(Box::new(model)),
        device,
        dtype: DType::F32,
        config: common_config,
        runtime_caps,
    };

    tracing::info!(
        elapsed_ms = load_start.elapsed().as_millis() as u64,
        arch = descriptor.arch_name,
        gpu_layers = n_gpu_layers,
        layers = executor.config.num_hidden_layers,
        vocab = executor.config.vocab_size,
        "llama.cpp GGUF model loaded"
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

/// Load Qwen3.5 (hybrid DeltaNet) from GGUF.
fn load_gguf_qwen35(
    ct: candle_core::quantized::gguf_file::Content,
    file: &mut std::fs::File,
    device: &Device,
    model_id: String,
    engine_config: EngineConfig,
    load_start: Instant,
    tokenizer: Tokenizer,
) -> Result<Engine, EngineError> {
    let (model, gguf_config) =
        Qwen3_5GgufModel::from_gguf(ct, file, device).map_err(candle_err)?;

    let eos_token_ids = if gguf_config.eos_token_ids.is_empty() {
        return Err(EngineError::InvalidRequest(
            "GGUF metadata missing `tokenizer.ggml.eos_token_id`".into(),
        ));
    } else {
        gguf_config.eos_token_ids.clone()
    };

    let common_config = CommonModelConfig {
        vocab_size: gguf_config.vocab_size,
        num_hidden_layers: gguf_config.num_hidden_layers,
        max_position_embeddings: gguf_config.max_position_embeddings,
        num_attention_heads: gguf_config.num_attention_heads,
        num_key_value_heads: gguf_config.num_key_value_heads,
        head_dim: gguf_config.head_dim,
    };

    let descriptor = ModelDescriptor {
        task: TaskKind::Generate,
        arch_name: "qwen3_5_gguf",
        backend: WeightsBackend::Gguf,
    };
    let runtime_caps = RuntimeCaps {
        supports_kv_cache: true,
        ..RuntimeCaps::default()
    };

    let executor = ModelExecutor {
        model: Mutex::new(Box::new(model)),
        device: device.clone(),
        dtype: DType::F32,
        config: common_config,
        runtime_caps,
    };

    tracing::info!(
        elapsed_ms = load_start.elapsed().as_millis() as u64,
        arch = descriptor.arch_name,
        layers = executor.config.num_hidden_layers,
        vocab = executor.config.vocab_size,
        "Qwen3.5 GGUF model loaded"
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

/// Load standard Qwen3 (and other dense architectures) from GGUF.
fn load_gguf_qwen3(
    ct: candle_core::quantized::gguf_file::Content,
    file: &mut std::fs::File,
    device: &Device,
    model_id: String,
    engine_config: EngineConfig,
    load_start: Instant,
    tokenizer: Tokenizer,
) -> Result<Engine, EngineError> {
    let (model, gguf_config) =
        Qwen3GgufModel::from_gguf(ct, file, device).map_err(candle_err)?;

    let eos_token_ids = if gguf_config.eos_token_ids.is_empty() {
        return Err(EngineError::InvalidRequest(
            "generation GGUF metadata is missing required field `tokenizer.ggml.eos_token_id`"
                .into(),
        ));
    } else {
        gguf_config.eos_token_ids.clone()
    };

    let common_config = CommonModelConfig {
        vocab_size: gguf_config.vocab_size,
        num_hidden_layers: gguf_config.num_hidden_layers,
        max_position_embeddings: gguf_config.max_position_embeddings,
        num_attention_heads: gguf_config.num_attention_heads,
        num_key_value_heads: gguf_config.num_key_value_heads,
        head_dim: gguf_config.head_dim,
    };

    let descriptor = ModelDescriptor {
        task: TaskKind::Generate,
        arch_name: "qwen3_gguf",
        backend: WeightsBackend::Gguf,
    };
    let runtime_caps = RuntimeCaps {
        supports_kv_cache: true,
        ..RuntimeCaps::default()
    };

    let executor = ModelExecutor {
        model: Mutex::new(Box::new(model)),
        device: device.clone(),
        dtype: DType::F32,
        config: common_config,
        runtime_caps,
    };

    tracing::info!(
        elapsed_ms = load_start.elapsed().as_millis() as u64,
        task = ?descriptor.task,
        arch = descriptor.arch_name,
        backend = ?descriptor.backend,
        runtime_caps = ?executor.runtime_caps,
        layers = executor.config.num_hidden_layers,
        vocab = executor.config.vocab_size,
        "GGUF model loaded"
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

fn load_embedding_modules_from_dir(
    model_path: &Path,
    arch_name: &str,
    task: TaskKind,
    dtype: DType,
    device: &Device,
) -> Result<Option<LoadedEmbeddingModules>, EngineError> {
    if task != TaskKind::Embed {
        return Ok(None);
    }

    let modules_path = model_path.join("modules.json");
    if !modules_path.exists() {
        tracing::warn!(
            path = %modules_path.display(),
            "embedding modules.json not found; using default last-token semantics"
        );
        return Ok(None);
    }

    load_embedding_modules_from_file(
        &modules_path,
        |relative| Ok(model_path.join(relative)),
        dtype,
        device,
        arch_name == "gemma3",
    )
}

fn load_embedding_modules_from_repo(
    repo: &hf_hub::api::sync::ApiRepo,
    arch_name: &str,
    task: TaskKind,
    dtype: DType,
    device: &Device,
) -> Result<Option<LoadedEmbeddingModules>, EngineError> {
    if task != TaskKind::Embed {
        return Ok(None);
    }

    let modules_path = match repo.get("modules.json") {
        Ok(path) => path,
        Err(err) => {
            tracing::warn!(error = %err, "failed to resolve embedding modules.json");
            return Ok(None);
        }
    };

    load_embedding_modules_from_file(
        &modules_path,
        |relative| {
            repo.get(relative).map_err(|err| {
                EngineError::Internal(format!("failed to download {relative}: {err}"))
            })
        },
        dtype,
        device,
        arch_name == "gemma3",
    )
}

fn load_embedding_modules_from_file<F>(
    modules_path: &Path,
    mut resolve_path: F,
    dtype: DType,
    device: &Device,
    load_dense_auxiliary: bool,
) -> Result<Option<LoadedEmbeddingModules>, EngineError>
where
    F: FnMut(&str) -> Result<PathBuf, EngineError>,
{
    let content = std::fs::read_to_string(modules_path).map_err(|err| {
        EngineError::Internal(format!(
            "failed to read embedding modules.json {}: {err}",
            modules_path.display()
        ))
    })?;

    let mut modules: Vec<SentenceTransformerModuleEntry> =
        serde_json::from_str(&content).map_err(|err| {
            EngineError::Internal(format!(
                "failed to parse embedding modules.json {}: {err}",
                modules_path.display()
            ))
        })?;
    modules.sort_by_key(|entry| entry.idx);

    let mut spec = EmbeddingSemantics::default();
    let mut auxiliary = Vec::new();

    for entry in modules {
        let is_normalize = entry.module_type.ends_with(".Normalize");
        if spec.normalization == EmbeddingNormalization::L2 && !is_normalize {
            return Err(EngineError::InvalidRequest(format!(
                "unsupported sentence-transformers module order in {}: Normalize must be the final module",
                modules_path.display()
            )));
        }

        if entry.module_type.ends_with(".Transformer") {
            continue;
        }

        if entry.module_type.ends_with(".Pooling") {
            let config_path = resolve_path(&module_relative_path(&entry.path, "config.json"))?;
            let pooling_cfg: SentenceTransformerPoolingConfig =
                read_json_file(&config_path, "sentence-transformers pooling config")?;
            spec.pooling = pooling_from_config(&pooling_cfg, &config_path)?;
            continue;
        }

        if entry.module_type.ends_with(".Dense") {
            let config_path = resolve_path(&module_relative_path(&entry.path, "config.json"))?;
            let dense_cfg: SentenceTransformerDenseConfig =
                read_json_file(&config_path, "sentence-transformers dense config")?;
            let activation = resolve_dense_activation(&dense_cfg)?;
            let bias = if load_dense_auxiliary {
                let weight_path =
                    resolve_path(&module_relative_path(&entry.path, "model.safetensors"))?;
                let vb = load_var_builder_from_filenames(&[weight_path.clone()], dtype, device)?;
                auxiliary.push(AuxiliaryVarBuilder {
                    module_path: entry.path.clone(),
                    vb,
                });
                dense_bias_from_config_or_weights(dense_cfg.bias, &weight_path)?
            } else {
                dense_cfg.bias.unwrap_or(false)
            };

            spec.dense_layers.push(EmbeddingDenseLayerSpec {
                module_path: entry.path.clone(),
                in_features: dense_cfg.in_features,
                out_features: dense_cfg.out_features,
                bias,
                activation,
            });
            continue;
        }

        if entry.module_type.ends_with(".Normalize") {
            if spec.normalization == EmbeddingNormalization::L2 {
                return Err(EngineError::InvalidRequest(format!(
                    "unsupported sentence-transformers module order in {}: multiple Normalize modules are not supported",
                    modules_path.display()
                )));
            }
            spec.normalization = EmbeddingNormalization::L2;
            continue;
        }

        return Err(EngineError::InvalidRequest(format!(
            "unsupported sentence-transformers module type for embeddings: {}",
            entry.module_type
        )));
    }

    Ok(Some(LoadedEmbeddingModules { spec, auxiliary }))
}

fn module_relative_path(module_path: &str, filename: &str) -> String {
    if module_path.is_empty() {
        filename.to_string()
    } else {
        format!("{module_path}/{filename}")
    }
}

fn dense_bias_from_config_or_weights(
    config_bias: Option<bool>,
    weight_path: &Path,
) -> Result<bool, EngineError> {
    if let Some(bias) = config_bias {
        return Ok(bias);
    }

    // Some sentence-transformers dense configs omit `bias`; use the safetensor
    // payload as the source of truth so we don't silently drop a present bias.
    let weights = unsafe { candle_core::safetensors::MmapedSafetensors::new(weight_path) }
        .map_err(candle_err)?;
    Ok(weights
        .tensors()
        .into_iter()
        .any(|(name, _)| name == "linear.bias"))
}

fn pooling_from_config(
    cfg: &SentenceTransformerPoolingConfig,
    path: &Path,
) -> Result<EmbeddingPooling, EngineError> {
    let supported = [
        (cfg.pooling_mode_lasttoken, EmbeddingPooling::LastToken),
        (cfg.pooling_mode_mean_tokens, EmbeddingPooling::Mean),
        (cfg.pooling_mode_cls_token, EmbeddingPooling::Cls),
    ];
    let enabled: Vec<_> = supported
        .into_iter()
        .filter_map(|(enabled, pooling): (bool, EmbeddingPooling)| enabled.then_some(pooling))
        .collect();

    if enabled.len() == 1
        && !cfg.pooling_mode_max_tokens
        && !cfg.pooling_mode_mean_sqrt_len_tokens
        && !cfg.pooling_mode_weightedmean_tokens
    {
        return Ok(enabled[0]);
    }

    Err(EngineError::InvalidRequest(format!(
        "unsupported sentence-transformers pooling config {}",
        path.display()
    )))
}

fn parse_dense_activation(value: Option<&str>) -> Result<EmbeddingActivation, EngineError> {
    let normalized = value.unwrap_or("").trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "torch.nn.modules.linear.identity" | "torch.nn.identity" | "identity" => {
            Ok(EmbeddingActivation::Identity)
        }
        _ => Err(EngineError::InvalidRequest(format!(
            "unsupported sentence-transformers dense activation: {}",
            value.unwrap_or("")
        ))),
    }
}

fn resolve_dense_activation(
    cfg: &SentenceTransformerDenseConfig,
) -> Result<EmbeddingActivation, EngineError> {
    let activation_function = cfg.activation_function.as_deref();
    let activation = cfg.activation.as_deref();

    if let (Some(lhs), Some(rhs)) = (activation_function, activation) {
        if !lhs.trim().eq_ignore_ascii_case(rhs.trim()) {
            return Err(EngineError::InvalidRequest(format!(
                "conflicting sentence-transformers dense activation values: activation_function={lhs}, activation={rhs}"
            )));
        }
    }

    parse_dense_activation(activation_function.or(activation))
}

fn read_json_file<T: serde::de::DeserializeOwned>(
    path: &Path,
    description: &str,
) -> Result<T, EngineError> {
    let content = std::fs::read_to_string(path).map_err(|err| {
        EngineError::Internal(format!(
            "failed to read {description} {}: {err}",
            path.display()
        ))
    })?;
    serde_json::from_str(&content).map_err(|err| {
        EngineError::Internal(format!(
            "failed to parse {description} {}: {err}",
            path.display()
        ))
    })
}

fn load_tokenizer(model_path: &Path) -> Result<Tokenizer, EngineError> {
    let tokenizer_path = model_path.join("tokenizer.json");
    load_tokenizer_file(&tokenizer_path)
}

fn load_tokenizer_file(tokenizer_path: &Path) -> Result<Tokenizer, EngineError> {
    Tokenizer::from_file(tokenizer_path).map_err(|e| {
        EngineError::Internal(format!("failed to load {}: {e}", tokenizer_path.display()))
    })
}

/// Download tokenizer.json from HuggingFace Hub.
fn download_tokenizer(model_id: &str) -> Result<Tokenizer, EngineError> {
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

pub(crate) struct BuiltModel {
    pub model: ModelVariant,
    pub descriptor: ModelDescriptor,
    pub runtime_caps: RuntimeCaps,
}

pub(crate) fn build_model_variant(
    resolved: &ResolvedModelConfig,
    vb: VarBuilder<'_>,
    embedding_modules: Option<&LoadedEmbeddingModules>,
    device: &Device,
    backend: WeightsBackend,
) -> Result<BuiltModel, EngineError> {
    let model = if resolved.spec.name() == "gemma3" && resolved.task == TaskKind::Embed {
        let build_ctx = Gemma3ModelBuildContext {
            main_vb: vb,
            embedding: embedding_modules.map(|modules| &modules.spec),
            auxiliary: embedding_modules
                .map(|modules| modules.auxiliary.as_slice())
                .unwrap_or(&[]),
        };
        build_gemma3_model_with_context(resolved.parsed.arch_config.as_ref(), &build_ctx)?
    } else {
        resolved
            .spec
            .build_model(resolved.parsed.arch_config.as_ref(), vb)?
    };

    Ok(BuiltModel {
        model,
        descriptor: ModelDescriptor {
            task: resolved.task,
            arch_name: resolved.spec.name(),
            backend,
        },
        runtime_caps: resolved.spec.runtime_caps(resolved.task, backend, device),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::safetensors::save as save_safetensors;
    use serde_json::json;
    use std::collections::HashMap;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    const TEST_SENTENCE_TRANSFORMER_MODEL_DIM: usize = 768;
    const TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM: usize = 3072;
    const TEST_SENTENCE_TRANSFORMER_DENSE_ACTIVATION: &str = "torch.nn.Identity";

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(name: &str) -> Self {
            let id = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "prelude-{name}-{}-{}",
                std::process::id(),
                id
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_json(path: &Path, value: &serde_json::Value) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
    }

    fn write_dense_weights_with_bias(
        path: &Path,
        in_features: usize,
        out_features: usize,
        include_bias: bool,
    ) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let weight = Tensor::from_vec(
            vec![0f32; in_features * out_features],
            (out_features, in_features),
            &Device::Cpu,
        )
        .unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("linear.weight".to_string(), weight);
        if include_bias {
            let bias =
                Tensor::from_vec(vec![0f32; out_features], out_features, &Device::Cpu).unwrap();
            tensors.insert("linear.bias".to_string(), bias);
        }
        save_safetensors(&tensors, path).unwrap();
    }

    fn write_dense_weights(path: &Path, in_features: usize, out_features: usize) {
        write_dense_weights_with_bias(path, in_features, out_features, false);
    }

    fn sentence_transformers_pooling_config() -> serde_json::Value {
        json!({
            "pooling_mode_cls_token": false,
            "pooling_mode_mean_tokens": true,
            "pooling_mode_lasttoken": false,
            "pooling_mode_max_tokens": false,
            "pooling_mode_mean_sqrt_len_tokens": false,
            "pooling_mode_weightedmean_tokens": false
        })
    }

    fn sentence_transformers_dense_config(
        in_features: usize,
        out_features: usize,
        bias: Option<bool>,
        activation: Option<&str>,
    ) -> serde_json::Value {
        let mut config = json!({
            "in_features": in_features,
            "out_features": out_features
        });
        if let Some(bias) = bias {
            config["bias"] = json!(bias);
        }
        if let Some(activation) = activation {
            config["activation_function"] = json!(activation);
        }
        config
    }

    fn sentence_transformers_dense_config_with_activation_alias(
        in_features: usize,
        out_features: usize,
        activation: &str,
    ) -> serde_json::Value {
        json!({
            "in_features": in_features,
            "out_features": out_features,
            "activation": activation
        })
    }

    #[test]
    fn parses_sentence_transformers_embedding_modules() {
        let dir = TestDir::new("sentence-transformers-modules");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling"
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Dense",
                    "type": "sentence_transformers.models.Dense"
                },
                {
                    "idx": 3,
                    "name": "3",
                    "path": "3_Dense",
                    "type": "sentence_transformers.models.Dense"
                },
                {
                    "idx": 4,
                    "name": "4",
                    "path": "4_Normalize",
                    "type": "sentence_transformers.models.Normalize"
                }
            ]),
        );

        write_json(
            &dir.path.join("1_Pooling/config.json"),
            &sentence_transformers_pooling_config(),
        );
        write_json(
            &dir.path.join("2_Dense/config.json"),
            &sentence_transformers_dense_config(
                TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
                TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
                Some(false),
                Some(TEST_SENTENCE_TRANSFORMER_DENSE_ACTIVATION),
            ),
        );
        write_json(
            &dir.path.join("3_Dense/config.json"),
            &sentence_transformers_dense_config(
                TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
                TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
                Some(false),
                None,
            ),
        );
        write_dense_weights(
            &dir.path.join("2_Dense/model.safetensors"),
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
        );
        write_dense_weights(
            &dir.path.join("3_Dense/model.safetensors"),
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
        );

        let loaded = load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            true,
        )
        .unwrap()
        .unwrap();

        assert_eq!(loaded.spec.pooling, EmbeddingPooling::Mean);
        assert_eq!(loaded.spec.normalization, EmbeddingNormalization::L2);
        assert_eq!(loaded.spec.dense_layers.len(), 2);
        assert_eq!(loaded.spec.dense_layers[0].module_path, "2_Dense");
        assert_eq!(
            loaded.spec.dense_layers[0].in_features,
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM
        );
        assert_eq!(
            loaded.spec.dense_layers[0].out_features,
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM
        );
        assert_eq!(
            loaded.spec.dense_layers[0].activation,
            EmbeddingActivation::Identity
        );
        assert_eq!(loaded.spec.dense_layers[1].module_path, "3_Dense");
        assert_eq!(
            loaded.spec.dense_layers[1].in_features,
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM
        );
        assert_eq!(
            loaded.spec.dense_layers[1].out_features,
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM
        );
        assert_eq!(loaded.auxiliary.len(), 2);
        assert_eq!(loaded.auxiliary[0].module_path, "2_Dense");
        assert_eq!(loaded.auxiliary[1].module_path, "3_Dense");
    }

    #[test]
    fn infers_embedding_dense_bias_from_weights_when_config_omits_flag() {
        let dir = TestDir::new("sentence-transformers-dense-bias-inference");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense"
                }
            ]),
        );
        write_json(
            &dir.path.join("1_Dense/config.json"),
            &sentence_transformers_dense_config(
                TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
                TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
                None,
                None,
            ),
        );
        write_dense_weights_with_bias(
            &dir.path.join("1_Dense/model.safetensors"),
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
            true,
        );

        let loaded = load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            true,
        )
        .unwrap()
        .unwrap();

        assert_eq!(loaded.spec.dense_layers.len(), 1);
        assert!(loaded.spec.dense_layers[0].bias);
    }

    #[test]
    fn parses_embedding_dense_activation_alias_key() {
        let dir = TestDir::new("sentence-transformers-dense-activation-alias");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense"
                }
            ]),
        );
        write_json(
            &dir.path.join("1_Dense/config.json"),
            &sentence_transformers_dense_config_with_activation_alias(
                TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
                TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
                TEST_SENTENCE_TRANSFORMER_DENSE_ACTIVATION,
            ),
        );
        write_dense_weights(
            &dir.path.join("1_Dense/model.safetensors"),
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
        );

        let loaded = load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            true,
        )
        .unwrap()
        .unwrap();

        assert_eq!(loaded.spec.dense_layers.len(), 1);
        assert_eq!(
            loaded.spec.dense_layers[0].activation,
            EmbeddingActivation::Identity
        );
    }

    #[test]
    fn rejects_non_terminal_sentence_transformers_normalize_module() {
        let dir = TestDir::new("sentence-transformers-normalize-non-terminal");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Normalize",
                    "type": "sentence_transformers.models.Normalize"
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Dense",
                    "type": "sentence_transformers.models.Dense"
                }
            ]),
        );

        let err = match load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            true,
        ) {
            Ok(_) => panic!("expected non-terminal Normalize module order to be rejected"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("Normalize must be the final module"),
            "{err}"
        );
    }

    #[test]
    fn rejects_multiple_sentence_transformers_normalize_modules() {
        let dir = TestDir::new("sentence-transformers-normalize-duplicate");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Normalize",
                    "type": "sentence_transformers.models.Normalize"
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Normalize",
                    "type": "sentence_transformers.models.Normalize"
                }
            ]),
        );

        let err = match load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            true,
        ) {
            Ok(_) => panic!("expected duplicate Normalize modules to be rejected"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("multiple Normalize modules are not supported"),
            "{err}"
        );
    }

    #[test]
    fn skips_dense_weight_loading_for_non_gemma_embeddings() {
        let dir = TestDir::new("sentence-transformers-dense-metadata-only");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense"
                }
            ]),
        );
        write_json(
            &dir.path.join("1_Dense/config.json"),
            &sentence_transformers_dense_config(
                TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
                TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
                Some(true),
                Some(TEST_SENTENCE_TRANSFORMER_DENSE_ACTIVATION),
            ),
        );

        let loaded = load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            false,
        )
        .unwrap()
        .unwrap();

        assert_eq!(loaded.spec.dense_layers.len(), 1);
        assert!(loaded.spec.dense_layers[0].bias);
        assert!(loaded.auxiliary.is_empty());
    }
}
