//! Engine construction: local path, HF Hub, safetensors, GGUF.
//!
//! All factory functions return a fully assembled `Engine`.

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use super::weight_loader::VarBuilder;
use crate::tensor::{DType, Device};
use fastokens::Tokenizer;

use crate::cache::manager::CacheManager;
use crate::config::EngineConfig;
use crate::engine::{
    EmbeddingSemantics, Engine, EngineError, ModelDescriptor, ModelExecutor, ModelVariant,
    ResolvedModelConfig, RuntimeCaps, TaskKind, TaskOverride, WeightsBackend, has_remote_file,
    init_runtime, load_model_config, load_safetensor_filenames, load_var_builder_from_filenames,
    load_weights, parse_model_config_for_source, select_device, tensor_err,
};
use crate::models::gemma3::meta::{Gemma3ModelBuildContext, build_gemma3_model_with_context};
use crate::tensor::quantized::gguf_file::Content as GgufContent;

const GGUF_BASE_MODEL_REPO_URL_KEY: &str = "general.base_model.0.repo_url";
const HF_REPO_URL_PREFIX: &str = "https://huggingface.co/";

mod embedding_modules;

use self::embedding_modules::{
    LoadedEmbeddingModules, load_embedding_modules_from_dir, load_embedding_modules_from_repo,
};

impl Engine {
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
        init_runtime(&device, &engine_config.runtime);

        let resolved = load_model_config(model_path, task_override)?;
        let embedding_modules = load_embedding_modules_from_dir(
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

    /// Sync wrapper around [`Self::from_hf_hub_with_task_async`] for
    /// callers outside an async context. Spins up a single-threaded
    /// tokio runtime to await the async impl. Async callers should use
    /// `from_hf_hub_with_task_async` directly.
    ///
    /// Panics if called from inside an existing tokio runtime — use the
    /// `_async` variant in that case.
    pub fn from_hf_hub_with_task(
        repo_id: &str,
        task_override: TaskOverride,
        engine_config: EngineConfig,
    ) -> Result<Self, EngineError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| EngineError::Internal(format!("failed to build tokio runtime: {e}")))?;
        runtime.block_on(Self::from_hf_hub_with_task_async(
            repo_id,
            task_override,
            engine_config,
        ))
    }

    pub async fn from_hf_hub_with_task_async(
        repo_id: &str,
        task_override: TaskOverride,
        engine_config: EngineConfig,
    ) -> Result<Self, EngineError> {
        let api = hf_hub::api::tokio::Api::new()
            .map_err(|e| EngineError::Internal(format!("failed to init hf-hub api: {e}")))?;
        let repo = api.model(repo_id.to_string());

        tracing::info!(repo = repo_id, "downloading model from HuggingFace Hub");

        // Try config.json — if missing, this might be a GGUF-only repo
        let config_path = match repo.get("config.json").await {
            Ok(path) => path,
            Err(_) => {
                tracing::info!("config.json not found, checking for GGUF files");
                return Self::from_hf_hub_gguf(repo_id, &repo, task_override, engine_config).await;
            }
        };
        let tokenizer_path = repo.get("tokenizer.json").await.map_err(|e| {
            EngineError::Internal(format!("failed to download tokenizer.json: {e}"))
        })?;

        let weight_files = load_safetensor_filenames(&repo).await?;

        let (device, dtype) = select_device(&engine_config.runtime)?;
        init_runtime(&device, &engine_config.runtime);
        let load_start = Instant::now();

        let resolved = {
            let content = std::fs::read_to_string(&config_path)
                .map_err(|e| EngineError::Internal(format!("failed to read config.json: {e}")))?;
            parse_model_config_for_source(
                &content,
                task_override,
                has_remote_file(&repo, "config_sentence_transformers.json").await,
            )?
        };
        let embedding_modules = load_embedding_modules_from_repo(
            &repo,
            resolved.spec.name(),
            resolved.task,
            DType::F32,
            &device,
        )
        .await?;

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
    async fn from_hf_hub_gguf(
        repo_id: &str,
        repo: &hf_hub::api::tokio::ApiRepo,
        task_override: TaskOverride,
        engine_config: EngineConfig,
    ) -> Result<Self, EngineError> {
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
}

/// Try to resolve the tokenizer source for a GGUF repo.
/// 1. If this repo has tokenizer.json, use it directly.
/// 2. Otherwise, read GGUF metadata for `general.base_model.0.repo_url` to find the base model.
/// 3. Fall back to stripping `-GGUF` suffix from repo_id.
fn resolve_gguf_tokenizer_repo(repo_id: &str, gguf_path: &Path) -> String {
    // Check if tokenizer.json exists next to GGUF or in repo
    if let Some(parent) = gguf_path.parent() {
        if parent.join("tokenizer.json").exists() {
            return repo_id.to_string();
        }
    }

    // Try reading base_model from GGUF metadata
    if let Ok(mut file) = std::fs::File::open(gguf_path) {
        if let Ok(ct) = GgufContent::read(&mut file)
            && let Some(repo) = gguf_base_model_repo(&ct)
        {
            tracing::info!(base_model = %repo, "resolved tokenizer from GGUF metadata");
            return repo;
        }
    }

    // Fallback: strip -GGUF suffix (e.g. "unsloth/Qwen3.5-0.8B-GGUF" → "unsloth/Qwen3.5-0.8B")
    let fallback = stripped_gguf_repo_id(repo_id).unwrap_or_else(|| repo_id.to_string());
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

    let mut built = build_model_variant(
        &resolved,
        vb,
        embedding_modules.as_ref(),
        &device,
        WeightsBackend::Safetensors,
    )?;
    let eos_token_ids = resolved.eos_token_ids;
    let common_config = resolved.parsed.common;
    let deltanet_config = resolved.parsed.deltanet;

    // Read KV sharing map from model before moving it into the executor.
    let kv_sharing = built.model.kv_cache_sharing();

    // ── Activation memory profiling (vLLM-style) ──────────────────
    //
    // Before sizing the KV cache, run a dummy forward pass to measure
    // peak GPU memory consumed by activations. This is the same
    // approach vLLM uses in `determine_available_memory`:
    //
    //   available_kv = total * utilization - weights - peak_activation
    //
    // Without this, the 10% reserve from `gpu_memory_utilization=0.9`
    // can be insufficient for large-vocab MoE models (Qwen3.5-35B-A3B
    // needs ~8 GB for the lm_head matmul alone at 8192 tokens).
    let peak_activation_bytes = if device.is_cuda() {
        profile_peak_activation(
            &mut built.model,
            &device,
            &common_config,
            dtype,
            engine_config.runtime.profile_tokens,
        )
        .unwrap_or(0)
    } else {
        0
    };

    let executor = ModelExecutor {
        model: Mutex::new(built.model),
        ops: crate::ops::select_ops(&device),
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
        &kv_sharing,
        peak_activation_bytes,
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

/// Guess the HF repo that ships the tokenizer for this GGUF when the
/// metadata doesn't carry `general.base_model.0.repo_url`. Pieces it
/// together from `general.basename` + `general.size_label` and a small
/// per-arch org map (Qwen, Meta-Llama, Google).
///
/// Covers the common case of `Qwen/Qwen3-*-GGUF` — those uploads drop
/// the base_model URL but encode enough metadata to reconstruct the
/// canonical "Qwen/Qwen3-0.6B" repo. Returns `None` for unmapped arches
/// (Mistral, DeepSeek, etc.); callers fall back to `model_id` and
/// surface a clear tokenizer-download error.
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
    let ct = GgufContent::read(&mut file).map_err(tensor_err)?;

    let arch = detect_gguf_arch(&ct);
    tracing::info!(arch = %arch, "detected GGUF architecture");

    // Resolve tokenizer:
    //   1. tokenizer.json sitting next to the GGUF file → use it directly.
    //   2. `general.base_model.0.repo_url` in GGUF metadata → derive HF repo.
    //   3. Strip `-GGUF` suffix from `model_id` (covers `Qwen/X-GGUF` →
    //      `Qwen/X` for users who passed an HF repo).
    //   4. Per-arch heuristic from `general.basename` + `general.size_label`
    //      (covers community GGUF uploads that drop the base_model URL —
    //      most notably `Qwen/Qwen3-*-GGUF`).
    //   5. Fall back to `model_id` as-is (which 404s — we surface a
    //      tokenizer download error rather than crashing with a path).
    //
    // Steps 2 and 4 are what make `--model /path/to/foo.gguf` work without
    // requiring a separate `--tokenizer` flag.
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

    // Look up architecture in the registry (inventory-based, no hardcoded model imports)
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

/// Profile peak GPU activation memory via a dummy forward pass.
///
/// Mirrors vLLM's `determine_available_memory` approach: measures the
/// GPU memory consumed by a single forward pass with
/// `max_num_batched_tokens` tokens, so the KV cache auto-sizer can
/// subtract this from available memory instead of using a fixed 10%
/// reserve that's insufficient for large-vocab MoE models.
///
/// Since our forward requires a full `BatchAttnContext` (which in turn
/// requires the KV cache we're trying to size — circular dependency),
/// we use `forward_with_cache` (the simpler non-paged path) with a
/// dummy input. This exercises the full model (all layers + lm_head)
/// without needing a paged KV cache.
///
/// Returns 0 on any failure — the caller falls back to the old
/// heuristic (which works for small models; only large-vocab MoE
/// models hit the edge case).
fn profile_peak_activation(
    model: &mut ModelVariant,
    device: &Device,
    config: &crate::engine::CommonModelConfig,
    dtype: DType,
    profile_tokens: usize,
) -> Result<usize, EngineError> {
    use crate::tensor::Tensor;

    let ops = crate::ops::select_ops(device);

    let free_before = ops.gpu_free_memory().unwrap_or(0);
    let total = ops.gpu_total_memory().unwrap_or(0);
    if free_before == 0 || total == 0 {
        return Ok(0);
    }

    // Profile at the configured max_num_batched_tokens (matches vLLM's
    // approach). This avoids linear extrapolation errors from non-linear
    // costs (attention workspace, MoE dispatch buffers). Caller threads
    // the value in from `RuntimeConfig::profile_tokens`, which the server
    // CLI sets from `--max-num-batched-tokens` so the profile shape and
    // the scheduler's per-step token budget always agree.
    let profile_tokens = profile_tokens.max(1);

    tracing::info!(
        profile_tokens,
        free_before_mb = free_before / (1024 * 1024),
        "profiling peak activation memory"
    );

    // Run a dummy forward through the model's cached path (doesn't need
    // paged KV cache — uses simple per-layer KV caching internally).
    //
    // Shape is 1D `(profile_tokens,)`, matching the packed-token layout
    // every model's `forward_with_cache` expects (the attention layer
    // reads `seq_len = x.dim(0)`). Earlier versions of this code passed
    // `(1, profile_tokens)`, which made `seq_len` resolve to 1 and the
    // profiler measured only one token's worth of activation — the
    // result was always tiny and KV auto-sizing silently fell back to
    // the old heuristic. Verified against `qwen3.rs::Attention::forward_with_cache`.
    let dummy_input =
        Tensor::zeros((profile_tokens,), crate::tensor::DType::U32, device).map_err(tensor_err)?;

    // forward_with_cache runs through all layers + lm_head, producing
    // [profile_tokens, vocab_size] logits — this is the peak
    // activation consumer.
    if let Some(m) = model.as_kv_cache_model() {
        let logits_result = m.forward_with_cache(&dummy_input, 0);
        let _logits = match logits_result {
            Ok(l) => l,
            Err(err) => {
                tracing::warn!(
                    %err,
                    profile_tokens,
                    "activation profiling forward failed — falling back to old KV sizing heuristic"
                );
                model.clear_kv_cache();
                return Ok(0);
            }
        };
        // logits are alive → peak memory is captured by cudaMemGetInfo
        let free_during = ops.gpu_free_memory().unwrap_or(free_before);
        let peak_activation = free_before.saturating_sub(free_during);

        drop(_logits);
        model.clear_kv_cache();

        tracing::info!(
            peak_activation_mb = peak_activation / (1024 * 1024),
            profile_tokens,
            "activation profiling complete"
        );

        Ok(peak_activation)
    } else {
        // Model doesn't support forward_with_cache (unusual).
        // Fall back to config-based estimate.
        let lm_head_bytes = profile_tokens * config.vocab_size * dtype.size_in_bytes();
        tracing::info!(
            lm_head_mb = lm_head_bytes / (1024 * 1024),
            "using config-based activation estimate (no KV cache model)"
        );
        Ok(lm_head_bytes)
    }
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
