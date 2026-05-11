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
    Engine, EngineError, ModelDescriptor, ModelExecutor, ModelVariant, ResolvedModelConfig,
    RuntimeCaps, TaskKind, TaskOverride, WeightsBackend, has_remote_file, init_runtime,
    load_model_config, load_safetensor_filenames, load_var_builder_from_filenames, load_weights,
    parse_model_config_for_source, select_device, tensor_err,
};
use crate::models::gemma3::meta::{Gemma3ModelBuildContext, build_gemma3_model_with_context};

mod embedding_modules;
mod gguf;
mod tokenizer;

use self::embedding_modules::{
    LoadedEmbeddingModules, load_embedding_modules_from_dir, load_embedding_modules_from_repo,
};
use self::gguf::{load_gguf, load_hf_hub_gguf};
use self::tokenizer::{load_tokenizer, load_tokenizer_file};

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
                return load_hf_hub_gguf(repo_id, &repo, task_override, engine_config).await;
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
