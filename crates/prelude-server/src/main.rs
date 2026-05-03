#[global_allocator]
static GLOBAL: bc_mimalloc::MiMalloc = bc_mimalloc::MiMalloc;

use std::net::SocketAddr;
use std::sync::Arc;

use clap::{ArgAction, Parser, ValueEnum};
use prelude_core::{
    Engine, EngineConfig, InferenceEngine, MoeBackendPolicy, PseudoEngine, ScheduledEngine,
    SchedulerConfig, TaskOverride,
};
use prelude_server::chat_template::ModelChatTemplate;
use tracing::info;

#[derive(Debug, Parser)]
#[command(author, version, about = "Prelude HTTP server")]
struct Cli {
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    #[arg(long, default_value_t = 8000)]
    port: u16,

    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    model: String,

    #[arg(
        long,
        help = "Local path to model directory (config.json + safetensors + tokenizer.json) or a .gguf file"
    )]
    model_path: Option<String>,

    #[arg(
        long,
        default_value_t = false,
        help = "Use pseudo engine (mock) instead of real model"
    )]
    pseudo: bool,

    #[arg(long, default_value_t = 25)]
    pseudo_latency_ms: u64,

    #[arg(
        long,
        default_value_t = 32,
        help = "Dynamic batching max requests per model call"
    )]
    max_batch_size: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Dynamic batching max wait time in milliseconds"
    )]
    max_batch_wait_ms: u64,

    #[arg(
        long,
        default_value_t = 256,
        help = "Max concurrent running requests in scheduler"
    )]
    max_running_requests: usize,

    #[arg(
        long,
        default_value_t = 8192,
        help = "Per-step total token budget (prefill + decode combined)"
    )]
    max_num_batched_tokens: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-request prefill token cap (0 = no cap, only limited by per-step budget)"
    )]
    long_prefill_token_threshold: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Max total tokens across all running requests. \
                0 = max_position_embeddings * max_running_requests"
    )]
    max_total_tokens: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-request cap for reserving future decode tokens in the scheduler. \
                0 = read model's max_position_embeddings"
    )]
    decode_reservation_cap: usize,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::SetTrue,
        help = "Enable chunked prefill: interleave prefill with decode steps for lower TPOT"
    )]
    chunked_prefill: bool,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        help = "Disable chunked prefill and schedule waiting prefills before decode"
    )]
    no_chunked_prefill: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = CliTaskOverride::Auto,
        help = "Force model task detection: auto, classify, embedding, or generation"
    )]
    task: CliTaskOverride,

    #[arg(
        long,
        help = "API key for authentication (repeatable). Also reads PRELUDE_API_KEY env var"
    )]
    api_key: Vec<String>,

    #[arg(
        long,
        help = "Allowed CORS origin (repeatable). No CORS headers are emitted unless at least one origin is configured."
    )]
    cors_allow_origin: Vec<String>,

    #[arg(
        long,
        help = "Override dtype: f32 or bf16 (default: bf16 for CPU, auto for GPU)"
    )]
    dtype: Option<String>,

    #[arg(
        long,
        value_enum,
        help = "MoE backend policy: auto, cutlass, or sequential. \
                CUDA auto requires FlashInfer CUTLASS. \
                Defaults to PRELUDE_MOE_BACKEND or auto."
    )]
    moe_backend: Option<CliMoeBackend>,

    #[arg(
        long,
        default_value_t = prelude_core::config::DEFAULT_GPU_MEMORY_UTILIZATION,
        help = "Fraction of free GPU memory for KV cache (0.0-1.0). \
                Ignored when PRELUDE_PAGED_ATTN_BLOCKS is set explicitly."
    )]
    gpu_memory_utilization: f32,

    #[arg(
        long,
        default_value_t = true,
        help = "CUDA graph capture for decode steps (use --no-cuda-graph to disable)"
    )]
    cuda_graph: bool,

    #[arg(long, default_value = "auto", help = "Device: auto, cpu, cuda, cuda:N")]
    device: String,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliTaskOverride {
    Auto,
    Classify,
    Embedding,
    Generation,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliMoeBackend {
    Auto,
    Cutlass,
    #[value(alias = "ref", alias = "reference")]
    Sequential,
}

impl From<CliMoeBackend> for MoeBackendPolicy {
    fn from(value: CliMoeBackend) -> Self {
        match value {
            CliMoeBackend::Auto => MoeBackendPolicy::Auto,
            CliMoeBackend::Cutlass => MoeBackendPolicy::Cutlass,
            CliMoeBackend::Sequential => MoeBackendPolicy::Sequential,
        }
    }
}

impl From<CliTaskOverride> for TaskOverride {
    fn from(value: CliTaskOverride) -> Self {
        match value {
            CliTaskOverride::Auto => TaskOverride::Auto,
            CliTaskOverride::Classify => TaskOverride::Classify,
            CliTaskOverride::Embedding => TaskOverride::Embed,
            CliTaskOverride::Generation => TaskOverride::Generate,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Register device backends before anything else.
    prelude_cpu::register();
    #[cfg(feature = "cuda")]
    prelude_cuda::register();
    // The `amd` feature + `prelude-amd` dep are commented out in
    // Cargo.toml until t0-gpu's `tests/t0_original` submodule pin is
    // fixed upstream. Re-add `#[cfg(feature = "amd")] prelude_amd::register();`
    // when both are uncommented; leaving the gated call here triggers
    // `unexpected_cfgs` warnings on every build.

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "prelude_server=info,prelude_core=info,tower_http=info".into()),
        )
        .init();

    let cli = Cli::parse();

    let engine = build_engine(&cli).await?;
    let chat_template = build_chat_template(&cli).await?.map(Arc::new);

    let mut api_keys = cli.api_key.clone();
    if let Ok(env_key) = std::env::var("PRELUDE_API_KEY")
        && !env_key.is_empty()
    {
        api_keys.push(env_key);
    }

    let app = prelude_server::build_router_with_options(
        engine,
        chat_template,
        api_keys,
        prelude_server::RouterOptions {
            cors_allowed_origins: cli.cors_allow_origin.clone(),
        },
    )?;

    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!("server listening on http://{}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}

async fn build_engine(cli: &Cli) -> anyhow::Result<Arc<dyn InferenceEngine>> {
    let task_override: TaskOverride = cli.task.into();
    if cli.pseudo {
        info!(model = %cli.model, "using pseudo engine (mock)");
        return Ok(Arc::new(PseudoEngine::new(
            cli.model.clone(),
            cli.pseudo_latency_ms,
        )));
    }

    let mut engine_config =
        EngineConfig::from_env().map_err(|e| anyhow::anyhow!("invalid engine config: {e}"))?;
    engine_config.runtime.device = cli.device.clone();
    if let Some(ref dtype) = cli.dtype {
        engine_config.runtime.dtype = Some(dtype.clone());
    }
    if let Some(moe_backend) = cli.moe_backend {
        engine_config.runtime.moe_backend = moe_backend.into();
    }
    engine_config.cache.gpu_memory_utilization = cli.gpu_memory_utilization;
    engine_config.runtime.cuda_graph = cli.cuda_graph;
    // Activation profiler probes at this token count so the resulting
    // peak_activation_bytes matches the largest forward the scheduler
    // will dispatch. Keeping these in lockstep avoids KV cache being
    // under-allocated (when CLI < default) or activation under-estimated
    // (when CLI > default).
    engine_config.runtime.profile_tokens = cli.max_num_batched_tokens;
    info!(?engine_config, "engine config loaded");

    // Auto-detect: explicit --model-path wins. Otherwise, if --model points
    // at a real file or directory on disk, treat it as a local path so users
    // can pass `--model /path/to/checkpoint.gguf` directly without having to
    // know about a separate flag. Falls back to HF Hub.
    let local_path: Option<&str> = cli.model_path.as_deref().or_else(|| {
        std::path::Path::new(&cli.model)
            .exists()
            .then(|| cli.model.as_str())
    });
    let base_engine = if let Some(path) = local_path {
        info!(path = %path, "loading model from local path");
        Engine::from_local_path_with_task(path, &cli.model, task_override, engine_config)?
    } else {
        info!(repo = %cli.model, "loading model from HuggingFace Hub");
        Engine::from_hf_hub_with_task_async(&cli.model, task_override, engine_config).await?
    };

    // PRELUDE_NO_SCHEDULER=1 bypasses the scheduler and uses the base Engine directly.
    // Used by accuracy tests for single-request determinism.
    if std::env::var("PRELUDE_NO_SCHEDULER").as_deref() == Ok("1") {
        info!("scheduler disabled (PRELUDE_NO_SCHEDULER=1), using base engine directly");
        return Ok(Arc::new(base_engine));
    }

    // 0 = "auto-size from model's max_position_embeddings". Models advertise
    // their full context window (e.g. Qwen3-0.6B: 40960) — fall back to that
    // instead of an arbitrary CLI default that silently truncates long prompts.
    let ctx_len = base_engine.max_context_len();
    let decode_reservation_cap = match cli.decode_reservation_cap {
        0 => ctx_len,
        n => n,
    };
    let max_total_tokens = match cli.max_total_tokens {
        0 => ctx_len.saturating_mul(cli.max_running_requests.max(1)),
        n => n,
    };

    let scheduler_config = SchedulerConfig {
        max_batch_size: cli.max_batch_size,
        max_batch_wait_ms: cli.max_batch_wait_ms,
        max_running_requests: cli.max_running_requests,
        max_num_batched_tokens: cli.max_num_batched_tokens,
        long_prefill_token_threshold: cli.long_prefill_token_threshold,
        max_total_tokens,
        decode_reservation_cap,
        chunked_prefill: cli.chunked_prefill && !cli.no_chunked_prefill,
        ..SchedulerConfig::default()
    };
    info!(
        max_batch_size = scheduler_config.max_batch_size,
        max_batch_wait_ms = scheduler_config.max_batch_wait_ms,
        max_running = scheduler_config.max_running_requests,
        max_num_batched_tokens = scheduler_config.max_num_batched_tokens,
        max_total_tokens = scheduler_config.max_total_tokens,
        decode_reservation_cap = scheduler_config.decode_reservation_cap,
        "scheduler enabled"
    );
    let scheduled = ScheduledEngine::new(base_engine, scheduler_config);
    Ok(Arc::new(scheduled))
}

/// Try loading chat template from HF Hub, with fallback for GGUF repos.
/// If the repo has no tokenizer_config.json, try stripping -GGUF suffix to find the base model.
async fn load_chat_template_with_gguf_fallback(
    model: &str,
) -> anyhow::Result<Option<ModelChatTemplate>> {
    match ModelChatTemplate::from_hf_hub(model).await {
        Ok(Some(t)) => return Ok(Some(t)),
        Ok(None) | Err(_) => {}
    }
    // Fallback: strip -GGUF suffix
    if let Some(base) = model
        .strip_suffix("-GGUF")
        .or_else(|| model.strip_suffix("-gguf"))
    {
        info!(base_model = %base, "trying base model for chat template");
        if let Ok(Some(t)) = ModelChatTemplate::from_hf_hub(base).await {
            return Ok(Some(t));
        }
    }
    Ok(None)
}

async fn build_chat_template(cli: &Cli) -> anyhow::Result<Option<ModelChatTemplate>> {
    if cli.pseudo {
        return Ok(None);
    }

    // Try local path first, then HF Hub. For GGUF repos (no tokenizer_config.json),
    // fall back to base model repo by stripping -GGUF suffix.
    let template = if let Some(ref path) = cli.model_path {
        let local = ModelChatTemplate::from_local_path(path)?;
        if local.is_some() {
            local
        } else {
            load_chat_template_with_gguf_fallback(&cli.model).await?
        }
    } else {
        load_chat_template_with_gguf_fallback(&cli.model).await?
    };

    if template.is_some() {
        info!(model = %cli.model, "chat template loaded");
    } else {
        info!(model = %cli.model, "chat template unavailable");
    }

    Ok(template)
}
