#[global_allocator]
static GLOBAL: bc_mimalloc::MiMalloc = bc_mimalloc::MiMalloc;

use std::net::SocketAddr;
use std::sync::Arc;

use clap::{Parser, ValueEnum};
use prelude_core::{
    Engine, EngineConfig, InferenceEngine, PseudoEngine, ScheduledEngine, SchedulerConfig,
    TaskOverride,
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
        default_value_t = 8,
        help = "Max concurrent running requests in scheduler"
    )]
    max_running_requests: usize,

    #[arg(
        long,
        default_value_t = 4096,
        help = "Max prefill tokens per scheduling step"
    )]
    max_prefill_tokens: usize,

    #[arg(
        long,
        default_value_t = 32768,
        help = "Max total tokens across all running requests"
    )]
    max_total_tokens: usize,

    #[arg(
        long,
        default_value_t = 4096,
        help = "Per-request cap for reserving future decode tokens in the scheduler"
    )]
    decode_reservation_cap: usize,

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
        default_value_t = 0.4,
        help = "Fraction of free GPU memory for KV cache (0.0-1.0, default 0.4). \
                Ignored when PRELUDE_PAGED_ATTN_BLOCKS is set explicitly."
    )]
    gpu_memory_utilization: f32,

    #[arg(long, default_value_t = true, help = "CUDA graph capture for decode steps (use --no-cuda-graph to disable)")]
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

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "prelude_server=info,prelude_core=info,tower_http=info".into()
            }),
        )
        .init();

    let cli = Cli::parse();

    let engine = build_engine(&cli)?;
    let chat_template = build_chat_template(&cli)?.map(Arc::new);

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

fn build_engine(cli: &Cli) -> anyhow::Result<Arc<dyn InferenceEngine>> {
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
    engine_config.cache.gpu_memory_utilization = cli.gpu_memory_utilization;
    engine_config.runtime.cuda_graph = cli.cuda_graph;
    info!(?engine_config, "engine config loaded");

    let base_engine = if let Some(ref path) = cli.model_path {
        info!(path = %path, "loading model from local path");
        Engine::from_local_path_with_task(path, &cli.model, task_override, engine_config)?
    } else {
        info!(repo = %cli.model, "loading model from HuggingFace Hub");
        Engine::from_hf_hub_with_task(&cli.model, task_override, engine_config)?
    };

    // PRELUDE_NO_SCHEDULER=1 bypasses the scheduler and uses the base Engine directly.
    // Used by accuracy tests for single-request determinism.
    if std::env::var("PRELUDE_NO_SCHEDULER").as_deref() == Ok("1") {
        info!("scheduler disabled (PRELUDE_NO_SCHEDULER=1), using base engine directly");
        return Ok(Arc::new(base_engine));
    }

    let scheduler_config = SchedulerConfig {
        max_batch_size: cli.max_batch_size,
        max_batch_wait_ms: cli.max_batch_wait_ms,
        max_running_requests: cli.max_running_requests,
        max_prefill_tokens: cli.max_prefill_tokens,
        max_total_tokens: cli.max_total_tokens,
        decode_reservation_cap: cli.decode_reservation_cap,
        ..SchedulerConfig::default()
    };
    info!(
        max_batch_size = scheduler_config.max_batch_size,
        max_batch_wait_ms = scheduler_config.max_batch_wait_ms,
        max_running = scheduler_config.max_running_requests,
        max_prefill_tokens = scheduler_config.max_prefill_tokens,
        max_total_tokens = scheduler_config.max_total_tokens,
        decode_reservation_cap = scheduler_config.decode_reservation_cap,
        "scheduler enabled"
    );
    let scheduled = ScheduledEngine::new(base_engine, scheduler_config);
    Ok(Arc::new(scheduled))
}

/// Try loading chat template from HF Hub, with fallback for GGUF repos.
/// If the repo has no tokenizer_config.json, try stripping -GGUF suffix to find the base model.
fn load_chat_template_with_gguf_fallback(
    model: &str,
) -> anyhow::Result<Option<ModelChatTemplate>> {
    match ModelChatTemplate::from_hf_hub(model) {
        Ok(Some(t)) => return Ok(Some(t)),
        Ok(None) | Err(_) => {}
    }
    // Fallback: strip -GGUF suffix
    if let Some(base) = model
        .strip_suffix("-GGUF")
        .or_else(|| model.strip_suffix("-gguf"))
    {
        info!(base_model = %base, "trying base model for chat template");
        if let Ok(Some(t)) = ModelChatTemplate::from_hf_hub(base) {
            return Ok(Some(t));
        }
    }
    Ok(None)
}

fn build_chat_template(cli: &Cli) -> anyhow::Result<Option<ModelChatTemplate>> {
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
            load_chat_template_with_gguf_fallback(&cli.model)?
        }
    } else {
        load_chat_template_with_gguf_fallback(&cli.model)?
    };

    if template.is_some() {
        info!(model = %cli.model, "chat template loaded");
    } else {
        info!(model = %cli.model, "chat template unavailable");
    }

    Ok(template)
}
