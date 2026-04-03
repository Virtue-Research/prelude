//! Centralized runtime configuration for the inference engine.
//!
//! All `PRELUDE_*` environment variables are parsed **once** in
//! [`EngineConfig::from_env()`] and threaded through Engine / CacheManager /
//! scheduler construction. No module outside this file should call
//! `std::env::var("PRELUDE_*")` directly.

// ── Global runtime config (for static accessors in model code) ───────────

static GLOBAL_RUNTIME: std::sync::OnceLock<RuntimeConfig> = std::sync::OnceLock::new();
static GLOBAL_CACHE: std::sync::OnceLock<CacheConfig> = std::sync::OnceLock::new();

/// Store the engine config globally so that static helper functions
/// (e.g. `fused_kv_cache_write_enabled()`)
/// can access it without threading a reference through every model layer.
pub fn init_global_config(config: &EngineConfig) {
    let _ = GLOBAL_RUNTIME.set(config.runtime.clone());
    let _ = GLOBAL_CACHE.set(config.cache.clone());
}

/// Read a runtime toggle from the globally stored config.
pub fn global_runtime() -> Option<&'static RuntimeConfig> {
    GLOBAL_RUNTIME.get()
}

/// Read cache config from the globally stored config.
pub fn global_cache_config() -> Option<&'static CacheConfig> {
    GLOBAL_CACHE.get()
}

// ── Top-level config ─────────────────────────────────────────────────────

/// Root configuration for the inference engine.
///
/// Constructed once at startup via [`EngineConfig::from_env()`] and then
/// threaded through `Engine::new` → `CacheManager::new`, etc.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub cache: CacheConfig,
    pub sampling: SamplingDefaults,
    pub runtime: RuntimeConfig,
    pub adaptive: AdaptiveConfig,
}

impl EngineConfig {
    /// Parse all `PRELUDE_*` environment variables and return a validated
    /// config. Call this once at startup before constructing the engine.
    pub fn from_env() -> Result<Self, String> {
        let config = Self {
            cache: CacheConfig::from_env(),
            sampling: SamplingDefaults::from_env(),
            runtime: RuntimeConfig::from_env(),
            adaptive: AdaptiveConfig::from_env(),
        };
        config.validate()?;
        Ok(config)
    }

    /// Sanity-check the configuration for contradictions.
    pub fn validate(&self) -> Result<(), String> {
        if self.cache.paged_block_size == 0 {
            return Err("PRELUDE_PAGED_BLOCK_SIZE must be > 0".into());
        }
        if self.cache.prefix_block_size == 0 {
            return Err("PRELUDE_PREFIX_BLOCK_SIZE must be > 0".into());
        }
        if self.adaptive.arrival_alpha <= 0.0 || self.adaptive.arrival_alpha > 1.0 {
            return Err(format!(
                "adaptive arrival_alpha must be in (0, 1], got {}",
                self.adaptive.arrival_alpha
            ));
        }
        if self.adaptive.gpu_alpha <= 0.0 || self.adaptive.gpu_alpha > 1.0 {
            return Err(format!(
                "adaptive gpu_alpha must be in (0, 1], got {}",
                self.adaptive.gpu_alpha
            ));
        }
        Ok(())
    }
}

// ── Cache config ─────────────────────────────────────────────────────────

/// KV cache pool and prefix cache settings.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Tokens per block in paged KV cache.
    pub paged_block_size: usize,
    /// Explicit number of paged attention blocks (0 = auto from gpu_memory_utilization).
    pub paged_attn_blocks: usize,
    /// Fraction of free GPU memory to use for KV cache (vLLM-style).
    /// Only used when paged_attn_blocks == 0. Default 0.4 (vLLM uses 0.9).
    pub gpu_memory_utilization: f32,
    /// Max cached prefix blocks (0 = disabled).
    pub prefix_cache_blocks: usize,
    /// Tokens per prefix cache block.
    pub prefix_block_size: usize,
    /// Max concurrent DeltaNet state slots (0 = disabled).
    pub deltanet_pool_slots: u32,
}

impl CacheConfig {
    fn from_env() -> Self {
        Self {
            paged_block_size: parse_env_usize(
                "PRELUDE_PAGED_BLOCK_SIZE",
                128, // adjusted at runtime for FA4/FlashInfer by cache manager
            ),
            paged_attn_blocks: parse_env_usize("PRELUDE_PAGED_ATTN_BLOCKS", 0),
            gpu_memory_utilization: 0.4,
            prefix_cache_blocks: parse_env_usize("PRELUDE_PREFIX_CACHE_BLOCKS", 0),
            prefix_block_size: parse_env_usize("PRELUDE_PREFIX_BLOCK_SIZE", 64),
            deltanet_pool_slots: parse_env_u32("PRELUDE_DELTANET_POOL_SLOTS", 8),
        }
    }
}

// ── Sampling defaults ────────────────────────────────────────────────────

/// Default sampling parameters applied when the API request does not specify them.
#[derive(Debug, Clone)]
pub struct SamplingDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub max_new_tokens: u32,
}

impl SamplingDefaults {
    fn from_env() -> Self {
        Self {
            temperature: parse_env_f32("PRELUDE_DEFAULT_TEMPERATURE", 0.7),
            top_p: parse_env_f32("PRELUDE_DEFAULT_TOP_P", 1.0),
            max_new_tokens: parse_env_u32("PRELUDE_DEFAULT_MAX_TOKENS", 4096),
        }
    }
}

// ── Runtime toggles ──────────────────────────────────────────────────────

/// Runtime feature toggles and device selection.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Device selection string: "auto", "cpu", "cuda", "cuda:N".
    pub device: String,
    /// Enable CUDA sync timing for profiling.
    pub sync_timing: bool,
    /// Force variable-length prefill path even when all sequences are same length.
    pub force_varlen_prefill: bool,
    /// Enable fused K-Norm + RoPE + KV cache write kernel.
    pub fused_kv_cache_write: bool,
    /// CPU IDs for NUMA-aware thread binding.
    pub cpu_thread_bind: Option<String>,
    /// Override dtype selection: "f32", "bf16".
    /// When None, auto-selects based on device (GPU: BF16 if supported, CPU: BF16).
    pub dtype: Option<String>,
    /// Enable CUDA graph capture for decode (Q=1) steps.
    pub cuda_graph: bool,
    /// Maximum batch size for CUDA graph capture (graphs captured for 1..=max_bs powers of 2).
    pub cuda_graph_max_bs: usize,
}

impl RuntimeConfig {
    fn from_env() -> Self {
        Self {
            device: "auto".to_string(),
            sync_timing: parse_env_bool("PRELUDE_SYNC_TIMING"),
            force_varlen_prefill: parse_env_bool("PRELUDE_FORCE_VARLEN_PREFILL"),
            fused_kv_cache_write: parse_env_bool_eq1("PRELUDE_FUSED_KV_CACHE_WRITE"),
            cpu_thread_bind: std::env::var("SGLANG_CPU_OMP_THREADS_BIND")
                .ok()
                .filter(|s| !s.is_empty()),
            dtype: None,
            cuda_graph: true,
            cuda_graph_max_bs: parse_env_usize("PRELUDE_CUDA_GRAPH_MAX_BS", 32),
        }
    }
}

// ── Adaptive batch config ────────────────────────────────────────────────

/// EWMA parameters for the adaptive batch scheduler.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// EWMA smoothing factor for arrival rate (higher = more responsive).
    pub arrival_alpha: f64,
    /// EWMA smoothing factor for GPU time (lower = more stable).
    pub gpu_alpha: f64,
    /// Initial arrival rate assumption (req/s) for cold start.
    pub initial_lambda: f64,
    /// Maximum instantaneous rate before clamping.
    pub max_instant_rate: f64,
}

impl AdaptiveConfig {
    fn from_env() -> Self {
        Self {
            arrival_alpha: parse_env_f64("PRELUDE_ADAPTIVE_ARRIVAL_ALPHA", 0.5),
            gpu_alpha: parse_env_f64("PRELUDE_ADAPTIVE_GPU_ALPHA", 0.4),
            initial_lambda: parse_env_f64("PRELUDE_ADAPTIVE_INITIAL_LAMBDA", 1000.0),
            max_instant_rate: parse_env_f64("PRELUDE_ADAPTIVE_MAX_RATE", 10000.0),
        }
    }
}

// ── Parsing helpers ──────────────────────────────────────────────────────

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Parse a boolean env var: matches "1", "true", "yes", "on" (case-insensitive).
fn parse_env_bool(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Parse a boolean env var: matches only "1" exactly.
fn parse_env_bool_eq1(name: &str) -> bool {
    std::env::var(name).map_or(false, |v| v == "1")
}
