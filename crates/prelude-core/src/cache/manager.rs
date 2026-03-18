use std::sync::Mutex;

#[cfg(feature = "cuda")]
use candle_core::Tensor;
use candle_core::{DType, Device};
use tracing::info;

use crate::cache::deltanet_pool::{DeltaNetPool, DeltaNetPoolConfig};
use crate::cache::prefix_cache::PrefixKvCache;
use crate::config::CacheConfig;
#[cfg(feature = "cuda")]
use crate::engine::candle_err;
use crate::engine::{CommonModelConfig, EngineError, RuntimeCaps};

#[cfg(feature = "cuda")]
use crate::engine::PagedKvPool;

/// Unified owner of all cache state: prefix cache, paged KV pool, block manager,
/// and DeltaNet state pool. Extracted from Engine to separate cache concerns
/// from model execution.
pub struct CacheManager {
    pub(crate) prefix_cache: Option<Mutex<PrefixKvCache>>,
    #[cfg(feature = "cuda")]
    pub(crate) paged_pool: Option<PagedKvPool>,
    #[cfg(feature = "cuda")]
    pub(crate) block_manager: Option<Mutex<crate::cache::block_manager::BlockManager>>,
    pub(crate) deltanet_pool: Option<Mutex<DeltaNetPool>>,
}

impl CacheManager {
    /// Build a CacheManager with all caches initialized based on runtime caps
    /// and the centralized [`CacheConfig`].
    pub(crate) fn new(
        model_config: &CommonModelConfig,
        deltanet_config: Option<&DeltaNetPoolConfig>,
        dtype: DType,
        device: &Device,
        runtime_caps: &RuntimeCaps,
        cache_config: &CacheConfig,
    ) -> Result<Self, EngineError> {
        let prefix_cache = if runtime_caps.supports_prefix_cache {
            Self::init_prefix_cache(device, model_config.num_hidden_layers, cache_config)
        } else {
            None
        };

        #[cfg(feature = "cuda")]
        let (paged_pool, block_manager) = if runtime_caps.supports_paged_attn {
            Self::init_paged_pool(model_config, dtype, device, cache_config)?
        } else {
            (None, None)
        };

        let deltanet_pool = if runtime_caps.supports_deltanet {
            Self::init_deltanet_pool(deltanet_config, dtype, device, cache_config)?
        } else {
            None
        };

        Ok(Self {
            prefix_cache,
            #[cfg(feature = "cuda")]
            paged_pool,
            #[cfg(feature = "cuda")]
            block_manager,
            deltanet_pool,
        })
    }

    /// Empty CacheManager for paths where no cache is needed (e.g. GGUF CPU).
    pub(crate) fn none() -> Self {
        Self {
            prefix_cache: None,
            #[cfg(feature = "cuda")]
            paged_pool: None,
            #[cfg(feature = "cuda")]
            block_manager: None,
            deltanet_pool: None,
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────

    #[allow(dead_code)]
    pub(crate) fn prefix_cache(&self) -> Option<&Mutex<PrefixKvCache>> {
        self.prefix_cache.as_ref()
    }

    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pub(crate) fn paged_pool(&self) -> Option<&PagedKvPool> {
        self.paged_pool.as_ref()
    }

    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pub(crate) fn block_manager(&self) -> Option<&Mutex<crate::cache::block_manager::BlockManager>> {
        self.block_manager.as_ref()
    }

    #[allow(dead_code)]
    pub(crate) fn deltanet_pool(&self) -> Option<&Mutex<DeltaNetPool>> {
        self.deltanet_pool.as_ref()
    }

    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pub(crate) fn has_paged_attn(&self) -> bool {
        self.paged_pool.is_some() && self.block_manager.is_some()
    }

    // ── Init helpers (read from CacheConfig, not env vars) ───────────────

    /// Create a prefix cache if enabled by config.
    fn init_prefix_cache(
        device: &Device,
        num_layers: usize,
        cache_config: &CacheConfig,
    ) -> Option<Mutex<PrefixKvCache>> {
        let max_blocks = cache_config.prefix_cache_blocks;
        if max_blocks == 0 {
            return None;
        }
        let block_size = cache_config.prefix_block_size;
        // flash layout: [B, L, H, D] → concat dim 1; standard: [B, H, L, D] → concat dim 2
        let is_flash =
            device.is_cuda() && (cfg!(feature = "flash-attn") || cfg!(feature = "flash-attn-v3"));
        let concat_dim = if is_flash { 1 } else { 2 };
        info!(
            max_blocks = max_blocks,
            block_size = block_size,
            concat_dim = concat_dim,
            num_layers = num_layers,
            "prefix cache enabled"
        );
        Some(Mutex::new(PrefixKvCache::new(
            block_size, concat_dim, num_layers, max_blocks,
        )))
    }

    /// Initialize paged KV cache pool from config.
    /// When `paged_attn_blocks == 0` (default), auto-sizes based on available GPU memory
    /// (similar to vLLM's gpu_memory_utilization approach).
    #[cfg(feature = "cuda")]
    fn init_paged_pool(
        config: &CommonModelConfig,
        dtype: DType,
        device: &Device,
        cache_config: &CacheConfig,
    ) -> Result<
        (
            Option<PagedKvPool>,
            Option<Mutex<crate::cache::block_manager::BlockManager>>,
        ),
        EngineError,
    > {
        if !device.is_cuda() {
            return Ok((None, None));
        }
        let mut paged_block_size = cache_config.paged_block_size;

        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        // flash-attn-v3 requires page_block_size % kBlockN == 0
        // kBlockN = 64 for head_dim=256, 128 otherwise
        #[cfg(feature = "flash-attn-v3")]
        {
            let min_block = if head_dim == 256 { 64 } else { 128 };
            if paged_block_size % min_block != 0 {
                let old = paged_block_size;
                paged_block_size = min_block;
                info!(
                    old_block_size = old,
                    new_block_size = paged_block_size,
                    head_dim,
                    "auto-adjusted paged_block_size for flash-attn-v3 kernel compatibility"
                );
            }
        }
        let num_layers = config.num_hidden_layers;
        let x = 16 / dtype.size_in_bytes(); // vectorization factor (8 for BF16, 16 for F16)

        // Auto-size: when paged_attn_blocks == 0, use gpu_memory_utilization fraction
        // of free GPU memory (vLLM-style). Override with PRELUDE_PAGED_ATTN_BLOCKS=N.
        let paged_blocks = if cache_config.paged_attn_blocks > 0 {
            cache_config.paged_attn_blocks
        } else {
            let bytes_per_block_per_layer = {
                let v1 = 2 * num_kv_heads * head_dim * paged_block_size * dtype.size_in_bytes();
                #[cfg(feature = "flash-attn-v3")]
                let flash = 2 * num_kv_heads * head_dim * paged_block_size * dtype.size_in_bytes();
                #[cfg(not(feature = "flash-attn-v3"))]
                let flash = 0;
                v1 + flash
            };
            let total_bytes_per_block = bytes_per_block_per_layer * num_layers;

            let free_bytes = cuda_free_memory().unwrap_or(0);
            let utilization = cache_config.gpu_memory_utilization;
            let usable = (free_bytes as f64 * utilization as f64) as usize;
            let auto_blocks = if total_bytes_per_block > 0 {
                (usable / total_bytes_per_block).max(16)
            } else {
                256
            };
            info!(
                auto_blocks,
                free_gpu_mb = free_bytes / (1024 * 1024),
                gpu_memory_utilization = utilization,
                bytes_per_block = total_bytes_per_block,
                "auto-sized paged KV cache from free GPU memory"
            );
            auto_blocks
        };

        let mut key_caches = Vec::with_capacity(num_layers);
        let mut value_caches = Vec::with_capacity(num_layers);
        #[cfg(feature = "flash-attn-v3")]
        let mut key_caches_flash = Vec::with_capacity(num_layers);
        #[cfg(feature = "flash-attn-v3")]
        let mut value_caches_flash = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            // key_cache: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
            key_caches.push(
                Tensor::zeros(
                    (
                        paged_blocks,
                        num_kv_heads,
                        head_dim / x,
                        paged_block_size,
                        x,
                    ),
                    dtype,
                    device,
                )
                .map_err(candle_err)?,
            );
            // value_cache: [num_blocks, num_kv_heads, head_dim, block_size]
            value_caches.push(
                Tensor::zeros(
                    (paged_blocks, num_kv_heads, head_dim, paged_block_size),
                    dtype,
                    device,
                )
                .map_err(candle_err)?,
            );

            #[cfg(feature = "flash-attn-v3")]
            {
                // flash-friendly paged layout: [num_blocks, block_size, num_kv_heads, head_dim]
                key_caches_flash.push(
                    Tensor::zeros(
                        (paged_blocks, paged_block_size, num_kv_heads, head_dim),
                        dtype,
                        device,
                    )
                    .map_err(candle_err)?,
                );
                value_caches_flash.push(
                    Tensor::zeros(
                        (paged_blocks, paged_block_size, num_kv_heads, head_dim),
                        dtype,
                        device,
                    )
                    .map_err(candle_err)?,
                );
            }
        }

        info!(
            num_blocks = paged_blocks,
            block_size = paged_block_size,
            num_layers,
            num_kv_heads,
            head_dim,
            "paged KV cache pool allocated"
        );

        let pool = PagedKvPool {
            key_caches,
            value_caches,
            #[cfg(feature = "flash-attn-v3")]
            key_caches_flash,
            #[cfg(feature = "flash-attn-v3")]
            value_caches_flash,
            block_size: paged_block_size,
        };
        let bm = crate::cache::block_manager::BlockManager::new(paged_blocks, paged_block_size);
        Ok((Some(pool), Some(Mutex::new(bm))))
    }

    /// Initialize a DeltaNet state pool for hybrid models (Qwen3.5, Qwen3-Next).
    /// Returns `None` for non-hybrid models.
    fn init_deltanet_pool(
        deltanet_config: Option<&DeltaNetPoolConfig>,
        dtype: DType,
        device: &Device,
        cache_config: &CacheConfig,
    ) -> Result<Option<Mutex<DeltaNetPool>>, EngineError> {
        let dn_cfg = deltanet_config.ok_or_else(|| {
            EngineError::Internal("missing DeltaNet config for hybrid model".into())
        })?;

        let max_slots = cache_config.deltanet_pool_slots;

        if max_slots == 0 {
            return Ok(None);
        }

        let pool = DeltaNetPool::new(dn_cfg, max_slots, dtype, device)
            .map_err(|e| EngineError::Internal(format!("failed to init DeltaNet pool: {e}")))?;

        tracing::info!(
            deltanet_layers = dn_cfg.num_deltanet_layers,
            max_slots,
            "DeltaNet state pool initialized"
        );

        Ok(Some(Mutex::new(pool)))
    }
}

/// Query free GPU memory via cudaMemGetInfo.
#[cfg(feature = "cuda")]
fn cuda_free_memory() -> Option<usize> {
    unsafe extern "C" {
        fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    }
    let mut free = 0usize;
    let mut total = 0usize;
    let ret = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    if ret == 0 {
        Some(free)
    } else {
        None
    }
}
