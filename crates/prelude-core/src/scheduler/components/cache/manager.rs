use std::sync::{Arc, Mutex};

use crate::tensor::{DType, Device, Tensor};
use tracing::info;

use crate::cache::deltanet_pool::{DeltaNetPool, DeltaNetPoolConfig};
use crate::cache::prefix_cache::PrefixKvCache;
use crate::config::CacheConfig;
use crate::engine::{CommonModelConfig, EngineError, PagedKvPool, RuntimeCaps};

/// Unified owner of all cache state: prefix cache, paged KV pool, block manager,
/// and DeltaNet state pool. Extracted from Engine to separate cache concerns
/// from model execution.
pub struct CacheManager {
    pub(crate) prefix_cache: Option<Mutex<PrefixKvCache>>,
    pub paged_pool: Option<PagedKvPool>,
    pub(crate) block_manager: Option<Arc<Mutex<crate::cache::block_manager::BlockManager>>>,
    pub deltanet_pool: Option<Mutex<DeltaNetPool>>,
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
        kv_sharing: &[Option<usize>],
        peak_activation_bytes: usize,
    ) -> Result<Self, EngineError> {
        let prefix_cache = if runtime_caps.supports_prefix_cache {
            Self::init_prefix_cache(device, model_config.num_hidden_layers, cache_config)
        } else {
            None
        };

        let (paged_pool, block_manager) = if runtime_caps.supports_paged_attn {
            Self::init_paged_pool(
                model_config,
                dtype,
                device,
                cache_config,
                kv_sharing,
                peak_activation_bytes,
            )?
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
            paged_pool,
            block_manager,
            deltanet_pool,
        })
    }

    /// Empty CacheManager for paths where no cache is needed (e.g. GGUF CPU).
    pub(crate) fn none() -> Self {
        Self {
            prefix_cache: None,
            paged_pool: None,
            block_manager: None,
            deltanet_pool: None,
        }
    }

    /// Get a shared reference to the block manager (for giving to the Scheduler).
    pub fn block_manager_arc(
        &self,
    ) -> Option<Arc<Mutex<crate::cache::block_manager::BlockManager>>> {
        self.block_manager.clone()
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
        let is_flash = device.is_cuda();
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
    /// Returns `(None, None)` for non-CUDA devices.
    fn init_paged_pool(
        config: &CommonModelConfig,
        dtype: DType,
        device: &Device,
        cache_config: &CacheConfig,
        kv_sharing: &[Option<usize>],
        peak_activation_bytes: usize,
    ) -> Result<
        (
            Option<PagedKvPool>,
            Option<Arc<Mutex<crate::cache::block_manager::BlockManager>>>,
        ),
        EngineError,
    > {
        if !device.is_cuda() {
            return Ok((None, None));
        }
        let mut paged_block_size = cache_config.paged_block_size;

        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        // Auto-adjust block size for attention backend compatibility.
        if device.is_cuda() && std::env::var("PRELUDE_PAGED_BLOCK_SIZE").is_err() {
            let ops = crate::ops::select_ops(device);
            let hint = ops.paged_block_size_hint(head_dim);
            if paged_block_size != hint {
                let old = paged_block_size;
                paged_block_size = hint;
                info!(
                    old_block_size = old,
                    new_block_size = paged_block_size,
                    head_dim,
                    "auto-adjusted paged_block_size per device backend hint"
                );
            }
        }
        let num_layers = config.num_hidden_layers;
        // KV sharing: count only layers that need independent cache allocations.
        let num_shared = kv_sharing.iter().filter(|s| s.is_some()).count();
        let num_physical_kv_layers = num_layers - num_shared;

        // Auto-size: when paged_attn_blocks == 0, use the vLLM formula:
        //
        //   available_kv = total_gpu * utilization - weights - peak_activation
        //   num_blocks = available_kv / bytes_per_block
        //
        // Before this change, we used `free * utilization` which didn't
        // account for activation memory. For large-vocab MoE models
        // (Qwen3.5-35B-A3B: vocab=248K, lm_head output ~8 GB at 8192
        // tokens) the 10% reserve was insufficient → OOM during first
        // forward pass.
        //
        // Override with PRELUDE_PAGED_ATTN_BLOCKS=N for manual control.
        let paged_blocks = if cache_config.paged_attn_blocks > 0 {
            cache_config.paged_attn_blocks
        } else {
            let bytes_per_block_per_layer =
                { 2 * num_kv_heads * head_dim * paged_block_size * dtype.size_in_bytes() };
            let total_bytes_per_block = bytes_per_block_per_layer * num_physical_kv_layers;

            let ops = crate::ops::select_ops(device);
            let total_bytes = ops.gpu_total_memory().unwrap_or(0);
            let free_bytes = ops.gpu_free_memory().unwrap_or(0);
            let utilization = cache_config.gpu_memory_utilization;

            // vLLM formula:
            //   requested = total * utilization
            //   non_kv = weights_memory + peak_activation
            //   available = requested - non_kv
            let weights_bytes = total_bytes.saturating_sub(free_bytes);
            let requested = (total_bytes as f64 * utilization as f64) as usize;
            let non_kv = weights_bytes + peak_activation_bytes;
            let available_for_kv = requested.saturating_sub(non_kv);

            let auto_blocks = if total_bytes_per_block > 0 {
                (available_for_kv / total_bytes_per_block).max(16)
            } else {
                256
            };
            info!(
                auto_blocks,
                total_gpu_mb = total_bytes / (1024 * 1024),
                free_gpu_mb = free_bytes / (1024 * 1024),
                weights_mb = weights_bytes / (1024 * 1024),
                peak_activation_mb = peak_activation_bytes / (1024 * 1024),
                available_for_kv_mb = available_for_kv / (1024 * 1024),
                gpu_memory_utilization = utilization,
                bytes_per_block = total_bytes_per_block,
                num_physical_kv_layers,
                "auto-sized paged KV cache (vLLM formula)"
            );
            auto_blocks
        };

        let tensor_err =
            |e: crate::tensor::Error| EngineError::Internal(format!("tensor error: {e}"));

        // Allocate cache tensors. Shared layers alias their source layer's tensor.
        let mut key_caches_flash: Vec<Tensor> = Vec::with_capacity(num_layers);
        let mut value_caches_flash: Vec<Tensor> = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let source = if i < kv_sharing.len() {
                kv_sharing[i]
            } else {
                None
            };
            if let Some(src) = source {
                // Shared layer: alias source layer's cache (Tensor is Arc — clone is pointer copy).
                key_caches_flash.push(key_caches_flash[src].clone());
                value_caches_flash.push(value_caches_flash[src].clone());
            } else {
                // Independent layer: allocate new cache tensors.
                key_caches_flash.push(
                    Tensor::zeros(
                        (paged_blocks, paged_block_size, num_kv_heads, head_dim),
                        dtype,
                        device,
                    )
                    .map_err(tensor_err)?,
                );
                value_caches_flash.push(
                    Tensor::zeros(
                        (paged_blocks, paged_block_size, num_kv_heads, head_dim),
                        dtype,
                        device,
                    )
                    .map_err(tensor_err)?,
                );
            }
        }

        if num_shared > 0 {
            info!(
                num_shared,
                num_physical_kv_layers, "KV cache sharing enabled"
            );
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
            key_caches_flash,
            value_caches_flash,
            block_size: paged_block_size,
        };
        let bm = crate::cache::block_manager::BlockManager::new(paged_blocks, paged_block_size);
        Ok((Some(pool), Some(Arc::new(Mutex::new(bm)))))
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
