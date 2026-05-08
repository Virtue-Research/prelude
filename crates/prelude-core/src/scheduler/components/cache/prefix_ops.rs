use crate::cache::deltanet_pool::DeltaNetPrefixState;
use crate::cache::manager::CacheManager;
use crate::engine::EngineError;
use crate::engine::PreparedGenerateRequest;
use tracing::debug;

fn tensor_err(e: crate::tensor::Error) -> EngineError {
    EngineError::Internal(format!("tensor error: {e}"))
}

impl CacheManager {
    pub(crate) fn allocate_paged_block(&self) -> Result<Option<u32>, EngineError> {
        let Some(ref bm_mutex) = self.block_manager else {
            return Ok(None);
        };
        let mut bm = bm_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
        Ok(bm.allocate())
    }

    /// Retain paged KV blocks that a sequence is about to reuse from prefix
    /// cache. The cache itself already owns one reference; active sequences
    /// need their own reference so normal request finalization can free them.
    pub(crate) fn retain_paged_blocks(&self, block_ids: &[u32]) -> Result<(), EngineError> {
        if block_ids.is_empty() {
            return Ok(());
        }
        let Some(ref bm_mutex) = self.block_manager else {
            return Ok(());
        };
        let mut bm = bm_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
        bm.increment_refs(block_ids);
        Ok(())
    }

    pub(crate) fn release_paged_blocks(&self, block_ids: &[u32]) -> Result<(), EngineError> {
        if block_ids.is_empty() {
            return Ok(());
        }
        let Some(ref bm_mutex) = self.block_manager else {
            return Ok(());
        };
        let mut bm = bm_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
        bm.decrement_refs(block_ids);
        Ok(())
    }

    pub(crate) fn copy_paged_kv_block(&self, src: u32, dst: u32) -> Result<(), EngineError> {
        if src == dst {
            return Ok(());
        }
        let Some(pool) = self.paged_pool.as_ref() else {
            return Ok(());
        };
        for cache in pool
            .active_key_caches()
            .iter()
            .chain(pool.active_value_caches().iter())
        {
            copy_tensor_block(cache, src as usize, dst as usize)?;
        }
        Ok(())
    }

    /// Match prefix cache for paged-attention runs and return only block IDs.
    /// This avoids assembling per-layer KV tensors when the caller can consume
    /// paged blocks directly.
    pub(crate) fn try_prefix_cache_match_paged_only(
        &self,
        tokens: &[u32],
    ) -> Result<(usize, Vec<u32>), EngineError> {
        let Some(ref pc_mutex) = self.prefix_cache else {
            return Ok((0, vec![]));
        };
        let mut pc = pc_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("prefix cache lock poisoned: {e}")))?;
        let (cached_len, paged_ids) = pc.match_paged_blocks_only(tokens).map_err(tensor_err)?;
        if cached_len > 0 {
            debug!(
                cached_tokens = cached_len,
                suffix_tokens = tokens.len() - cached_len,
                paged_blocks = paged_ids.len(),
                "prefix cache match (paged blocks only)"
            );
        }
        let evicted = pc.take_evicted_paged_blocks();
        if !evicted.is_empty()
            && let Some(ref bm_mutex) = self.block_manager
        {
            let mut bm = bm_mutex
                .lock()
                .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
            bm.decrement_refs(&evicted);
        }
        Ok((cached_len, paged_ids))
    }

    /// Match prefix cache for hybrid paged-attention runs. A hit is only valid
    /// when both paged KV blocks and the corresponding DeltaNet state snapshot
    /// are present at the same prefix boundary.
    pub(crate) fn try_prefix_cache_match_paged_with_deltanet_state(
        &self,
        tokens: &[u32],
    ) -> Result<(usize, Vec<u32>, Option<DeltaNetPrefixState>), EngineError> {
        let Some(ref pc_mutex) = self.prefix_cache else {
            return Ok((0, vec![], None));
        };
        let mut pc = pc_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("prefix cache lock poisoned: {e}")))?;
        let (cached_len, paged_ids, state) = pc
            .match_paged_blocks_with_deltanet_state(tokens)
            .map_err(tensor_err)?;
        if cached_len > 0 {
            debug!(
                cached_tokens = cached_len,
                suffix_tokens = tokens.len() - cached_len,
                paged_blocks = paged_ids.len(),
                "prefix cache match (paged blocks + deltanet state)"
            );
        }
        let evicted = pc.take_evicted_paged_blocks();
        if !evicted.is_empty()
            && let Some(ref bm_mutex) = self.block_manager
        {
            let mut bm = bm_mutex
                .lock()
                .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
            bm.decrement_refs(&evicted);
        }
        Ok((cached_len, paged_ids, state))
    }

    /// Insert only paged block IDs into prefix cache (no KV tensor extraction).
    pub(crate) fn try_prefix_cache_insert_paged_only(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        paged_block_size: usize,
    ) -> Result<(), EngineError> {
        let Some(ref pc_mutex) = self.prefix_cache else {
            return Ok(());
        };
        let mut pc = pc_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("prefix cache lock poisoned: {e}")))?;
        let stored_paged_ids = pc.insert_paged_blocks_only(tokens, paged_block_size, block_table);
        if !stored_paged_ids.is_empty()
            && let Some(ref bm_mutex) = self.block_manager
        {
            let mut bm = bm_mutex
                .lock()
                .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
            bm.increment_refs(&stored_paged_ids);
        }
        let evicted = pc.take_evicted_paged_blocks();
        if !evicted.is_empty()
            && let Some(ref bm_mutex) = self.block_manager
        {
            let mut bm = bm_mutex
                .lock()
                .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
            bm.decrement_refs(&evicted);
        }
        debug!(
            prompt_tokens = tokens.len(),
            cached_blocks = pc.cached_blocks(),
            stored_paged_blocks = stored_paged_ids.len(),
            "prefix cache insert (paged blocks only)"
        );
        Ok(())
    }

    /// Insert paged block IDs plus a DeltaNet state snapshot for a hybrid
    /// prefix boundary.
    pub(crate) fn try_prefix_cache_insert_paged_with_deltanet_state(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        paged_block_size: usize,
        deltanet_state: DeltaNetPrefixState,
    ) -> Result<(), EngineError> {
        let Some(ref pc_mutex) = self.prefix_cache else {
            return Ok(());
        };
        let mut pc = pc_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("prefix cache lock poisoned: {e}")))?;
        let stored_paged_ids = pc.insert_paged_blocks_with_deltanet_state(
            tokens,
            paged_block_size,
            block_table,
            deltanet_state,
        );
        if !stored_paged_ids.is_empty()
            && let Some(ref bm_mutex) = self.block_manager
        {
            let mut bm = bm_mutex
                .lock()
                .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
            bm.increment_refs(&stored_paged_ids);
        }
        let evicted = pc.take_evicted_paged_blocks();
        if !evicted.is_empty()
            && let Some(ref bm_mutex) = self.block_manager
        {
            let mut bm = bm_mutex
                .lock()
                .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
            bm.decrement_refs(&evicted);
        }
        debug!(
            prompt_tokens = tokens.len(),
            cached_blocks = pc.cached_blocks(),
            stored_paged_blocks = stored_paged_ids.len(),
            "prefix cache insert (paged blocks + deltanet state)"
        );
        Ok(())
    }

    /// Find the longest common prefix among all requests in a batch.
    /// Returns the common prefix tokens.
    pub(crate) fn find_common_prefix<'a>(batch: &'a [PreparedGenerateRequest]) -> &'a [u32] {
        if batch.is_empty() {
            return &[];
        }
        if batch.len() == 1 {
            return &batch[0].prompt_tokens;
        }

        let first = &batch[0].prompt_tokens;
        let min_len = batch
            .iter()
            .map(|b| b.prompt_tokens.len())
            .min()
            .unwrap_or(0);

        let mut common_len = 0;
        for i in 0..min_len {
            let token = first[i];
            if batch.iter().all(|b| b.prompt_tokens[i] == token) {
                common_len = i + 1;
            } else {
                break;
            }
        }

        &first[..common_len]
    }
}

fn copy_tensor_block(
    cache: &crate::tensor::Tensor,
    src: usize,
    dst: usize,
) -> Result<(), EngineError> {
    let row = cache
        .narrow(0, src, 1)
        .map_err(tensor_err)
        .and_then(|t| (&t + 0.0f64).map_err(tensor_err))?
        .contiguous()
        .map_err(tensor_err)?;
    cache.slice_set(&row, 0, dst).map_err(tensor_err)
}
