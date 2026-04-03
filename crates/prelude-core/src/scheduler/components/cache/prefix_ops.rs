use crate::cache::manager::CacheManager;
#[cfg(feature = "cuda")]
use crate::engine::candle_err;
#[cfg(feature = "cuda")]
use crate::engine::EngineError;
use crate::engine::PreparedGenerateRequest;
#[cfg(feature = "cuda")]
use tracing::debug;

impl CacheManager {
    /// Match prefix cache for paged-attention runs and return only block IDs.
    /// This avoids assembling per-layer KV tensors when the caller can consume
    /// paged blocks directly.
    #[cfg(feature = "cuda")]
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
        let (cached_len, paged_ids) = pc.match_paged_blocks_only(tokens).map_err(candle_err)?;
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

    /// Insert only paged block IDs into prefix cache (no KV tensor extraction).
    #[cfg(feature = "cuda")]
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
        let stored_paged_ids =
            pc.insert_paged_blocks_only(tokens, paged_block_size, block_table);
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
