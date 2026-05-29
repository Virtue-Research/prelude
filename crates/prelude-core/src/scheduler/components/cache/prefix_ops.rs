use std::sync::atomic::Ordering;

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

    /// Demand-driven reclaim: free at least `needed` physical KV blocks by
    /// evicting idle (cache-only, ref_count == 1) prefix-cache leaves, LRU-first.
    /// Returns the number of blocks actually freed (may be < needed if the rest
    /// of the cache is live-shared — the caller then falls through to defer/preempt).
    ///
    /// Lock order is the global invariant: prefix_cache FIRST, then block_manager.
    /// The reclaimability predicate takes a SHORT block-manager lock per candidate
    /// (never held across the final `release_paged_blocks`, which also locks it),
    /// so the non-reentrant mutex is never nested with itself.
    pub(crate) fn reclaim_idle_prefix_blocks(&self, needed: usize) -> Result<usize, EngineError> {
        if needed == 0 {
            return Ok(0);
        }
        let Some(ref pc_mutex) = self.prefix_cache else {
            return Ok(0);
        };
        let Some(ref bm_mutex) = self.block_manager else {
            return Ok(0);
        };
        let bm_arc = bm_mutex.clone();
        let mut pc = pc_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("prefix cache lock poisoned: {e}")))?;
        let mut is_reclaimable = |ids: &[u32]| -> bool {
            match bm_arc.lock() {
                Ok(bm) => ids.iter().all(|&b| bm.ref_count(b) == 1),
                Err(_) => false,
            }
        };
        let evicted = pc.reclaim_idle_blocks(needed, &mut is_reclaimable);
        drop(pc);
        let freed = evicted.len();
        self.release_paged_blocks(&evicted)?;
        if freed > 0 {
            debug!(needed, freed, "prefix cache demand reclaim");
        }
        Ok(freed)
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
        self.release_paged_blocks(&evicted)?;
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
        self.release_paged_blocks(&evicted)?;
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
        if !stored_paged_ids.is_empty() {
            self.prefix_cache_gen.fetch_add(1, Ordering::Relaxed);
        }
        self.retain_paged_blocks(&stored_paged_ids)?;
        let evicted = pc.take_evicted_paged_blocks();
        self.release_paged_blocks(&evicted)?;
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
        if !stored_paged_ids.is_empty() {
            self.prefix_cache_gen.fetch_add(1, Ordering::Relaxed);
        }
        self.retain_paged_blocks(&stored_paged_ids)?;
        let evicted = pc.take_evicted_paged_blocks();
        self.release_paged_blocks(&evicted)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::block_manager::BlockManager;
    use crate::cache::prefix_cache::PrefixKvCache;
    use std::sync::atomic::AtomicU64;
    use std::sync::{Arc, Mutex};

    // Two full blocks of 4 tokens each: trie chain root -> leaf.
    const TOKENS: &[u32] = &[0, 1, 2, 3, 4, 5, 6, 7];

    fn manager(bm: BlockManager, pc: PrefixKvCache) -> CacheManager {
        CacheManager {
            prefix_cache: Some(Mutex::new(pc)),
            paged_pool: None,
            block_manager: Some(Arc::new(Mutex::new(bm))),
            deltanet_pool: None,
            prefix_cache_gen: AtomicU64::new(0),
        }
    }

    fn lock_bm(cm: &CacheManager) -> std::sync::MutexGuard<'_, BlockManager> {
        cm.block_manager.as_ref().unwrap().lock().unwrap()
    }

    fn exhaust(cm: &CacheManager) {
        let mut bm = lock_bm(cm);
        while bm.allocate().is_some() {}
        assert_eq!(bm.available(), 0);
    }

    // Directly drives the A2 path that the live benchmark never reaches (admission
    // control + budget-LRU preempt it): allocate-exhausted -> reclaim -> freed.
    #[test]
    fn reclaim_frees_idle_cache_blocks_on_exhaustion() {
        let bs = 4;
        let mut bm = BlockManager::new(16, bs);
        let (b0, b1) = (bm.allocate().unwrap(), bm.allocate().unwrap());
        let cm = manager(bm, PrefixKvCache::new(bs, 1, 1, 64));

        // Insert the two blocks as a cached chain, retain (cache ref), then drop
        // the sequence ref → blocks are idle (rc==1, cache-only, not in free list).
        {
            let mut pc = cm.prefix_cache.as_ref().unwrap().lock().unwrap();
            let stored = pc.insert_paged_blocks_only(TOKENS, bs, &[b0, b1]);
            lock_bm(&cm).increment_refs(&stored);
        }
        lock_bm(&cm).decrement_refs(&[b0, b1]);
        assert_eq!(lock_bm(&cm).ref_count(b0), 1); // cache-only idle

        // Fill the rest of the pool with "live" blocks (rc==1, NOT in the trie).
        exhaust(&cm);

        let freed = cm.reclaim_idle_prefix_blocks(2).unwrap();
        assert_eq!(freed, 2, "both idle cache leaves reclaimed");
        let bm = lock_bm(&cm);
        assert_eq!(bm.available(), 2, "reclaimed blocks returned to the pool");
        assert_eq!(bm.ref_count(b0), 0);
        assert_eq!(bm.ref_count(b1), 0);
    }

    // The rc==1 predicate must protect blocks also referenced by a live sequence.
    #[test]
    fn reclaim_skips_live_shared_cache_blocks() {
        let bs = 4;
        let mut bm = BlockManager::new(16, bs);
        let (b0, b1) = (bm.allocate().unwrap(), bm.allocate().unwrap());
        let cm = manager(bm, PrefixKvCache::new(bs, 1, 1, 64));

        // Insert + retain but KEEP the sequence ref → blocks are live-shared (rc==2).
        {
            let mut pc = cm.prefix_cache.as_ref().unwrap().lock().unwrap();
            let stored = pc.insert_paged_blocks_only(TOKENS, bs, &[b0, b1]);
            lock_bm(&cm).increment_refs(&stored);
        }
        assert_eq!(lock_bm(&cm).ref_count(b0), 2);
        exhaust(&cm);

        let freed = cm.reclaim_idle_prefix_blocks(2).unwrap();
        assert_eq!(freed, 0, "live-shared cache blocks must never be reclaimed");
        let bm = lock_bm(&cm);
        assert_eq!(bm.available(), 0);
        assert!(bm.ref_count(b0) >= 2 && bm.ref_count(b1) >= 2);
    }
}
