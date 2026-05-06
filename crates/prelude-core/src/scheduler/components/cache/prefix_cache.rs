//! Block-level prefix KV cache with hash-trie matching and LRU eviction.
//!
//! Wraps `PrefixMatchIndex` (tensor-free) with
//! actual KV tensor storage and assembly via `Tensor::cat`/`narrow`.
//!
//! When paged attention is active, entries also store paged block IDs.
//! On a cache hit the prefix blocks are already in the paged pool, so the
//! engine only needs to scatter-write the suffix KV.
//!
//! Enable via environment variables:
//! - `PRELUDE_PREFIX_CACHE_BLOCKS=256` (max cached blocks, 0 = disabled)
//! - `PRELUDE_PREFIX_BLOCK_SIZE=64`    (tokens per block)

use std::collections::HashMap;

use crate::cache::deltanet_pool::DeltaNetPrefixState;
use crate::cache::prefix_index::PrefixMatchIndex;
use crate::cache::prefix_plan::PrefixResources;
use crate::tensor::{Result, Tensor};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Cached assembled KV tensors for a specific prefix chain.
/// Key is the hash of the last block in the chain.
struct AssembledKvCache {
    /// Per-layer (key, value) tensors for the entire prefix.
    kv_per_layer: Vec<(Tensor, Tensor)>,
    /// Number of tokens this assembled cache covers.
    cached_len: usize,
}

/// Block-level prefix KV cache with hash-trie matching.
pub struct PrefixKvCache {
    /// Tensor-free trie index (matching, LRU, paged block tracking).
    index: PrefixMatchIndex,
    /// Dimension along which to concatenate blocks (1 for flash, 2 for standard).
    concat_dim: usize,
    /// Number of transformer layers.
    num_layers: usize,
    /// Per-layer KV tensors for each cached block, keyed by block hash.
    kv_store: HashMap<u64, Vec<(Tensor, Tensor)>>,
    /// Optional hybrid-model state at selected block boundaries.
    deltanet_state_store: HashMap<u64, DeltaNetPrefixState>,
    /// Cache of pre-assembled KV tensors, keyed by the last block hash.
    assembled_cache: HashMap<u64, AssembledKvCache>,
}

impl PrefixKvCache {
    pub fn new(block_size: usize, concat_dim: usize, num_layers: usize, max_blocks: usize) -> Self {
        Self {
            index: PrefixMatchIndex::new(block_size, max_blocks),
            concat_dim,
            num_layers,
            kv_store: HashMap::new(),
            deltanet_state_store: HashMap::new(),
            assembled_cache: HashMap::new(),
        }
    }

    /// Whether the cache is enabled (has capacity).
    #[inline]
    pub fn enabled(&self) -> bool {
        self.index.enabled()
    }

    /// Number of blocks currently cached.
    #[inline]
    pub fn cached_blocks(&self) -> usize {
        self.index.cached_blocks()
    }

    // -----------------------------------------------------------------------
    // Match + assemble
    // -----------------------------------------------------------------------

    /// Match prefix and return paged block IDs (for unified paged attention flow).
    ///
    /// Returns `(cached_token_count, paged_block_ids, Option<layer_kvs>)`.
    pub fn match_and_assemble_paged(
        &mut self,
        tokens: &[u32],
    ) -> Result<(usize, Vec<u32>, Option<Vec<(Tensor, Tensor)>>)> {
        let m = self.index.match_prefix(tokens);
        if m.matched_hashes.is_empty() {
            return Ok((0, vec![], None));
        }

        if !self.index.all_have_paged(&m.matched_hashes) {
            // Fall back to tensor-only match
            let assembled = self.assemble_kv(&m.matched_hashes, m.cached_len)?;
            return Ok((m.cached_len, vec![], Some(assembled)));
        }

        let paged_blocks = self.index.collect_paged_blocks(&m.matched_hashes);
        let assembled = self.assemble_kv(&m.matched_hashes, m.cached_len)?;
        Ok((m.cached_len, paged_blocks, Some(assembled)))
    }

    /// Match prefix and return only paged block IDs.
    pub fn match_paged_blocks_only(&mut self, tokens: &[u32]) -> Result<(usize, Vec<u32>)> {
        let m = self.index.match_prefix(tokens);
        if m.matched_hashes.is_empty() {
            return Ok((0, vec![]));
        }

        if !self.index.all_have_paged(&m.matched_hashes) {
            return Ok((0, vec![]));
        }

        let paged_blocks = self.index.collect_paged_blocks(&m.matched_hashes);
        Ok((m.cached_len, paged_blocks))
    }

    /// Match prefix for hybrid models and require a DeltaNet state snapshot at
    /// the returned boundary.
    pub fn match_paged_blocks_with_deltanet_state(
        &mut self,
        tokens: &[u32],
    ) -> Result<(usize, Vec<u32>, Option<DeltaNetPrefixState>)> {
        let m = self.index.match_prefix(tokens);
        if m.matched_hashes.is_empty() {
            return Ok((0, vec![], None));
        }

        for end in (1..=m.matched_hashes.len()).rev() {
            let hashes = &m.matched_hashes[..end];
            let Some(state) = self.deltanet_state_store.get(hashes.last().unwrap()) else {
                continue;
            };
            let resources = PrefixResources {
                paged_kv: self.index.all_have_paged(hashes),
                recurrent_state: true,
            };
            if !(resources.paged_kv && resources.recurrent_state) {
                continue;
            }

            let cached_len = end * self.index.block_size();
            let paged_blocks = self.index.collect_paged_blocks(hashes);
            return Ok((cached_len, paged_blocks, Some(state.clone())));
        }

        Ok((0, vec![], None))
    }

    // -----------------------------------------------------------------------
    // Insert
    // -----------------------------------------------------------------------

    /// Insert block-aligned KV data along with paged block IDs.
    ///
    /// Returns the list of paged block IDs that were stored (for ref count incrementing).
    pub fn insert_blocks_with_paged(
        &mut self,
        tokens: &[u32],
        layer_kvs: &[(Tensor, Tensor)],
        paged_block_size: usize,
        full_block_table: &[u32],
    ) -> Result<Vec<u32>> {
        let paged_map = PrefixMatchIndex::compute_paged_map(
            tokens.len(),
            self.index.block_size(),
            paged_block_size,
            full_block_table,
        );
        self.insert_blocks_inner(tokens, layer_kvs, Some(&paged_map))
    }

    /// Insert only paged block IDs into the prefix trie, without storing KV tensors.
    ///
    /// Used by classify/embed where KV lives in the paged pool and the model's
    /// internal KV cache is disabled. Returns the list of paged block IDs that
    /// were stored (for ref count incrementing).
    pub fn insert_paged_blocks_only(
        &mut self,
        tokens: &[u32],
        paged_block_size: usize,
        full_block_table: &[u32],
    ) -> Vec<u32> {
        self.insert_paged_blocks_only_inner(tokens, paged_block_size, full_block_table, None)
    }

    /// Insert paged block IDs and attach a DeltaNet state snapshot to the
    /// block-aligned prefix boundary represented by `tokens`.
    pub fn insert_paged_blocks_with_deltanet_state(
        &mut self,
        tokens: &[u32],
        paged_block_size: usize,
        full_block_table: &[u32],
        deltanet_state: DeltaNetPrefixState,
    ) -> Vec<u32> {
        self.insert_paged_blocks_only_inner(
            tokens,
            paged_block_size,
            full_block_table,
            Some(deltanet_state),
        )
    }

    fn insert_paged_blocks_only_inner(
        &mut self,
        tokens: &[u32],
        paged_block_size: usize,
        full_block_table: &[u32],
        deltanet_state: Option<DeltaNetPrefixState>,
    ) -> Vec<u32> {
        if !self.index.enabled() {
            return Vec::new();
        }

        let paged_map = PrefixMatchIndex::compute_paged_map(
            tokens.len(),
            self.index.block_size(),
            paged_block_size,
            full_block_table,
        );
        let plan = self.index.insert_blocks(tokens, Some(&paged_map));

        if let Some(state) = deltanet_state
            && tokens.len() >= self.index.block_size()
            && tokens.len() % self.index.block_size() == 0
        {
            let hash = Self::last_full_block_hash(tokens, self.index.block_size());
            self.deltanet_state_store.insert(hash, state);
        }

        // No KV tensor storage needed — KV lives in the paged pool.
        // Clean up any evicted entries.
        for hash in self.index.take_evicted_hashes() {
            self.kv_store.remove(&hash);
            self.deltanet_state_store.remove(&hash);
            self.assembled_cache.remove(&hash);
        }

        plan.stored_paged_blocks
    }

    fn last_full_block_hash(tokens: &[u32], block_size: usize) -> u64 {
        let mut parent_hash = 0u64;
        for block_tokens in tokens.chunks(block_size) {
            if block_tokens.len() < block_size {
                break;
            }
            parent_hash = PrefixMatchIndex::hash_block(parent_hash, block_tokens);
        }
        parent_hash
    }

    fn insert_blocks_inner(
        &mut self,
        tokens: &[u32],
        layer_kvs: &[(Tensor, Tensor)],
        paged_map: Option<&[Vec<u32>]>,
    ) -> Result<Vec<u32>> {
        if !self.index.enabled() || layer_kvs.len() != self.num_layers {
            return Ok(Vec::new());
        }

        let block_size = self.index.block_size();
        let plan = self.index.insert_blocks(tokens, paged_map);

        // Extract and store KV tensors for newly created blocks
        for &(block_idx, hash) in &plan.new_blocks {
            let block_start = block_idx * block_size;
            let mut kv_per_layer = Vec::with_capacity(self.num_layers);
            for (k_full, v_full) in layer_kvs {
                let k_block = k_full
                    .narrow(self.concat_dim, block_start, block_size)?
                    .contiguous()?;
                let v_block = v_full
                    .narrow(self.concat_dim, block_start, block_size)?
                    .contiguous()?;
                kv_per_layer.push((k_block, v_block));
            }
            self.kv_store.insert(hash, kv_per_layer);
        }

        // Clean up KV data for evicted entries
        for hash in self.index.take_evicted_hashes() {
            self.kv_store.remove(&hash);
            self.deltanet_state_store.remove(&hash);
            self.assembled_cache.remove(&hash);
        }

        Ok(plan.stored_paged_blocks)
    }

    /// Take any paged block IDs that were collected during eviction.
    /// The caller must decrement ref counts on these in the BlockManager.
    pub fn take_evicted_paged_blocks(&mut self) -> Vec<u32> {
        self.index.take_evicted_paged_blocks()
    }

    // -----------------------------------------------------------------------
    // Internal: KV assembly
    // -----------------------------------------------------------------------

    /// Assemble per-layer KV tensors from cached blocks.
    fn assemble_kv(
        &mut self,
        matched_hashes: &[u64],
        cached_len: usize,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let last_hash = *matched_hashes.last().unwrap();

        // Check pre-assembled cache
        if let Some(cached) = self.assembled_cache.get(&last_hash) {
            if cached.cached_len == cached_len {
                return Ok(cached
                    .kv_per_layer
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect());
            }
        }

        // Assemble per-layer KV by concatenating matched blocks
        let mut assembled: Vec<(Tensor, Tensor)> = Vec::with_capacity(self.num_layers);
        for layer_idx in 0..self.num_layers {
            if matched_hashes.len() == 1 {
                let kv = &self.kv_store[&matched_hashes[0]][layer_idx];
                assembled.push((kv.0.clone(), kv.1.clone()));
            } else {
                let k_blocks: Vec<&Tensor> = matched_hashes
                    .iter()
                    .map(|h| &self.kv_store[h][layer_idx].0)
                    .collect();
                let v_blocks: Vec<&Tensor> = matched_hashes
                    .iter()
                    .map(|h| &self.kv_store[h][layer_idx].1)
                    .collect();
                let k = Tensor::cat(&k_blocks, self.concat_dim)?;
                let v = Tensor::cat(&v_blocks, self.concat_dim)?;
                assembled.push((k, v));
            }
        }

        // Cache the assembled result
        self.assembled_cache.insert(
            last_hash,
            AssembledKvCache {
                kv_per_layer: assembled
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
                cached_len,
            },
        );

        Ok(assembled)
    }
}

#[cfg(test)]
impl PrefixKvCache {
    /// Find the longest cached prefix and assemble per-layer KV tensors.
    /// Test-only: production code uses `match_and_assemble_paged()`.
    fn match_and_assemble(
        &mut self,
        tokens: &[u32],
    ) -> Result<(usize, Option<Vec<(Tensor, Tensor)>>)> {
        let m = self.index.match_prefix(tokens);
        if m.matched_hashes.is_empty() {
            return Ok((0, None));
        }

        let assembled = self.assemble_kv(&m.matched_hashes, m.cached_len)?;
        Ok((m.cached_len, Some(assembled)))
    }

    /// Insert block-aligned KV data from a completed prefill.
    /// Test-only: production code uses `insert_blocks_with_paged()` or `insert_paged_blocks_only()`.
    fn insert_blocks(&mut self, tokens: &[u32], layer_kvs: &[(Tensor, Tensor)]) -> Result<()> {
        self.insert_blocks_inner(tokens, layer_kvs, None)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, Device};

    fn make_kv(block_size: usize, num_layers: usize) -> Vec<(Tensor, Tensor)> {
        (0..num_layers)
            .map(|_| {
                let k = Tensor::zeros((1, block_size, 2, 64), DType::F32, &Device::Cpu).unwrap();
                let v = Tensor::zeros((1, block_size, 2, 64), DType::F32, &Device::Cpu).unwrap();
                (k, v)
            })
            .collect()
    }

    #[test]
    fn test_basic_insert_and_match() {
        let mut cache = PrefixKvCache::new(4, 1, 2, 16);
        let tokens: Vec<u32> = (0..12).collect();
        let layer_kvs = make_kv(12, 2);

        cache.insert_blocks(&tokens, &layer_kvs).unwrap();
        assert_eq!(cache.cached_blocks(), 3);

        let (cached_len, kvs) = cache.match_and_assemble(&tokens).unwrap();
        assert_eq!(cached_len, 8);
        assert!(kvs.is_some());
        let kvs = kvs.unwrap();
        assert_eq!(kvs.len(), 2);
        assert_eq!(kvs[0].0.dim(1).unwrap(), 8);
    }

    #[test]
    fn test_no_match_empty_cache() {
        let mut cache = PrefixKvCache::new(4, 1, 2, 16);
        let tokens: Vec<u32> = (0..10).collect();
        let (cached_len, kvs) = cache.match_and_assemble(&tokens).unwrap();
        assert_eq!(cached_len, 0);
        assert!(kvs.is_none());
    }

    #[test]
    fn test_partial_prefix_match() {
        let mut cache = PrefixKvCache::new(4, 1, 2, 16);

        let tokens_a: Vec<u32> = (0..8).collect();
        let layer_kvs = make_kv(8, 2);
        cache.insert_blocks(&tokens_a, &layer_kvs).unwrap();

        let mut tokens_b: Vec<u32> = (0..8).collect();
        tokens_b.extend(100..104);
        let (cached_len, kvs) = cache.match_and_assemble(&tokens_b).unwrap();
        assert_eq!(cached_len, 8);
        assert!(kvs.is_some());
    }

    #[test]
    fn test_eviction() {
        let mut cache = PrefixKvCache::new(4, 1, 2, 2);
        let tokens: Vec<u32> = (0..12).collect();
        let layer_kvs = make_kv(12, 2);
        cache.insert_blocks(&tokens, &layer_kvs).unwrap();
        assert!(cache.cached_blocks() <= 2);
    }

    #[test]
    fn test_guarantees_suffix_token() {
        let mut cache = PrefixKvCache::new(4, 1, 2, 16);

        let tokens: Vec<u32> = (0..8).collect();
        let layer_kvs = make_kv(8, 2);
        cache.insert_blocks(&tokens, &layer_kvs).unwrap();

        let (cached_len, _) = cache.match_and_assemble(&tokens).unwrap();
        assert_eq!(cached_len, 4);

        let tokens9: Vec<u32> = (0..9).collect();
        let (cached_len, _) = cache.match_and_assemble(&tokens9).unwrap();
        assert_eq!(cached_len, 8);
    }

    #[test]
    fn test_disabled_cache() {
        let mut cache = PrefixKvCache::new(4, 1, 2, 0);
        assert!(!cache.enabled());
        let tokens: Vec<u32> = (0..8).collect();
        let (cached_len, _) = cache.match_and_assemble(&tokens).unwrap();
        assert_eq!(cached_len, 0);
    }

    #[test]
    fn test_paged_block_ids_insert_and_match() {
        let mut cache = PrefixKvCache::new(4, 1, 2, 16);
        let tokens: Vec<u32> = (0..12).collect();
        let layer_kvs = make_kv(12, 2);

        let full_block_table: Vec<u32> = vec![10, 11, 12, 13, 14, 15];
        let stored = cache
            .insert_blocks_with_paged(&tokens, &layer_kvs, 2, &full_block_table)
            .unwrap();
        assert_eq!(stored.len(), 6);

        let (cached_len, paged_ids, kvs) = cache.match_and_assemble_paged(&tokens).unwrap();
        assert_eq!(cached_len, 8);
        assert_eq!(paged_ids, vec![10, 11, 12, 13]);
        assert!(kvs.is_some());
    }

    #[test]
    fn test_eviction_collects_paged_blocks() {
        let mut cache = PrefixKvCache::new(4, 1, 2, 2);
        let tokens: Vec<u32> = (0..12).collect();
        let layer_kvs = make_kv(12, 2);
        let full_block_table: Vec<u32> = vec![10, 11, 12, 13, 14, 15];

        cache
            .insert_blocks_with_paged(&tokens, &layer_kvs, 2, &full_block_table)
            .unwrap();

        let evicted = cache.take_evicted_paged_blocks();
        assert!(!evicted.is_empty());
    }

    #[test]
    fn test_match_paged_blocks_only_dedups_adjacent_ids() {
        let mut cache = PrefixKvCache::new(4, 1, 2, 16);
        let tokens: Vec<u32> = (0..12).collect();
        let layer_kvs = make_kv(12, 2);

        let full_block_table: Vec<u32> = vec![20, 21];
        cache
            .insert_blocks_with_paged(&tokens, &layer_kvs, 8, &full_block_table)
            .unwrap();

        let (cached_len, paged_ids) = cache.match_paged_blocks_only(&tokens).unwrap();
        assert_eq!(cached_len, 8);
        assert_eq!(paged_ids, vec![20]);
    }
}
