//! Block-level prefix matching index with hash-trie and LRU eviction.
//!
//! This is the tensor-free core of prefix KV caching. It tracks which
//! token blocks are cached, their parent/child relationships, LRU order,
//! and optional paged block IDs — but stores no tensor data.
//!
//! The engine layer wraps this with actual KV tensor storage and assembly.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct PrefixEntry {
    /// Hash of the parent block (None for the first block).
    parent: Option<u64>,
    /// Paged block IDs that store this entry's KV in the paged pool.
    /// When Some, these blocks are ref-counted in the BlockManager.
    paged_block_ids: Option<Vec<u32>>,
    /// Number of child blocks that reference this block as parent.
    children: usize,
    /// Monotonic counter for LRU tracking.
    access_id: u64,
}

/// Result of a prefix match operation.
pub struct PrefixMatch {
    /// Hashes of matched blocks, in order.
    pub matched_hashes: Vec<u64>,
    /// Total number of cached tokens (matched_hashes.len() * block_size).
    pub cached_len: usize,
}

/// Result of a prefix insert operation.
pub struct PrefixInsertPlan {
    /// Paged block IDs that were stored in newly inserted or updated entries.
    /// The caller should increment ref counts on these in the BlockManager.
    pub stored_paged_blocks: Vec<u32>,
    /// Blocks that were newly created: (block_index_in_sequence, hash).
    /// The caller needs to store KV tensor data for these hashes.
    pub new_blocks: Vec<(usize, u64)>,
}

/// Block-level prefix matching index with hash-trie and LRU eviction.
///
/// Tracks which token blocks are cached, their trie structure, LRU ordering,
/// and optional paged block IDs. Does NOT store any tensor data — that's the
/// engine layer's responsibility.
pub struct PrefixMatchIndex {
    /// Number of tokens per block.
    block_size: usize,
    /// Hash → entry mapping.
    entries: HashMap<u64, PrefixEntry>,
    /// Set of leaf hashes (entries with children == 0).
    leaf_set: HashSet<u64>,
    /// LRU queue for leaf eviction: (hash, access_id).
    leaf_lru: VecDeque<(u64, u64)>,
    /// Monotonic access counter.
    access_counter: u64,
    /// Maximum number of blocks to cache.
    max_blocks: usize,
    /// Paged block IDs pending release after eviction.
    evicted_paged_blocks: Vec<u32>,
    /// Hashes of entries that were evicted (for KV store cleanup).
    evicted_hashes: Vec<u64>,
}

impl PrefixMatchIndex {
    pub fn new(block_size: usize, max_blocks: usize) -> Self {
        Self {
            block_size: block_size.max(1),
            entries: HashMap::new(),
            leaf_set: HashSet::new(),
            leaf_lru: VecDeque::new(),
            access_counter: 0,
            max_blocks,
            evicted_paged_blocks: Vec::new(),
            evicted_hashes: Vec::new(),
        }
    }

    /// Whether the cache is enabled (has capacity).
    #[inline]
    pub fn enabled(&self) -> bool {
        self.max_blocks > 0
    }

    /// Number of blocks currently cached.
    #[inline]
    pub fn cached_blocks(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    // -----------------------------------------------------------------------
    // Match
    // -----------------------------------------------------------------------

    /// Find the longest cached prefix for the given tokens.
    ///
    /// Returns matched block hashes and cached token count.
    /// Guarantees at least 1 token remains for the suffix forward pass.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        let empty = PrefixMatch {
            matched_hashes: Vec::new(),
            cached_len: 0,
        };

        if !self.enabled() || tokens.len() < self.block_size + 1 {
            return empty;
        }

        let max_matchable = (tokens.len() - 1) / self.block_size;
        if max_matchable == 0 {
            return empty;
        }

        let mut matched_hashes: Vec<u64> = Vec::new();
        let mut parent_hash = 0u64;
        for block_tokens in tokens.chunks(self.block_size).take(max_matchable) {
            if block_tokens.len() < self.block_size {
                break;
            }
            let hash = Self::hash_block(parent_hash, block_tokens);
            if self.entries.contains_key(&hash) {
                matched_hashes.push(hash);
                self.touch(hash);
                parent_hash = hash;
            } else {
                break;
            }
        }

        let cached_len = matched_hashes.len() * self.block_size;
        PrefixMatch {
            matched_hashes,
            cached_len,
        }
    }

    /// Check if all given hashes have paged block IDs.
    pub fn all_have_paged(&self, hashes: &[u64]) -> bool {
        hashes.iter().all(|h| {
            self.entries
                .get(h)
                .is_some_and(|e| e.paged_block_ids.is_some())
        })
    }

    /// Build a canonical paged-attn block table from matched prefix entries.
    ///
    /// Adjacent duplicate page IDs are collapsed (can happen when prefix-cache
    /// block size is smaller than paged-attn block size).
    pub fn collect_paged_blocks(&self, hashes: &[u64]) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for h in hashes {
            if let Some(entry) = self.entries.get(h) {
                if let Some(ref ids) = entry.paged_block_ids {
                    for &id in ids {
                        if out.last().copied() != Some(id) {
                            out.push(id);
                        }
                    }
                }
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Insert
    // -----------------------------------------------------------------------

    /// Insert block-aligned entries for the given tokens.
    ///
    /// For each full block in `tokens`:
    /// - If already cached: touch for LRU, optionally add paged block IDs.
    /// - If new: create entry, update parent/children, push to leaf LRU.
    ///
    /// `paged_map[i]` contains paged block IDs for block `i` (if paged attention is active).
    ///
    /// Returns a plan telling the caller which blocks are new (need KV data stored).
    /// Also runs eviction if needed — caller must check `take_evicted_hashes()`.
    pub fn insert_blocks(
        &mut self,
        tokens: &[u32],
        paged_map: Option<&[Vec<u32>]>,
    ) -> PrefixInsertPlan {
        let mut plan = PrefixInsertPlan {
            stored_paged_blocks: Vec::new(),
            new_blocks: Vec::new(),
        };

        if !self.enabled() {
            return plan;
        }

        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            return plan;
        }

        let mut parent_hash: Option<u64> = None;
        for (block_idx, block_tokens) in
            tokens.chunks(self.block_size).take(full_blocks).enumerate()
        {
            if block_tokens.len() < self.block_size {
                break;
            }

            let hash = Self::hash_block(parent_hash.unwrap_or(0), block_tokens);

            if self.entries.contains_key(&hash) {
                // Already cached — touch for LRU freshness
                self.touch(hash);
                // If entry doesn't have paged block IDs yet but we have them now, store them
                if let Some(entry) = self.entries.get_mut(&hash) {
                    if entry.paged_block_ids.is_none() {
                        if let Some(ref pm) = paged_map {
                            if block_idx < pm.len() {
                                let ids = pm[block_idx].clone();
                                plan.stored_paged_blocks.extend_from_slice(&ids);
                                entry.paged_block_ids = Some(ids);
                            }
                        }
                    }
                }
            } else {
                // New block — create entry
                let paged_block_ids = paged_map.and_then(|pm| {
                    if block_idx < pm.len() {
                        let ids = pm[block_idx].clone();
                        plan.stored_paged_blocks.extend_from_slice(&ids);
                        Some(ids)
                    } else {
                        None
                    }
                });

                // Update parent's children count
                if let Some(p) = parent_hash {
                    if let Some(parent_entry) = self.entries.get_mut(&p) {
                        if parent_entry.children == 0 {
                            self.leaf_set.remove(&p);
                        }
                        parent_entry.children += 1;
                    }
                }

                let access_id = self.next_access_id();
                self.entries.insert(
                    hash,
                    PrefixEntry {
                        parent: parent_hash,
                        paged_block_ids,
                        children: 0,
                        access_id,
                    },
                );
                self.leaf_set.insert(hash);
                self.leaf_lru.push_back((hash, access_id));

                plan.new_blocks.push((block_idx, hash));
            }

            parent_hash = Some(hash);
        }

        self.evict_if_needed();
        plan
    }

    /// Compute the paged block mapping for a prefix insert with paged attention.
    ///
    /// Given prefix_block_size and the full paged block table, returns a map
    /// where `result[i]` contains the paged block IDs overlapping prefix block `i`.
    pub fn compute_paged_map(
        tokens_len: usize,
        prefix_block_size: usize,
        paged_block_size: usize,
        full_block_table: &[u32],
    ) -> Vec<Vec<u32>> {
        let paged_block_size = paged_block_size.max(1);
        let full_blocks = tokens_len / prefix_block_size;
        let mut paged_map: Vec<Vec<u32>> = Vec::with_capacity(full_blocks);
        for block_idx in 0..full_blocks {
            let token_start = block_idx * prefix_block_size;
            let token_end = token_start + prefix_block_size;
            let paged_start = token_start / paged_block_size;
            let paged_end = token_end.div_ceil(paged_block_size);
            if paged_start >= full_block_table.len() {
                paged_map.push(Vec::new());
                continue;
            }
            let paged_end = paged_end.min(full_block_table.len());
            paged_map.push(full_block_table[paged_start..paged_end].to_vec());
        }
        paged_map
    }

    /// Take paged block IDs that were collected during eviction.
    /// The caller must decrement ref counts on these in the BlockManager.
    pub fn take_evicted_paged_blocks(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.evicted_paged_blocks)
    }

    /// Take hashes of entries that were evicted.
    /// The caller should remove corresponding KV tensor data.
    pub fn take_evicted_hashes(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.evicted_hashes)
    }

    // -----------------------------------------------------------------------
    // Eviction (LRU on leaf nodes only)
    // -----------------------------------------------------------------------

    fn evict_if_needed(&mut self) {
        while self.entries.len() > self.max_blocks {
            let Some((hash, access_id)) = self.leaf_lru.pop_front() else {
                break;
            };
            // Skip stale or non-leaf entries
            if !self.leaf_set.contains(&hash) {
                continue;
            }
            let Some(entry) = self.entries.get(&hash) else {
                continue;
            };
            if entry.access_id != access_id || entry.children > 0 {
                continue;
            }

            let entry = self.entries.remove(&hash).unwrap();
            self.leaf_set.remove(&hash);
            self.evicted_hashes.push(hash);

            // Collect paged block IDs for deferred ref count decrement
            if let Some(paged_ids) = entry.paged_block_ids {
                self.evicted_paged_blocks.extend_from_slice(&paged_ids);
            }

            // If parent becomes childless, it becomes a new leaf
            if let Some(parent) = entry.parent {
                if let Some(parent_entry) = self.entries.get_mut(&parent) {
                    parent_entry.children = parent_entry.children.saturating_sub(1);
                    if parent_entry.children == 0 {
                        self.leaf_set.insert(parent);
                        self.leaf_lru.push_back((parent, parent_entry.access_id));
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn touch(&mut self, hash: u64) {
        let access_id = self.next_access_id();
        if let Some(entry) = self.entries.get_mut(&hash) {
            entry.access_id = access_id;
            if self.leaf_set.contains(&hash) {
                self.leaf_lru.push_back((hash, access_id));
            }
        }
    }

    fn next_access_id(&mut self) -> u64 {
        self.access_counter = self.access_counter.wrapping_add(1);
        self.access_counter
    }

    /// Hash a block of tokens with parent chaining.
    pub fn hash_block(parent_hash: u64, tokens: &[u32]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        parent_hash.hash(&mut hasher);
        tokens.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_and_match() {
        let mut index = PrefixMatchIndex::new(4, 16);
        let tokens: Vec<u32> = (0..12).collect(); // 3 blocks of 4

        let plan = index.insert_blocks(&tokens, None);
        assert_eq!(plan.new_blocks.len(), 3);
        assert_eq!(index.cached_blocks(), 3);

        // Match: max_matchable = (12-1)/4 = 2
        let m = index.match_prefix(&tokens);
        assert_eq!(m.cached_len, 8); // 2 blocks matched
        assert_eq!(m.matched_hashes.len(), 2);
    }

    #[test]
    fn test_no_match_empty() {
        let mut index = PrefixMatchIndex::new(4, 16);
        let tokens: Vec<u32> = (0..10).collect();
        let m = index.match_prefix(&tokens);
        assert_eq!(m.cached_len, 0);
        assert!(m.matched_hashes.is_empty());
    }

    #[test]
    fn test_partial_prefix_match() {
        let mut index = PrefixMatchIndex::new(4, 16);

        // Insert [0..8]
        let tokens_a: Vec<u32> = (0..8).collect();
        index.insert_blocks(&tokens_a, None);

        // Query [0..8] ++ [100..104]
        let mut tokens_b: Vec<u32> = (0..8).collect();
        tokens_b.extend(100..104);
        let m = index.match_prefix(&tokens_b);
        assert_eq!(m.cached_len, 8);
    }

    #[test]
    fn test_eviction() {
        let mut index = PrefixMatchIndex::new(4, 2); // max 2 blocks
        let tokens: Vec<u32> = (0..12).collect(); // 3 blocks

        let _plan = index.insert_blocks(&tokens, None);
        assert!(index.cached_blocks() <= 2);

        let evicted = index.take_evicted_hashes();
        assert!(!evicted.is_empty());
    }

    #[test]
    fn test_guarantees_suffix_token() {
        let mut index = PrefixMatchIndex::new(4, 16);

        // Insert 2 blocks
        let tokens: Vec<u32> = (0..8).collect();
        index.insert_blocks(&tokens, None);

        // Query with exactly 8 tokens: max_matchable = (8-1)/4 = 1
        let m = index.match_prefix(&tokens);
        assert_eq!(m.cached_len, 4);

        // Query with 9 tokens: max_matchable = (9-1)/4 = 2
        let tokens9: Vec<u32> = (0..9).collect();
        let m = index.match_prefix(&tokens9);
        assert_eq!(m.cached_len, 8);
    }

    #[test]
    fn test_disabled() {
        let mut index = PrefixMatchIndex::new(4, 0);
        assert!(!index.enabled());
        let m = index.match_prefix(&(0..8).collect::<Vec<_>>());
        assert_eq!(m.cached_len, 0);
    }

    #[test]
    fn test_paged_block_ids() {
        let mut index = PrefixMatchIndex::new(4, 16);
        let tokens: Vec<u32> = (0..12).collect();

        // paged_block_size=2: each prefix block (4 tokens) maps to 2 paged blocks
        let paged_map = PrefixMatchIndex::compute_paged_map(12, 4, 2, &[10, 11, 12, 13, 14, 15]);
        assert_eq!(paged_map.len(), 3);
        assert_eq!(paged_map[0], vec![10, 11]);
        assert_eq!(paged_map[1], vec![12, 13]);

        let plan = index.insert_blocks(&tokens, Some(&paged_map));
        assert_eq!(plan.stored_paged_blocks.len(), 6);

        // Match and get paged blocks
        let m = index.match_prefix(&tokens);
        assert!(index.all_have_paged(&m.matched_hashes));
        let paged = index.collect_paged_blocks(&m.matched_hashes);
        assert_eq!(paged, vec![10, 11, 12, 13]); // 2 prefix blocks = 4 paged blocks
    }

    #[test]
    fn test_eviction_collects_paged_blocks() {
        let mut index = PrefixMatchIndex::new(4, 2); // max 2 blocks
        let tokens: Vec<u32> = (0..12).collect();
        let paged_map = PrefixMatchIndex::compute_paged_map(12, 4, 2, &[10, 11, 12, 13, 14, 15]);

        index.insert_blocks(&tokens, Some(&paged_map));
        let evicted_paged = index.take_evicted_paged_blocks();
        assert!(!evicted_paged.is_empty());
    }

    #[test]
    fn test_paged_dedup_adjacent() {
        let mut index = PrefixMatchIndex::new(4, 16);
        let tokens: Vec<u32> = (0..12).collect();

        // paged_block_size=8 > prefix_block_size=4:
        // prefix blocks map to paged IDs [20], [20], [21]
        let paged_map = PrefixMatchIndex::compute_paged_map(12, 4, 8, &[20, 21]);
        index.insert_blocks(&tokens, Some(&paged_map));

        let m = index.match_prefix(&tokens);
        let paged = index.collect_paged_blocks(&m.matched_hashes);
        // Adjacent duplicate 20 should be collapsed
        assert_eq!(paged, vec![20]);
    }

    #[test]
    fn test_existing_entry_gets_paged_ids() {
        let mut index = PrefixMatchIndex::new(4, 16);
        let tokens: Vec<u32> = (0..8).collect();

        // First insert without paged
        index.insert_blocks(&tokens, None);
        let m = index.match_prefix(&(0..9).collect::<Vec<_>>());
        assert!(!index.all_have_paged(&m.matched_hashes));

        // Second insert with paged — should update existing entries
        let paged_map = PrefixMatchIndex::compute_paged_map(8, 4, 2, &[10, 11, 12, 13]);
        let plan = index.insert_blocks(&tokens, Some(&paged_map));
        assert!(!plan.stored_paged_blocks.is_empty());

        let m = index.match_prefix(&(0..9).collect::<Vec<_>>());
        assert!(index.all_have_paged(&m.matched_hashes));
    }
}
