//! Physical block manager for paged KV cache with ref counting and a
//! vLLM-style lazy prefix cache.
//!
//! Manages a fixed pool of GPU KV cache blocks. Each block holds
//! `block_size` tokens worth of key/value data. Sequences are assigned
//! blocks on demand and freed when complete.
//!
//! Ref counting enables shared blocks between active sequences and the
//! prefix cache — blocks are only freed when all references are released.
//!
//! ## Lazy prefix cache (vLLM-style)
//!
//! Full blocks can carry a content hash (`assign_block_hashes`). When a
//! block's ref count drops to 0 it enters the free LRU queue **keeping its
//! hash**, so it remains matchable (`lookup_and_touch_prefix` revives it,
//! ref 0 → 1). The hash is only dropped — i.e. the cache entry is only
//! evicted — when the physical block is actually popped for reuse by
//! `allocate`. The cache window is therefore the whole pool, there is no
//! separate cache capacity, and eviction order is LRU over free blocks
//! (callers free sequence tables tail-first via `free`, so chain tails are
//! evicted before their parents).
//!
//! The free queue uses lazy deletion: each push is stamped; revived or
//! re-freed blocks leave stale entries behind that `allocate` skips
//! (`free_stamp` mismatch or ref_count > 0). `available()` tracks the true
//! number of ref==0 blocks.

use std::collections::{HashMap, VecDeque};

pub struct BlockManager {
    block_size: usize,
    /// Free LRU queue: (block_id, stamp). Entries are valid iff the stamp
    /// matches `free_stamp[block_id]` and the block still has ref_count 0.
    free_blocks: VecDeque<(u32, u64)>,
    /// Latest queue stamp per block (for lazy deletion of stale entries).
    free_stamp: Vec<u64>,
    next_stamp: u64,
    /// True number of blocks with ref_count == 0.
    num_free: usize,
    ref_counts: Vec<u32>,
    /// Content hash carried by each block (lazy prefix cache).
    block_hash: Vec<Option<u64>>,
    /// hash → block_id for prefix matching. An entry may point at a block
    /// that is currently free (ref 0, still in the queue) — that is the
    /// whole point of lazy eviction.
    hash_to_block: HashMap<u64, u32>,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let free_blocks: VecDeque<(u32, u64)> = (0..num_blocks as u32).map(|b| (b, 0)).collect();
        Self {
            block_size,
            free_blocks,
            free_stamp: vec![0u64; num_blocks],
            next_stamp: 0,
            num_free: num_blocks,
            ref_counts: vec![0u32; num_blocks],
            block_hash: vec![None; num_blocks],
            hash_to_block: HashMap::new(),
        }
    }

    /// Number of free (ref_count == 0) blocks available. Cached-but-free
    /// blocks count as available: allocating them evicts their cache entry.
    #[inline]
    pub fn available(&self) -> usize {
        self.num_free
    }

    /// Total physical block capacity (free + in-use), constant after `new`.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.ref_counts.len()
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Allocate a single physical block (ref_count = 1). Returns `None` if
    /// the pool is exhausted (every block referenced). Evicts the block's
    /// prefix-cache entry, if any, at this point — the lazy eviction moment.
    pub fn allocate(&mut self) -> Option<u32> {
        while let Some((block_id, stamp)) = self.free_blocks.pop_front() {
            let idx = block_id as usize;
            // Skip stale entries: revived (ref > 0) or superseded by a newer push.
            if self.free_stamp[idx] != stamp || self.ref_counts[idx] != 0 {
                continue;
            }
            if let Some(hash) = self.block_hash[idx].take() {
                self.hash_to_block.remove(&hash);
            }
            self.ref_counts[idx] = 1;
            self.num_free -= 1;
            return Some(block_id);
        }
        None
    }

    /// Allocate enough blocks for `num_tokens` tokens (each with ref_count = 1).
    /// Returns the block table, or `None` if not enough blocks available.
    pub fn allocate_for_tokens(&mut self, num_tokens: usize) -> Option<Vec<u32>> {
        let needed = num_tokens.div_ceil(self.block_size);
        if needed > self.num_free {
            return None;
        }
        let mut table = Vec::with_capacity(needed);
        for _ in 0..needed {
            let Some(block_id) = self.allocate() else {
                self.free(&table);
                return None;
            };
            table.push(block_id);
        }
        Some(table)
    }

    /// Increment ref counts on a set of block IDs.
    /// Used when a request attaches to blocks it did not allocate itself.
    pub fn increment_refs(&mut self, block_ids: &[u32]) {
        for &id in block_ids {
            let idx = id as usize;
            if self.ref_counts[idx] == 0 {
                self.num_free -= 1;
            }
            self.ref_counts[idx] += 1;
        }
    }

    /// Decrement ref counts. Blocks reaching ref_count=0 enter the free LRU
    /// queue (keeping their prefix-cache hash, if any).
    pub fn decrement_refs(&mut self, block_ids: &[u32]) {
        for &id in block_ids {
            self.release_one(id);
        }
    }

    #[inline]
    fn release_one(&mut self, id: u32) {
        let idx = id as usize;
        let rc = &mut self.ref_counts[idx];
        *rc = rc.saturating_sub(1);
        if *rc == 0 {
            self.next_stamp += 1;
            self.free_stamp[idx] = self.next_stamp;
            self.free_blocks.push_back((id, self.next_stamp));
            self.num_free += 1;
        }
    }

    /// Free all blocks in a sequence's block table (decrements ref counts).
    ///
    /// Blocks are released in REVERSE table order so that chain tails enter
    /// the free queue (and are therefore evicted) before their parents —
    /// preserving the usefulness of cached prefixes (mirrors vLLM).
    pub fn free(&mut self, block_table: &[u32]) {
        for &id in block_table.iter().rev() {
            self.release_one(id);
        }
    }

    /// Ref count for a block. With the lazy prefix cache, `0` means free
    /// (possibly still matchable via its hash); `>= 1` means live in one or
    /// more running sequences.
    #[inline]
    pub fn ref_count(&self, block_id: u32) -> u32 {
        self.ref_counts[block_id as usize]
    }

    // ── Lazy prefix cache ───────────────────────────────────────────────

    /// Walk a chained-hash sequence and return the block IDs of the longest
    /// cached prefix, taking a reference on each matched block (reviving
    /// free blocks). Stops at the first miss. The returned blocks are owned
    /// by the caller (release with `free`/`decrement_refs` on abort).
    pub fn lookup_and_touch_prefix(&mut self, hashes: &[u64]) -> Vec<u32> {
        let mut blocks = Vec::new();
        for hash in hashes {
            let Some(&block_id) = self.hash_to_block.get(hash) else {
                break;
            };
            let idx = block_id as usize;
            debug_assert_eq!(self.block_hash[idx], Some(*hash));
            if self.ref_counts[idx] == 0 {
                // Revive from the free queue (stale entry skipped by stamp).
                self.num_free -= 1;
            }
            self.ref_counts[idx] += 1;
            blocks.push(block_id);
        }
        blocks
    }

    /// Register content hashes for full blocks of a sequence. `pairs` maps
    /// hash → block_id, in prefix-chain order. First writer wins: a hash
    /// already cached (by any block) is skipped, as is a block that already
    /// carries a hash. Returns the number of newly cached blocks.
    pub fn assign_block_hashes(&mut self, pairs: &[(u64, u32)]) -> usize {
        let mut newly_cached = 0;
        for &(hash, block_id) in pairs {
            let idx = block_id as usize;
            if self.hash_to_block.contains_key(&hash) || self.block_hash[idx].is_some() {
                continue;
            }
            self.block_hash[idx] = Some(hash);
            self.hash_to_block.insert(hash, block_id);
            newly_cached += 1;
        }
        newly_cached
    }

    /// Number of hash entries currently matchable (cached blocks, live or free).
    #[inline]
    pub fn cached_hashes(&self) -> usize {
        self.hash_to_block.len()
    }

    /// Compute the slot index for a token at position `pos` within a block table.
    ///
    /// `slot = block_table[pos / block_size] * block_size + (pos % block_size)`
    #[inline]
    pub fn slot(block_table: &[u32], pos: usize, block_size: usize) -> i64 {
        let block_idx = pos / block_size;
        let offset = pos % block_size;
        (block_table[block_idx] as i64) * (block_size as i64) + (offset as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let mut bm = BlockManager::new(4, 16);
        assert_eq!(bm.available(), 4);

        let b = bm.allocate().unwrap();
        assert_eq!(bm.available(), 3);
        assert_eq!(bm.ref_count(b), 1);

        bm.free(&[b]);
        assert_eq!(bm.available(), 4);
        assert_eq!(bm.ref_count(b), 0);
    }

    #[test]
    fn test_allocate_for_tokens() {
        let mut bm = BlockManager::new(8, 16);
        // 33 tokens → ceil(33/16) = 3 blocks
        let table = bm.allocate_for_tokens(33).unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(bm.available(), 5);

        bm.free(&table);
        assert_eq!(bm.available(), 8);
    }

    #[test]
    fn test_exhaustion() {
        let mut bm = BlockManager::new(2, 16);
        let _ = bm.allocate_for_tokens(32).unwrap(); // uses both blocks
        assert!(bm.allocate().is_none());
        assert!(bm.allocate_for_tokens(1).is_none());
    }

    #[test]
    fn test_slot_computation() {
        let table = vec![5, 10, 3];
        let block_size = 16;
        assert_eq!(BlockManager::slot(&table, 0, block_size), 80);
        assert_eq!(BlockManager::slot(&table, 15, block_size), 95);
        assert_eq!(BlockManager::slot(&table, 16, block_size), 160);
        assert_eq!(BlockManager::slot(&table, 33, block_size), 49);
    }

    #[test]
    fn test_ref_counting() {
        let mut bm = BlockManager::new(4, 16);
        let table = bm.allocate_for_tokens(32).unwrap(); // 2 blocks
        assert_eq!(bm.available(), 2);

        bm.increment_refs(&table);
        assert_eq!(bm.ref_count(table[0]), 2);
        assert_eq!(bm.ref_count(table[1]), 2);

        bm.free(&table);
        assert_eq!(bm.available(), 2); // still 2 free (not freed yet)
        assert_eq!(bm.ref_count(table[0]), 1);

        bm.decrement_refs(&table);
        assert_eq!(bm.available(), 4);
        assert_eq!(bm.ref_count(table[0]), 0);
    }

    #[test]
    fn test_shared_prefix_blocks() {
        let mut bm = BlockManager::new(8, 16);
        let table1 = bm.allocate_for_tokens(32).unwrap();
        assert_eq!(bm.available(), 6);

        bm.increment_refs(&table1[..1]);
        assert_eq!(bm.ref_count(table1[0]), 2);

        bm.free(&table1);
        assert_eq!(bm.available(), 7);
        assert_eq!(bm.ref_count(table1[0]), 1);
        assert_eq!(bm.ref_count(table1[1]), 0);

        bm.increment_refs(&table1[..1]);
        assert_eq!(bm.ref_count(table1[0]), 2);

        bm.decrement_refs(&table1[..1]);
        assert_eq!(bm.ref_count(table1[0]), 1);
    }

    // ── Lazy prefix cache ───────────────────────────────────────────────

    #[test]
    fn test_lazy_cache_hit_on_free_block_revives() {
        let mut bm = BlockManager::new(4, 16);
        let table = bm.allocate_for_tokens(32).unwrap(); // blocks for 2 full pages
        assert_eq!(bm.assign_block_hashes(&[(11, table[0]), (22, table[1])]), 2);

        // Request ends: blocks free but still cached.
        bm.free(&table);
        assert_eq!(bm.available(), 4);
        assert_eq!(bm.cached_hashes(), 2);

        // New request with the same prefix revives both blocks.
        let hit = bm.lookup_and_touch_prefix(&[11, 22, 33]);
        assert_eq!(hit, table);
        assert_eq!(bm.ref_count(table[0]), 1);
        assert_eq!(bm.available(), 2);

        bm.free(&hit);
        assert_eq!(bm.available(), 4);
    }

    #[test]
    fn test_lazy_eviction_only_on_reuse() {
        let mut bm = BlockManager::new(2, 16);
        let table = bm.allocate_for_tokens(32).unwrap();
        bm.assign_block_hashes(&[(11, table[0]), (22, table[1])]);
        bm.free(&table); // tail-first: free order is table[1], table[0]

        // Pool "full of cache": allocation must still succeed by evicting.
        let b = bm.allocate().unwrap();
        // Tail freed first → tail (table[1]) is at the queue head → evicted first.
        assert_eq!(b, table[1]);
        assert_eq!(bm.cached_hashes(), 1);
        // The parent block (table[0]) is still matchable.
        assert_eq!(bm.lookup_and_touch_prefix(&[11, 22]), vec![table[0]]);
    }

    #[test]
    fn test_lazy_match_stops_at_first_miss() {
        let mut bm = BlockManager::new(4, 16);
        let table = bm.allocate_for_tokens(48).unwrap();
        // Cache blocks 0 and 2 but not 1: match must stop after block 0.
        bm.assign_block_hashes(&[(11, table[0]), (33, table[2])]);
        let hit = bm.lookup_and_touch_prefix(&[11, 22, 33]);
        assert_eq!(hit, vec![table[0]]);
        bm.decrement_refs(&hit);
        bm.free(&table);
    }

    #[test]
    fn test_lazy_stale_queue_entries_skipped() {
        let mut bm = BlockManager::new(2, 16);
        let t = bm.allocate_for_tokens(32).unwrap();
        bm.assign_block_hashes(&[(11, t[0]), (22, t[1])]);
        bm.free(&t);
        // Revive both (stale queue entries remain), then free again.
        let hit = bm.lookup_and_touch_prefix(&[11, 22]);
        assert_eq!(hit.len(), 2);
        bm.free(&hit);
        // Allocate both: stale entries must be skipped, each block returned once.
        let a = bm.allocate().unwrap();
        let b = bm.allocate().unwrap();
        assert_ne!(a, b);
        assert!(bm.allocate().is_none());
        assert_eq!(bm.cached_hashes(), 0);
    }

    #[test]
    fn test_duplicate_hash_first_writer_wins() {
        let mut bm = BlockManager::new(4, 16);
        let t1 = bm.allocate_for_tokens(16).unwrap();
        let t2 = bm.allocate_for_tokens(16).unwrap();
        assert_eq!(bm.assign_block_hashes(&[(11, t1[0])]), 1);
        // Same content prefilled by a concurrent duplicate request.
        assert_eq!(bm.assign_block_hashes(&[(11, t2[0])]), 0);
        assert_eq!(bm.lookup_and_touch_prefix(&[11]), vec![t1[0]]);
        bm.decrement_refs(&[t1[0]]);
        bm.free(&t1);
        bm.free(&t2);
    }
}
