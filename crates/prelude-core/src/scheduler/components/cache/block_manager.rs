//! Physical block manager for paged KV cache with ref counting.
//!
//! Manages a fixed pool of GPU KV cache blocks. Each block holds
//! `block_size` tokens worth of key/value data. Sequences are assigned
//! blocks on demand and freed when complete.
//!
//! Ref counting enables shared blocks between active sequences and the
//! prefix cache — blocks are only freed when all references are released.

use std::collections::VecDeque;

pub struct BlockManager {
    block_size: usize,
    free_blocks: VecDeque<u32>,
    ref_counts: Vec<u32>,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let free_blocks: VecDeque<u32> = (0..num_blocks as u32).collect();
        let ref_counts = vec![0u32; num_blocks];
        Self {
            block_size,
            free_blocks,
            ref_counts,
        }
    }

    /// Number of free blocks available.
    #[inline]
    pub fn available(&self) -> usize {
        self.free_blocks.len()
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Allocate a single physical block (ref_count = 1). Returns `None` if pool exhausted.
    pub fn allocate(&mut self) -> Option<u32> {
        let block_id = self.free_blocks.pop_front()?;
        self.ref_counts[block_id as usize] = 1;
        Some(block_id)
    }

    /// Allocate enough blocks for `num_tokens` tokens (each with ref_count = 1).
    /// Returns the block table, or `None` if not enough blocks available.
    pub fn allocate_for_tokens(&mut self, num_tokens: usize) -> Option<Vec<u32>> {
        let needed = num_tokens.div_ceil(self.block_size);
        if needed > self.free_blocks.len() {
            return None;
        }
        let mut table = Vec::with_capacity(needed);
        for _ in 0..needed {
            let block_id = self.free_blocks.pop_front().unwrap();
            self.ref_counts[block_id as usize] = 1;
            table.push(block_id);
        }
        Some(table)
    }

    /// Increment ref counts on a set of block IDs.
    /// Used when the prefix cache takes a reference to active blocks.
    pub fn increment_refs(&mut self, block_ids: &[u32]) {
        for &id in block_ids {
            self.ref_counts[id as usize] += 1;
        }
    }

    /// Decrement ref counts. Blocks reaching ref_count=0 are returned to the free pool.
    pub fn decrement_refs(&mut self, block_ids: &[u32]) {
        for &id in block_ids {
            let rc = &mut self.ref_counts[id as usize];
            *rc = rc.saturating_sub(1);
            if *rc == 0 {
                self.free_blocks.push_back(id);
            }
        }
    }

    /// Free all blocks in a sequence's block table (decrements ref counts).
    pub fn free(&mut self, block_table: &[u32]) {
        self.decrement_refs(block_table);
    }

    /// Get the ref count for a block (for debugging/testing).
    #[cfg(test)]
    pub fn ref_count(&self, block_id: u32) -> u32 {
        self.ref_counts[block_id as usize]
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
}
