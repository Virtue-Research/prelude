use std::collections::HashMap;

use crate::cache::prefix_index::PrefixMatchIndex;

/// Block-aligned prefix boundary identified by the existing chained block hash.
///
/// A matching `(blocks, hash)` means all complete blocks up to this boundary
/// match, modulo hash collision risk already accepted by `PrefixMatchIndex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefixBoundary {
    pub blocks: usize,
    pub tokens: usize,
    pub hash: u64,
}

/// Cache resources available at a prefix boundary.
///
/// Attention-only models need paged KV. Hybrid models additionally need the
/// recurrent-state snapshot at the same boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixResources {
    pub paged_kv: bool,
    pub recurrent_state: bool,
}

impl PrefixResources {
    #[inline]
    pub fn paged_only() -> Self {
        Self {
            paged_kv: true,
            recurrent_state: false,
        }
    }

    #[inline]
    pub fn paged_with_recurrent_state() -> Self {
        Self {
            paged_kv: true,
            recurrent_state: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedPrefixAssignment {
    pub request_index: usize,
    pub boundary: PrefixBoundary,
    pub peers: usize,
    pub saved_tokens: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct SharedPrefixPrompt<'a> {
    pub tokens: &'a [u32],
    /// Ignore boundaries at or before this token offset. Running prefills use
    /// this to avoid waiting for a checkpoint that can no longer be materialized.
    pub min_boundary_tokens: usize,
}

/// Finds useful block-aligned shared prefix boundaries without building a
/// token-level radix tree.
pub struct SharedPrefixPlanner {
    block_size: usize,
    min_peers: usize,
}

impl SharedPrefixPlanner {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
            min_peers: 2,
        }
    }

    /// Return every reusable boundary in prompt order, leaving at least one
    /// suffix token for the forward pass.
    pub fn boundaries_for_tokens(&self, tokens: &[u32]) -> Vec<PrefixBoundary> {
        self.boundaries_for_prompt(SharedPrefixPrompt {
            tokens,
            min_boundary_tokens: 0,
        })
    }

    pub fn boundaries_for_prompt(&self, prompt: SharedPrefixPrompt<'_>) -> Vec<PrefixBoundary> {
        let tokens = prompt.tokens;
        if tokens.len() < self.block_size + 1 {
            return Vec::new();
        }

        let full_blocks = (tokens.len() - 1) / self.block_size;
        let mut out = Vec::with_capacity(full_blocks);
        let mut parent_hash = 0u64;
        for (block_idx, block_tokens) in
            tokens.chunks(self.block_size).take(full_blocks).enumerate()
        {
            if block_tokens.len() < self.block_size {
                break;
            }
            parent_hash = PrefixMatchIndex::hash_block(parent_hash, block_tokens);
            let blocks = block_idx + 1;
            let boundary = PrefixBoundary {
                blocks,
                tokens: blocks * self.block_size,
                hash: parent_hash,
            };
            if boundary.tokens > prompt.min_boundary_tokens {
                out.push(boundary);
            }
        }
        out
    }

    /// Assign each request to the deepest boundary shared with at least
    /// `min_peers` prompts.
    pub fn assign<'a, I>(&self, prompts: I) -> Vec<Option<SharedPrefixAssignment>>
    where
        I: IntoIterator<Item = &'a [u32]>,
    {
        let prompts: Vec<SharedPrefixPrompt<'a>> = prompts
            .into_iter()
            .map(|tokens| SharedPrefixPrompt {
                tokens,
                min_boundary_tokens: 0,
            })
            .collect();
        self.assign_prompts(&prompts)
    }

    pub fn assign_prompts(
        &self,
        prompts: &[SharedPrefixPrompt<'_>],
    ) -> Vec<Option<SharedPrefixAssignment>> {
        let boundaries: Vec<Vec<PrefixBoundary>> = prompts
            .iter()
            .map(|&prompt| self.boundaries_for_prompt(prompt))
            .collect();

        let mut counts: HashMap<PrefixBoundary, usize> = HashMap::new();
        for request_boundaries in &boundaries {
            for &boundary in request_boundaries {
                *counts.entry(boundary).or_insert(0) += 1;
            }
        }

        boundaries
            .iter()
            .enumerate()
            .map(|(request_index, request_boundaries)| {
                request_boundaries.iter().rev().find_map(|&boundary| {
                    let peers = *counts.get(&boundary)?;
                    if peers < self.min_peers {
                        return None;
                    }
                    Some(SharedPrefixAssignment {
                        request_index,
                        boundary,
                        peers,
                        saved_tokens: boundary.tokens * peers.saturating_sub(1),
                    })
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_returns_block_aligned_boundaries_with_suffix() {
        let planner = SharedPrefixPlanner::new(4);
        let boundaries = planner.boundaries_for_tokens(&(0..12).collect::<Vec<_>>());

        assert_eq!(boundaries.len(), 2);
        assert_eq!(boundaries[0].tokens, 4);
        assert_eq!(boundaries[1].tokens, 8);
    }

    #[test]
    fn planner_assigns_deepest_shared_boundary() {
        let planner = SharedPrefixPlanner::new(4);
        let a: Vec<u32> = (0..13).collect();
        let mut b: Vec<u32> = (0..12).collect();
        b.push(100);
        let c: Vec<u32> = (200..213).collect();

        let assignments = planner.assign([a.as_slice(), b.as_slice(), c.as_slice()]);

        assert_eq!(assignments[0].as_ref().unwrap().boundary.tokens, 12);
        assert_eq!(assignments[1].as_ref().unwrap().boundary.tokens, 12);
        assert!(assignments[2].is_none());
    }

    #[test]
    fn planner_ignores_boundaries_already_crossed_by_running_prompt() {
        let planner = SharedPrefixPlanner::new(4);
        let a: Vec<u32> = (0..17).collect();
        let b: Vec<u32> = (0..17).collect();

        let assignments = planner.assign_prompts(&[
            SharedPrefixPrompt {
                tokens: a.as_slice(),
                min_boundary_tokens: 12,
            },
            SharedPrefixPrompt {
                tokens: b.as_slice(),
                min_boundary_tokens: 0,
            },
        ]);

        assert_eq!(assignments[0].as_ref().unwrap().boundary.tokens, 16);
        assert_eq!(assignments[1].as_ref().unwrap().boundary.tokens, 16);
    }

    #[test]
    fn planner_uses_closest_deeper_group_over_wider_shallow_group() {
        let planner = SharedPrefixPlanner::new(4);
        let a: Vec<u32> = (0..17).collect();
        let b: Vec<u32> = (0..17).collect();
        let mut c: Vec<u32> = (0..9).collect();
        c.extend(100..108);

        let assignments = planner.assign([a.as_slice(), b.as_slice(), c.as_slice()]);

        assert_eq!(assignments[0].as_ref().unwrap().boundary.tokens, 16);
        assert_eq!(assignments[1].as_ref().unwrap().boundary.tokens, 16);
        assert_eq!(assignments[2].as_ref().unwrap().boundary.tokens, 8);
    }
}
