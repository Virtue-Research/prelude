use crate::cache::manager::CacheManager;
use crate::engine::*;

fn build_cache_allocation_entries(
    seq_lens: &[usize],
    cached_len: usize,
    shared_block_count: usize,
    block_size: usize,
) -> Result<(Vec<CacheAllocationPlanEntry>, usize), EngineError> {
    let mut entries = Vec::with_capacity(seq_lens.len());
    let mut max_total_blocks = 0usize;

    for &prompt_len in seq_lens {
        if prompt_len < cached_len {
            return Err(EngineError::Internal(
                "cache allocation plan received prompt shorter than cached prefix".into(),
            ));
        }

        let total_blocks = prompt_len.div_ceil(block_size);
        if total_blocks < shared_block_count {
            return Err(EngineError::Internal(
                "cache allocation plan produced fewer total blocks than shared prefix blocks"
                    .into(),
            ));
        }

        let entry = CacheAllocationPlanEntry {
            prompt_len,
            suffix_len: prompt_len - cached_len,
            total_blocks,
            new_blocks: total_blocks - shared_block_count,
        };
        max_total_blocks = max_total_blocks.max(entry.total_blocks);
        entries.push(entry);
    }

    Ok((entries, max_total_blocks))
}

impl Engine {
    pub(crate) fn build_prefix_reuse_candidate(
        &self,
        items: &[PreparedGenerateRequest],
        seq_lens: &[usize],
    ) -> Option<PrefixReuseCandidate> {
        if self.cache.prefix_cache.is_none() {
            return None;
        }

        let common_prefix = CacheManager::find_common_prefix(items);
        if common_prefix.is_empty() {
            return None;
        }

        Some(PrefixReuseCandidate {
            common_prefix_tokens: common_prefix.to_vec(),
            min_prompt_len: seq_lens.iter().copied().min().unwrap_or(0),
        })
    }

    pub(crate) fn resolve_paged_prefix_reuse(
        &self,
        prefill_plan: &PrefillPlan,
    ) -> Result<ResolvedPrefixReuse, EngineError> {
        let Some(candidate) = prefill_plan.prefix_reuse.as_ref() else {
            return Ok(ResolvedPrefixReuse::default());
        };
        if self.cache.paged_pool.is_none() {
            return Ok(ResolvedPrefixReuse::default());
        }

        let (cached_len, cached_block_ids) = self
            .cache
            .try_prefix_cache_match_paged_only(&candidate.common_prefix_tokens)?;

        if cached_len == 0 || cached_block_ids.is_empty() || cached_len >= candidate.min_prompt_len
        {
            return Ok(ResolvedPrefixReuse::default());
        }

        Ok(ResolvedPrefixReuse {
            cached_len,
            cached_block_ids,
        })
    }

    pub(crate) fn build_cache_allocation_plan(
        &self,
        seq_lens: &[usize],
        prefix_reuse: &ResolvedPrefixReuse,
    ) -> Result<CacheAllocationPlan, EngineError> {
        let pool = self.cache.paged_pool.as_ref().ok_or_else(|| {
            EngineError::Internal("cache allocation plan requires paged attention pool".into())
        })?;

        let (entries, max_total_blocks) = build_cache_allocation_entries(
            seq_lens,
            prefix_reuse.cached_len,
            prefix_reuse.cached_block_ids.len(),
            pool.block_size,
        )?;

        Ok(CacheAllocationPlan {
            prefix_reuse: prefix_reuse.clone(),
            entries,
            max_total_blocks,
        })
    }

    pub(crate) fn allocate_block_tables_from_plan(
        &self,
        allocation_plan: &CacheAllocationPlan,
        context: &'static str,
    ) -> Result<Vec<Vec<u32>>, EngineError> {
        let bm_mutex = self.cache.block_manager.as_ref().ok_or_else(|| {
            EngineError::Internal(format!("{context}: block manager unavailable"))
        })?;
        let shared_blocks = &allocation_plan.prefix_reuse.cached_block_ids;
        let mut block_tables = Vec::with_capacity(allocation_plan.entries.len());

        // Reclaim idle (cache-only) prefix blocks to cover any shortfall BEFORE
        // taking the (non-reentrant) bm lock below — reclaim itself locks bm, so
        // it must run first. With the half-pool clamp gone this is what keeps the
        // prefix cache from pinning the pool and starving this allocation.
        //
        // CRITICAL: the matched shared-prefix blocks (`shared_blocks`) are still
        // cache-only (ref_count == 1) at this point — they are not pinned to this
        // request until the per-entry `increment_refs` in the loop below. Reclaim
        // would treat them as idle and could free the very chain we are about to
        // reuse (then `increment_refs` revives a freed id and `allocate()` could
        // alias it into another entry → KV corruption). So pin them across the
        // reclaim, then drop that one temporary reference once we hold bm; each
        // entry re-pins its own reference in the loop. (Unlike the AR-loop reuse
        // path, which retains at match time, the planner matches without pinning.)
        let new_total: usize = allocation_plan.entries.iter().map(|e| e.new_blocks).sum();
        if !shared_blocks.is_empty() {
            self.cache.retain_paged_blocks(shared_blocks)?;
        }
        if new_total > 0 {
            let available = {
                let bm = bm_mutex
                    .lock()
                    .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
                bm.available()
            };
            if available < new_total {
                if let Err(e) = self.cache.reclaim_idle_prefix_blocks(new_total - available) {
                    if !shared_blocks.is_empty() {
                        let _ = self.cache.release_paged_blocks(shared_blocks);
                    }
                    return Err(e);
                }
            }
        }

        let mut bm = bm_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
        if !shared_blocks.is_empty() {
            // Drop the temporary cross-reclaim pin; the loop below takes one
            // reference per reusing sequence (preserving the original accounting).
            bm.decrement_refs(shared_blocks);
        }

        // All-or-nothing: if the pool runs dry mid-loop (more likely now that the
        // cache may hold nearly the whole pool and reclaim can free < requested),
        // roll back every ref/allocation this call already committed before
        // returning Err — block tables are plain Vec with no Drop, so a partial
        // commit would otherwise leak shared-prefix refs and private blocks.
        let n_shared = shared_blocks.len();
        for entry in &allocation_plan.entries {
            let mut bt = Vec::with_capacity(entry.total_blocks);
            if !shared_blocks.is_empty() {
                bt.extend_from_slice(shared_blocks);
                bm.increment_refs(shared_blocks);
            }
            for _ in 0..entry.new_blocks {
                match bm.allocate() {
                    Some(block) => bt.push(block),
                    None => {
                        block_tables.push(bt); // include the partial table in the undo set
                        for t in &block_tables {
                            let k = n_shared.min(t.len());
                            bm.decrement_refs(&t[..k]); // undo this call's shared increment_refs
                            bm.free(&t[k..]); // free freshly-allocated private blocks
                        }
                        return Err(EngineError::Internal(format!("{context}: no free blocks")));
                    }
                }
            }
            block_tables.push(bt);
        }

        Ok(block_tables)
    }
}

#[cfg(test)]
mod tests {
    use super::build_cache_allocation_entries;
    use crate::engine::CacheAllocationPlanEntry;

    #[test]
    fn builds_entries_without_prefix_reuse() {
        let (entries, max_blocks) = build_cache_allocation_entries(&[8, 17], 0, 0, 16).unwrap();

        assert_eq!(
            entries,
            vec![
                CacheAllocationPlanEntry {
                    prompt_len: 8,
                    suffix_len: 8,
                    total_blocks: 1,
                    new_blocks: 1,
                },
                CacheAllocationPlanEntry {
                    prompt_len: 17,
                    suffix_len: 17,
                    total_blocks: 2,
                    new_blocks: 2,
                },
            ]
        );
        assert_eq!(max_blocks, 2);
    }

    #[test]
    fn builds_entries_with_prefix_reuse() {
        let (entries, max_blocks) =
            build_cache_allocation_entries(&[160, 192], 128, 2, 64).unwrap();

        assert_eq!(
            entries,
            vec![
                CacheAllocationPlanEntry {
                    prompt_len: 160,
                    suffix_len: 32,
                    total_blocks: 3,
                    new_blocks: 1,
                },
                CacheAllocationPlanEntry {
                    prompt_len: 192,
                    suffix_len: 64,
                    total_blocks: 3,
                    new_blocks: 1,
                },
            ]
        );
        assert_eq!(max_blocks, 3);
    }

    #[test]
    fn rejects_prompt_shorter_than_cached_prefix() {
        let err = build_cache_allocation_entries(&[64], 128, 2, 64).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("prompt shorter than cached prefix"));
    }
}
