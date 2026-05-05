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
        let Some(pool) = self.cache.paged_pool.as_ref() else {
            return Ok(ResolvedPrefixReuse::default());
        };

        let block_ok = if self.executor.config.head_dim == 256 {
            pool.block_size % 64 == 0
        } else {
            pool.block_size % 128 == 0
        };
        if !block_ok {
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
        let mut bm = bm_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;

        for entry in &allocation_plan.entries {
            let mut bt = Vec::with_capacity(entry.total_blocks);
            if !shared_blocks.is_empty() {
                bt.extend_from_slice(shared_blocks);
                bm.increment_refs(shared_blocks);
            }

            for _ in 0..entry.new_blocks {
                let block = bm
                    .allocate()
                    .ok_or_else(|| EngineError::Internal(format!("{context}: no free blocks")))?;
                bt.push(block);
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
