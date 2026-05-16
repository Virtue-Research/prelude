use std::hash::{Hash, Hasher};

use crate::cache::deltanet_pool::DeltaNetPrefixState;
use crate::engine::{BatchPrefillResult, Engine, EngineError};
use crate::scheduler::{Scheduler, Sequence};

const PREFIX_CACHE_KEY_BLOCKS: usize = 8;

struct PrefixAttach {
    deltanet_state: Option<DeltaNetPrefixState>,
    replaced_blocks: Vec<u32>,
}

fn attach_prefix_cache_reuse(engine: &Engine, seq: &mut Sequence) -> Option<PrefixAttach> {
    if engine.cache.prefix_cache.is_none() || seq.input_ids.len() <= 1 {
        return None;
    }
    let Some(pool) = engine.cache.paged_pool.as_ref() else {
        return None;
    };

    let (cached_len, cached_blocks, deltanet_state) = match if engine.cache.deltanet_pool.is_some()
    {
        engine
            .cache
            .try_prefix_cache_match_paged_with_deltanet_state(&seq.input_ids)
    } else {
        engine
            .cache
            .try_prefix_cache_match_paged_only(&seq.input_ids)
            .map(|(len, blocks)| (len, blocks, None))
    } {
        Ok(hit) => hit,
        Err(error) => {
            tracing::warn!(request_id = %seq.request_id, error = %error, "prefix cache match failed");
            return None;
        }
    };

    if cached_len == 0 || cached_len <= seq.kv_computed_len || cached_len >= seq.input_ids.len() {
        return None;
    }

    let block_size = pool.block_size.max(1);
    let full_shared_blocks = cached_len / block_size;
    let partial_tokens = cached_len % block_size;
    let mut retained_blocks = Vec::new();
    let mut block_table = Vec::new();
    let mut private_partial_block = None;

    if partial_tokens == 0 {
        if full_shared_blocks == 0 || cached_blocks.len() < full_shared_blocks {
            return None;
        }
        retained_blocks.extend_from_slice(&cached_blocks[..full_shared_blocks]);
        block_table.extend_from_slice(&retained_blocks);
    } else {
        if cached_blocks.len() < full_shared_blocks + 1 {
            return None;
        }
        retained_blocks.extend_from_slice(&cached_blocks[..full_shared_blocks]);
        block_table.extend_from_slice(&retained_blocks);

        let source_partial_block = cached_blocks[full_shared_blocks];
        let private_block = match engine.cache.allocate_paged_block() {
            Ok(Some(block)) => block,
            Ok(None) => {
                tracing::warn!(request_id = %seq.request_id, "prefix cache partial-page copy skipped: no free KV block");
                return None;
            }
            Err(error) => {
                tracing::warn!(request_id = %seq.request_id, error = %error, "prefix cache partial-page block allocation failed");
                return None;
            }
        };
        if let Err(error) = engine
            .cache
            .copy_paged_kv_block(source_partial_block, private_block)
        {
            let _ = engine.cache.release_paged_blocks(&[private_block]);
            tracing::warn!(request_id = %seq.request_id, error = %error, "prefix cache partial-page KV copy failed");
            return None;
        }
        private_partial_block = Some(private_block);
        block_table.push(private_block);
    }

    if block_table.is_empty() {
        if let Some(block) = private_partial_block {
            let _ = engine.cache.release_paged_blocks(&[block]);
        }
        return None;
    }

    if let Err(error) = engine.cache.retain_paged_blocks(&retained_blocks) {
        tracing::warn!(request_id = %seq.request_id, error = %error, "prefix cache block retain failed");
        if let Some(block) = private_partial_block {
            let _ = engine.cache.release_paged_blocks(&[block]);
        }
        return None;
    }

    seq.kv_computed_len = cached_len;
    let replaced_blocks = std::mem::replace(&mut seq.block_table, block_table);
    tracing::debug!(
        request_id = %seq.request_id,
        cached_len,
        cached_blocks = seq.block_table.len(),
        partial_tokens,
        suffix_len = seq.input_ids.len() - cached_len,
        "attached AR prefix-cache reuse"
    );
    Some(PrefixAttach {
        deltanet_state,
        replaced_blocks,
    })
}

pub(super) fn prefix_cache_key(engine: &Engine, tokens: &[u32]) -> Option<u64> {
    if engine.cache.prefix_cache.is_none() {
        return None;
    }
    let block_size = engine.cache.paged_pool.as_ref()?.block_size;
    // Group by the first several full pages. One page is too coarse for
    // chat templates shared across unrelated policies; several pages capture
    // the policy/system prompt while still avoiding full-prompt hashing.
    let key_tokens = (block_size * PREFIX_CACHE_KEY_BLOCKS).min(tokens.len());
    if key_tokens < block_size {
        return None;
    }
    let key_tokens = key_tokens - (key_tokens % block_size);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    block_size.hash(&mut hasher);
    tokens[..key_tokens].hash(&mut hasher);
    Some(hasher.finish())
}

fn should_insert_deltanet_prefix_state(seq: &Sequence, block_size: usize, computed: usize) -> bool {
    if block_size == 0
        || computed == 0
        || computed >= seq.input_ids.len()
        || computed % block_size != 0
        || seq.prefix_cache_key.is_none()
    {
        return false;
    }

    if seq.prefix_cache_target_len == Some(computed) {
        return true;
    }

    if computed == bootstrap_deltanet_prefix_len(seq.input_ids.len(), block_size) {
        return true;
    }

    computed == final_reusable_deltanet_prefix_len(seq.input_ids.len(), block_size)
}

fn bootstrap_deltanet_prefix_len(prompt_len: usize, block_size: usize) -> usize {
    if block_size == 0 {
        return 0;
    }
    let key_tokens = (block_size * PREFIX_CACHE_KEY_BLOCKS).min(prompt_len);
    let key_tokens = key_tokens - (key_tokens % block_size);
    if key_tokens >= block_size && key_tokens < prompt_len {
        key_tokens
    } else {
        0
    }
}

fn final_reusable_deltanet_prefix_len(prompt_len: usize, block_size: usize) -> usize {
    if block_size == 0 || prompt_len <= block_size {
        return 0;
    }
    let aligned = prompt_len - (prompt_len % block_size);
    if aligned == prompt_len {
        aligned.saturating_sub(block_size)
    } else {
        aligned
    }
}

pub(super) fn refresh_waiting_prefix_cache(engine: &Engine, scheduler: &mut Scheduler) {
    if engine.cache.prefix_cache.is_none() {
        return;
    }

    let planned = scheduler.plan_waiting_shared_prefixes();
    if planned > 0 {
        tracing::debug!(planned, "planned shared prefix-cache boundaries");
    }

    let mut refreshed = 0usize;
    let mut blocks_to_release: Vec<Vec<u32>> = Vec::new();
    scheduler.for_each_waiting_mut(|seq| {
        if seq.kv_computed_len > 0
            && seq
                .prefix_cache_target_len
                .is_none_or(|target_len| seq.kv_computed_len >= target_len)
        {
            return;
        }
        let Some(attach) = attach_prefix_cache_reuse(engine, seq) else {
            return;
        };
        if !attach.replaced_blocks.is_empty() {
            blocks_to_release.push(attach.replaced_blocks);
        }
        if let Some(state) = attach.deltanet_state {
            let restored_slot = if let Some(slot) = seq.deltanet_slot {
                engine.cache.deltanet_pool.as_ref().and_then(|pool_mutex| {
                    pool_mutex
                        .lock()
                        .ok()
                        .and_then(|pool| match pool.restore_slot(slot, &state) {
                            Ok(()) => Some(slot),
                            Err(error) => {
                                tracing::warn!(request_id = %seq.request_id, slot, error = %error, "DeltaNet prefix state restore failed");
                                None
                            }
                        })
                })
            } else {
                engine.cache.deltanet_pool.as_ref().and_then(|pool_mutex| {
                    pool_mutex.lock().ok().and_then(|mut pool| {
                        let slot = pool.allocate()?;
                        if let Err(error) = pool.restore_slot(slot, &state) {
                            tracing::warn!(request_id = %seq.request_id, slot, error = %error, "DeltaNet prefix state restore failed");
                            pool.free(slot);
                            return None;
                        }
                        Some(slot)
                    })
                })
            };
            if let Some(slot) = restored_slot {
                seq.deltanet_slot = Some(slot);
            } else {
                seq.kv_computed_len = 0;
                blocks_to_release.push(std::mem::take(&mut seq.block_table));
                return;
            }
        }
        if seq.kv_computed_len > 0 {
            refreshed += 1;
        }
    });
    for blocks in blocks_to_release {
        scheduler.free_blocks(&blocks);
    }

    if refreshed > 0 {
        tracing::debug!(refreshed, "refreshed waiting prefix-cache reuse");
    }
}

pub(super) fn try_insert_prefill_prefix_cache(
    engine: &Engine,
    scheduler: &Scheduler,
    request_id: &str,
    computed: usize,
    is_final: bool,
    prefill_result: Option<&BatchPrefillResult>,
) {
    if engine.cache.prefix_cache.is_none() {
        return;
    }
    let (Some(seq), Some(pool), Some(result)) = (
        scheduler.get_sequence(request_id),
        engine.cache.paged_pool.as_ref(),
        prefill_result,
    ) else {
        return;
    };
    if result.block_table.is_empty() || seq.input_ids.is_empty() {
        return;
    }

    if let Some(dn_pool_mutex) = engine.cache.deltanet_pool.as_ref() {
        if computed == 0 || computed >= seq.input_ids.len() {
            return;
        }
        if computed % pool.block_size != 0 {
            tracing::debug!(
                request_id = %request_id,
                computed,
                block_size = pool.block_size,
                "skip hybrid prefix cache insert at non-block boundary"
            );
            return;
        }
        if !should_insert_deltanet_prefix_state(seq, pool.block_size, computed) {
            tracing::debug!(
                request_id = %request_id,
                computed,
                prompt_len = seq.input_ids.len(),
                target_len = ?seq.prefix_cache_target_len,
                "skip hybrid prefix cache insert at unplanned boundary"
            );
            return;
        }
        let Some(slot) = seq.deltanet_slot else {
            return;
        };
        let deltanet_state = match dn_pool_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("deltanet pool lock: {e}")))
            .and_then(|pool| {
                pool.snapshot_slot(slot).map_err(|e| {
                    EngineError::Internal(format!("DeltaNet prefix state snapshot failed: {e}"))
                })
            }) {
            Ok(state) => state,
            Err(e) => {
                tracing::warn!(error = %e, "hybrid prefix state snapshot failed");
                return;
            }
        };
        if let Err(e) = engine
            .cache
            .try_prefix_cache_insert_paged_with_deltanet_state(
                &seq.input_ids[..computed],
                &result.block_table,
                pool.block_size,
                deltanet_state,
            )
        {
            tracing::warn!("hybrid prefix cache insert failed: {e}");
        }
        return;
    }

    // Pure paged-attention path (no DeltaNet). Besides the final whole-prompt
    // insert, publish the *planned shared prefix* as soon as the leader has
    // prefilled through it — without waiting for is_final. When many requests
    // arrive with the same long system prompt, plan_waiting_shared_prefixes()
    // elects one leader and parks the peers; with an is_final-only insert the
    // peers stay parked until the leader also finishes its own (possibly very
    // long) suffix. Publishing at the shared boundary lets the peers attach
    // and overlap their suffix prefill with the leader's instead of idling.
    // Only the block-aligned shared portion is cached (not the leader's
    // private suffix); the later is_final whole-prompt insert is an idempotent
    // superset in the block-hash prefix trie.
    let insert_tokens: Option<&[u32]> = if is_final {
        Some(seq.input_ids.as_slice())
    } else if let Some(target) = seq.prefix_cache_target_len {
        if seq.prefix_cache_key.is_some()
            && target >= pool.block_size
            && target < seq.input_ids.len()
            && computed >= target
        {
            Some(&seq.input_ids[..target])
        } else {
            None
        }
    } else {
        None
    };
    if let Some(tokens) = insert_tokens
        && let Err(e) = engine.cache.try_prefix_cache_insert_paged_only(
            tokens,
            &result.block_table,
            pool.block_size,
        )
    {
        tracing::warn!("prefix cache insert (ar_loop) failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::SamplingParams;

    fn seq_with_prefix(len: usize, target_len: Option<usize>) -> Sequence {
        let mut seq = Sequence::new(
            "r".to_string(),
            (0..len as u32).collect(),
            SamplingParams::default(),
            1,
            Vec::new(),
            Vec::new(),
            None,
        );
        seq.prefix_cache_key = Some(1);
        seq.prefix_cache_target_len = target_len;
        seq
    }

    #[test]
    fn deltanet_prefix_state_insert_accepts_planned_boundary() {
        let seq = seq_with_prefix(33, Some(16));
        assert!(should_insert_deltanet_prefix_state(&seq, 4, 16));
    }

    #[test]
    fn deltanet_prefix_state_insert_accepts_bootstrap_boundary() {
        let seq = seq_with_prefix(65, None);
        assert!(should_insert_deltanet_prefix_state(&seq, 4, 32));
    }

    #[test]
    fn deltanet_prefix_state_insert_accepts_final_reusable_boundary() {
        let seq = seq_with_prefix(33, None);
        assert!(should_insert_deltanet_prefix_state(&seq, 4, 32));

        let exact_block_prompt = seq_with_prefix(32, None);
        assert!(should_insert_deltanet_prefix_state(
            &exact_block_prompt,
            4,
            28
        ));
    }

    #[test]
    fn deltanet_prefix_state_insert_rejects_unplanned_intermediate_boundary() {
        let seq = seq_with_prefix(33, None);
        assert!(!should_insert_deltanet_prefix_state(&seq, 4, 8));
        assert!(!should_insert_deltanet_prefix_state(&seq, 4, 33));
        assert!(!should_insert_deltanet_prefix_state(&seq, 4, 30));

        let mut no_key = seq_with_prefix(33, Some(16));
        no_key.prefix_cache_key = None;
        assert!(!should_insert_deltanet_prefix_state(&no_key, 4, 16));
    }
}
