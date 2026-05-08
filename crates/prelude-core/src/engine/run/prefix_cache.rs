use std::hash::{Hash, Hasher};

use crate::cache::deltanet_pool::DeltaNetPrefixState;
use crate::engine::{BatchPrefillResult, Engine, EngineError};
use crate::scheduler::{Scheduler, Sequence};

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
    let key_tokens = (block_size * 8).min(tokens.len());
    if key_tokens < block_size {
        return None;
    }
    let key_tokens = key_tokens - (key_tokens % block_size);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    block_size.hash(&mut hasher);
    tokens[..key_tokens].hash(&mut hasher);
    Some(hasher.finish())
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

    if is_final
        && let Err(e) = engine.cache.try_prefix_cache_insert_paged_only(
            &seq.input_ids,
            &result.block_table,
            pool.block_size,
        )
    {
        tracing::warn!("prefix cache insert (ar_loop) failed: {e}");
    }
}
