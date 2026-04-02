#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
use super::super::*;

#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
pub(crate) struct PackedSequenceBatch {
    pub flat_tokens: Vec<u32>,
    pub cu_seqlens: Vec<u32>,
    pub seq_lens: Vec<usize>,
    pub position_ids: Vec<u32>,
    pub item_seq_counts: Vec<usize>,
    pub total_tokens: usize,
    pub batch_size: usize,
    pub max_seqlen: usize,
}

/// Result of a batched varlen forward pass, shared by classify, embed, and generation.
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
pub(crate) struct PrefillForwardResult {
    /// Raw model output tensor (not converted to F32 — callers decide dtype).
    /// Shape: (batch_size, vocab_size) — last token logits per sequence.
    pub output: candle_core::Tensor,
    /// Number of sequences per input item (for classify/embed grouping).
    pub item_seq_counts: Vec<usize>,
    /// Per-sequence query lengths (suffix lengths after prefix cache skip).
    pub seq_lens: Vec<usize>,
    /// Hidden states before lm_head (total_tokens, hidden_dim).
    /// Only populated when prompt_logprobs requested. Much smaller than
    /// full logits (hidden_dim vs vocab_size), safe to pass across threads.
    pub hidden_states: Option<candle_core::Tensor>,
}

/// Find the longest common prefix across all token sequences in the groups.
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
fn find_common_prefix_from_groups(token_groups: &[&[Vec<u32>]]) -> Vec<u32> {
    let all_seqs: Vec<&[u32]> = token_groups
        .iter()
        .flat_map(|g| g.iter().map(|t| t.as_slice()))
        .collect();

    if all_seqs.is_empty() {
        return vec![];
    }
    if all_seqs.len() == 1 {
        return all_seqs[0].to_vec();
    }

    let first = all_seqs[0];
    let min_len = all_seqs.iter().map(|s| s.len()).min().unwrap_or(0);
    let mut common_len = 0;
    for i in 0..min_len {
        if all_seqs.iter().all(|s| s[i] == first[i]) {
            common_len = i + 1;
        } else {
            break;
        }
    }
    first[..common_len].to_vec()
}

#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
impl Engine {
    /// Unified prefill pipeline for all task types (classify, embed, generate).
    ///
    /// Serial pipeline:
    /// 1. Prefix cache lookup — find common prefix, check paged cache
    /// 2. Pack tokens — skip cached prefix, build varlen tensors
    /// 3. Varlen forward — with optional paged KV from prefix cache
    /// 4. Cleanup — free temporary paged blocks
    ///
    /// Returns `None` if the batch is empty.
    pub(crate) fn prefill_pipeline(
        &self,
        token_groups: &[&[Vec<u32>]],
    ) -> Result<Option<PrefillForwardResult>, EngineError> {
        self.prefill_pipeline_inner(token_groups, false)
    }

    /// Like `prefill_pipeline` but also returns hidden states (before lm_head)
    /// for prompt logprobs extraction via chunked `compute_logits`.
    pub(crate) fn prefill_pipeline_with_hidden_states(
        &self,
        token_groups: &[&[Vec<u32>]],
    ) -> Result<Option<PrefillForwardResult>, EngineError> {
        self.prefill_pipeline_inner(token_groups, true)
    }

    fn prefill_pipeline_inner(
        &self,
        token_groups: &[&[Vec<u32>]],
        need_hidden_states: bool,
    ) -> Result<Option<PrefillForwardResult>, EngineError> {
        use crate::models::common::BatchAttnContext;

        // ── Step 1: Prefix cache lookup ──
        let common_prefix = find_common_prefix_from_groups(token_groups);
        let min_seq_len = token_groups
            .iter()
            .flat_map(|g| g.iter().map(|t| t.len()))
            .min()
            .unwrap_or(0);
        let (cached_len, cached_block_ids) = if !common_prefix.is_empty() {
            self.try_prefix_match_for_prefill(&common_prefix, min_seq_len)?
        } else {
            (0, vec![])
        };

        // ── Step 2: Pack tokens with prefix offset ──
        let PackedSequenceBatch {
            flat_tokens,
            cu_seqlens,
            seq_lens,
            position_ids,
            item_seq_counts,
            total_tokens,
            batch_size,
            max_seqlen,
        } = pack_varlen_tokens(token_groups, cached_len);

        if batch_size == 0 {
            return Ok(None);
        }

        let device = &self.executor.device;
        let packed_input = candle_core::Tensor::from_vec(flat_tokens, (total_tokens,), device)
            .map_err(candle_err)?;
        let cu_seqlens_q_t = candle_core::Tensor::from_vec(cu_seqlens, (batch_size + 1,), device)
            .map_err(candle_err)?;
        let position_ids_t = candle_core::Tensor::from_vec(position_ids, (total_tokens,), device)
            .map_err(candle_err)?;

        // ── Step 3: Varlen forward ──
        let mut model = self
            .executor
            .model
            .lock()
            .map_err(|e| EngineError::Internal(format!("model lock poisoned: {e}")))?;

        // ── Determine paged forward strategy ──
        // (flash-attn-v3 implies cuda implies paged-attn, so paged infra is always available here)
        use crate::engine::ResolvedPrefixReuse;
        let prefix_reuse = if cached_len > 0 && !cached_block_ids.is_empty() {
            ResolvedPrefixReuse {
                cached_len,
                cached_block_ids,
            }
        } else {
            ResolvedPrefixReuse::default()
        };

        let (output, hidden_states) = {
            {
                let is_cache_hit = prefix_reuse.cached_len > 0;
                let should_populate_prefix_cache = !is_cache_hit
                    && self.cache.prefix_cache.is_some()
                    && self.cache.paged_pool.is_some()
                    && !common_prefix.is_empty();
                let use_paged_forward = is_cache_hit || should_populate_prefix_cache;

                let mut allocated_block_tables: Option<Vec<Vec<u32>>> = None;
                let mut paged_block_size: usize = 0;
                // Full sequence lengths (cached prefix + suffix) for block allocation.
                let full_seq_lens: Vec<usize> = seq_lens
                    .iter()
                    .map(|&q| prefix_reuse.cached_len + q)
                    .collect();
                let run = (|| -> Result<(candle_core::Tensor, Option<candle_core::Tensor>), EngineError> {
                    if use_paged_forward {
                        if let Some(pool) = &self.cache.paged_pool {
                            use crate::models::common::PagedKvBatchContext;

                            paged_block_size = pool.block_size;

                            // Use planner helpers for block allocation.
                            let allocation_plan =
                                self.build_cache_allocation_plan(&full_seq_lens, &prefix_reuse)?;
                            let block_tables = self.allocate_block_tables_from_plan(
                                &allocation_plan,
                                "prefill pipeline paged",
                            )?;
                            let max_blocks = allocation_plan.max_total_blocks;

                            // Flatten block tables → [batch_size, max_blocks].
                            let mut flat_bt: Vec<u32> = Vec::with_capacity(batch_size * max_blocks);
                            for bt in &block_tables {
                                flat_bt.extend_from_slice(bt);
                                for _ in bt.len()..max_blocks {
                                    flat_bt.push(0);
                                }
                            }
                            let block_tables_t = candle_core::Tensor::from_vec(
                                flat_bt,
                                (batch_size, max_blocks),
                                device,
                            )
                            .map_err(candle_err)?
                            .to_dtype(candle_core::DType::U32)
                            .map_err(candle_err)?;

                            // Slot mapping for suffix tokens being written to paged cache.
                            let mut slots: Vec<i64> = Vec::with_capacity(total_tokens);
                            for (i, &q_len) in seq_lens.iter().enumerate() {
                                for t in 0..q_len {
                                    let pos = prefix_reuse.cached_len + t;
                                    slots.push(crate::cache::block_manager::BlockManager::slot(
                                        &block_tables[i],
                                        pos,
                                        pool.block_size,
                                    ));
                                }
                            }
                            let slot_mapping_t = candle_core::Tensor::new(slots.as_slice(), device)
                                .map_err(candle_err)?;

                            // cu_seqlens_k: cumulative full seq lengths (cached + suffix).
                            let mut cu_seqlens_k = Vec::with_capacity(batch_size + 1);
                            cu_seqlens_k.push(0u32);
                            let mut max_seqlen_k = 0usize;
                            for &q_len in &seq_lens {
                                let k_len = prefix_reuse.cached_len + q_len;
                                cu_seqlens_k.push(cu_seqlens_k.last().unwrap_or(&0) + k_len as u32);
                                max_seqlen_k = max_seqlen_k.max(k_len);
                            }
                            let cu_seqlens_k_t = candle_core::Tensor::from_vec(
                                cu_seqlens_k,
                                (batch_size + 1,),
                                device,
                            )
                            .map_err(candle_err)?;

                            allocated_block_tables = Some(block_tables);

                            tracing::debug!(
                                cached_len = prefix_reuse.cached_len,
                                batch_size,
                                max_seqlen_k,
                                is_cache_hit,
                                should_populate_prefix_cache,
                                "prefill pipeline: using paged KV"
                            );

                            let paged_kv = PagedKvBatchContext {
                                key_caches: &pool.active_key_caches(),
                                value_caches: &pool.active_value_caches(),
                                slot_mapping: &slot_mapping_t,
                                block_tables: &block_tables_t,
                                cu_seqlens_k: &cu_seqlens_k_t,
                                max_seqlen_k,
                            };
                            let mut ctx = BatchAttnContext {
                                ops: self.executor.ops,
                                cu_seqlens_q: &cu_seqlens_q_t,
                                max_seqlen_q: max_seqlen,
                                position_ids: &position_ids_t,
                                seq_lens: &seq_lens,
                                paged_kv: Some(&paged_kv),
                                deltanet_pool: None,
                                deltanet_slots: None,
                            };
                            self.executor.ops.session.begin_forward();
                            let result = if need_hidden_states {
                                let lm = model.as_logits_model_mut()
                                    .expect("hidden states requested but model doesn't support LogitsSplitModel");
                                let hidden = lm.forward_hidden_states(&packed_input, &mut ctx).map_err(candle_err)?;
                                let last = crate::models::common::last_token_select(&hidden, &seq_lens)
                                    .map_err(candle_err)?;
                                let logits = lm.compute_logits(&last).map_err(candle_err)?
                                    .unsqueeze(1).map_err(candle_err)?;
                                Ok((logits, Some(hidden)))
                            } else {
                                Ok((model.forward(&packed_input, &mut ctx).map_err(candle_err)?, None))
                            };
                            self.executor.ops.session.end_forward();
                            return result;
                        }
                    }

                    // No paged path — plain varlen.
                    let mut ctx = BatchAttnContext {
                        ops: self.executor.ops,
                        cu_seqlens_q: &cu_seqlens_q_t,
                        max_seqlen_q: max_seqlen,
                        position_ids: &position_ids_t,
                        seq_lens: &seq_lens,
                        paged_kv: None,
                        deltanet_pool: None,
                        deltanet_slots: None,
                    };
                    self.executor.ops.session.begin_forward();
                    let result = if need_hidden_states {
                        let lm = model.as_logits_model_mut()
                            .expect("hidden states requested but model doesn't support LogitsSplitModel");
                        let hidden = lm.forward_hidden_states(&packed_input, &mut ctx).map_err(candle_err)?;
                        let last = crate::models::common::last_token_select(&hidden, &seq_lens)
                            .map_err(candle_err)?;
                        let logits = lm.compute_logits(&last).map_err(candle_err)?
                            .unsqueeze(1).map_err(candle_err)?;
                        Ok((logits, Some(hidden)))
                    } else {
                        Ok((model.forward(&packed_input, &mut ctx).map_err(candle_err)?, None))
                    };
                    self.executor.ops.session.end_forward();
                    result
                })();

                // Insert into prefix cache on cache miss (populate for future hits).
                if should_populate_prefix_cache
                    && let Some(ref block_tables) = allocated_block_tables
                {
                    if let Err(e) = self.cache.try_prefix_cache_insert_paged_only(
                        &common_prefix,
                        &block_tables[0],
                        paged_block_size,
                    ) {
                        tracing::warn!("prefix cache insert failed: {e}");
                    }
                }

                // Free temporarily allocated paged blocks.
                // Blocks stored in prefix cache survive (ref count > 0 after decrement).
                if let Some(block_tables) = allocated_block_tables
                    && let Some(ref bm_mutex) = self.cache.block_manager
                    && let Ok(mut bm) = bm_mutex.lock()
                {
                    for bt in &block_tables {
                        bm.free(bt);
                    }
                }

                run?
            }
        };
        drop(model);

        Ok(Some(PrefillForwardResult {
            output,
            item_seq_counts,
            seq_lens,
            hidden_states,
        }))
    }

    /// Try to match prefix cache for a common prefix across all sequences.
    /// Returns (cached_len, cached_block_ids).
    fn try_prefix_match_for_prefill(
        &self,
        common_prefix: &[u32],
        min_seq_len: usize,
    ) -> Result<(usize, Vec<u32>), EngineError> {
        if self.cache.prefix_cache.is_none() {
            return Ok((0, vec![]));
        }

        // (flash-attn-v3 implies cuda, so paged infra is always available here)
        {
            // Block-size alignment check: paged attention kernels require
            // block_size aligned to head_dim constraints.
            if let Some(pool) = self.cache.paged_pool.as_ref() {
                let block_ok = if self.executor.config.head_dim == 256 {
                    pool.block_size % 64 == 0
                } else {
                    pool.block_size % 128 == 0
                };
                if !block_ok {
                    return Ok((0, vec![]));
                }
            }

            let (cached_len, cached_block_ids) = self
                .cache
                .try_prefix_cache_match_paged_only(common_prefix)?;
            if cached_len > 0 && cached_len < min_seq_len {
                tracing::debug!(
                    cached_len,
                    common_prefix_len = common_prefix.len(),
                    min_seq_len,
                    cached_blocks = cached_block_ids.len(),
                    "prefill pipeline: prefix cache hit"
                );
                return Ok((cached_len, cached_block_ids));
            }
        }

        Ok((0, vec![]))
    }
}

/// Pack token groups into varlen format, skipping the first `offset` tokens
/// from each sequence (used when prefix cache provides the leading tokens).
/// Position IDs account for the offset so the model sees correct positions.
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
fn pack_varlen_tokens(
    token_groups: &[&[Vec<u32>]],
    offset: usize,
) -> PackedSequenceBatch {
    let mut flat_tokens: Vec<u32> = Vec::new();
    let mut cu_seqlens = vec![0u32];
    let mut seq_lens: Vec<usize> = Vec::new();
    let mut max_seqlen = 0usize;
    let mut item_seq_counts: Vec<usize> = Vec::with_capacity(token_groups.len());

    for group in token_groups {
        item_seq_counts.push(group.len());
        for tokens in *group {
            let suffix = &tokens[offset.min(tokens.len())..];
            let len = suffix.len();
            flat_tokens.extend_from_slice(suffix);
            cu_seqlens.push(cu_seqlens.last().unwrap_or(&0) + len as u32);
            seq_lens.push(len);
            max_seqlen = max_seqlen.max(len);
        }
    }

    let total_tokens = flat_tokens.len();
    let batch_size = seq_lens.len();
    let mut position_ids: Vec<u32> = Vec::with_capacity(total_tokens);
    for &len in &seq_lens {
        for pos in 0..len as u32 {
            position_ids.push(offset as u32 + pos);
        }
    }

    PackedSequenceBatch {
        flat_tokens,
        cu_seqlens,
        seq_lens,
        position_ids,
        item_seq_counts,
        total_tokens,
        batch_size,
        max_seqlen,
    }
}
