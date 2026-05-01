use crate::engine::*;
use crate::models::commons::BatchAttnContext;
use crate::models::commons::PagedKvBatchContext;

impl Engine {
    /// Batch prefill multiple requests using varlen paged attention.
    ///
    /// Supports chunked prefill: when `prefill_plan.computed_lens` is non-empty,
    /// each request may have tokens already in KV cache from previous chunks.
    /// Only the new tokens (after computed_len) are processed.
    ///
    /// Returns first token + block table per request. Blocks are NOT freed —
    /// the caller must either call `stream_decode_with_blocks` or free them manually.
    pub(crate) fn batch_prefill_paged(
        &self,
        items: &mut [PreparedGenerateRequest],
        prefill_plan: &PrefillPlan,
    ) -> Result<Vec<BatchPrefillResult>, EngineError> {
        let pool = self.cache.paged_pool.as_ref().ok_or_else(|| {
            EngineError::Internal("batch_prefill_paged requires paged attention pool".into())
        })?;

        let batch_size = items.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        if prefill_plan.seq_lens.len() != batch_size {
            return Err(EngineError::Internal(
                "batch_prefill_paged received mismatched prefill plan".into(),
            ));
        }

        let seq_lens = &prefill_plan.seq_lens;
        let has_chunked = !prefill_plan.computed_lens.is_empty();

        // Resolve per-request cached_len: either from chunked prefill state
        // or from prefix cache (global cached_len for all requests).
        let prefix_reuse = self.resolve_paged_prefix_reuse(prefill_plan)?;
        let global_cached_len = prefix_reuse.cached_len;

        let per_request_cached: Vec<usize> = if has_chunked {
            // Chunked prefill: per-request computed tokens override prefix cache.
            prefill_plan
                .computed_lens
                .iter()
                .map(|&cl| cl.max(global_cached_len))
                .collect()
        } else {
            vec![global_cached_len; batch_size]
        };

        // Build packed varlen input from uncached suffix tokens.
        let mut flat_tokens: Vec<u32> = Vec::new();
        let mut cu_seqlens_q = vec![0u32];
        let mut cu_seqlens_k = vec![0u32];
        let mut q_seq_lens: Vec<usize> = Vec::with_capacity(batch_size);
        let mut max_seqlen_q = 0usize;
        let mut max_seqlen_k = 0usize;

        for (i, item) in items.iter().enumerate() {
            let cached = per_request_cached[i];
            let suffix = &item.prompt_tokens[cached..];
            let q_len = suffix.len();
            let k_len = item.prompt_tokens.len();
            flat_tokens.extend_from_slice(suffix);
            cu_seqlens_q.push(cu_seqlens_q.last().unwrap() + q_len as u32);
            cu_seqlens_k.push(cu_seqlens_k.last().unwrap() + k_len as u32);
            q_seq_lens.push(q_len);
            max_seqlen_q = max_seqlen_q.max(q_len);
            max_seqlen_k = max_seqlen_k.max(k_len);
        }

        let total_tokens = flat_tokens.len();
        let packed_input = Tensor::from_vec(flat_tokens, (total_tokens,), &self.executor.device)
            .map_err(tensor_err)?;
        let cu_seqlens_q_t =
            Tensor::from_vec(cu_seqlens_q, (batch_size + 1,), &self.executor.device)
                .map_err(tensor_err)?;
        let cu_seqlens_k_t =
            Tensor::from_vec(cu_seqlens_k, (batch_size + 1,), &self.executor.device)
                .map_err(tensor_err)?;

        // Position IDs: per-request offset from cached tokens.
        let mut position_ids: Vec<u32> = Vec::with_capacity(total_tokens);
        for (i, &q_len) in q_seq_lens.iter().enumerate() {
            let cached = per_request_cached[i];
            for pos in 0..q_len {
                position_ids.push((cached + pos) as u32);
            }
        }
        let position_ids_t = Tensor::from_vec(position_ids, (total_tokens,), &self.executor.device)
            .map_err(tensor_err)?;

        // Allocate paged blocks — handle both standard and chunked prefill.
        let block_tables = if has_chunked {
            self.allocate_blocks_chunked(
                seq_lens,
                &prefill_plan.existing_block_tables,
                pool.block_size,
            )?
        } else {
            let allocation_plan = self.build_cache_allocation_plan(seq_lens, &prefix_reuse)?;
            self.allocate_block_tables_from_plan(&allocation_plan, "batch_prefill_paged")?
        };

        // Build block_tables tensor [batch_size, max_blocks]
        let max_blocks = block_tables.iter().map(|bt| bt.len()).max().unwrap_or(0);
        let mut flat_bt: Vec<u32> = Vec::with_capacity(batch_size * max_blocks);
        for bt in &block_tables {
            flat_bt.extend_from_slice(bt);
            for _ in bt.len()..max_blocks {
                flat_bt.push(0);
            }
        }
        let block_tables_t = if max_blocks > 0 {
            Tensor::from_vec(flat_bt, (batch_size, max_blocks), &self.executor.device)
                .map_err(tensor_err)?
                .to_dtype(DType::U32)
                .map_err(tensor_err)?
        } else {
            Tensor::zeros((batch_size, 0), DType::U32, &self.executor.device)
                .map_err(tensor_err)?
        };

        // Build slot_mapping for new tokens only (suffix tokens).
        let mut slots: Vec<i64> = Vec::with_capacity(total_tokens);
        for (i, &q_len) in q_seq_lens.iter().enumerate() {
            let cached = per_request_cached[i];
            for t in 0..q_len {
                slots.push(crate::cache::block_manager::BlockManager::slot(
                    &block_tables[i],
                    cached + t,
                    pool.block_size,
                ));
            }
        }
        let slot_mapping_t =
            Tensor::new(slots.as_slice(), &self.executor.device).map_err(tensor_err)?;

        // Allocate DeltaNet pool slots if this is a hybrid model.
        // For chunked prefill continuation, the slot was already allocated in a previous chunk
        // and is tracked in ArSequenceState. Only allocate for new requests (no existing blocks).
        let deltanet_slots: Option<Vec<u32>> = if let Some(ref dn_pool_mutex) =
            self.cache.deltanet_pool
        {
            let mut dn_pool = dn_pool_mutex
                .lock()
                .map_err(|e| EngineError::Internal(format!("deltanet pool lock: {e}")))?;
            let mut slots = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                if has_chunked && !prefill_plan.existing_block_tables.get(i).map_or(true, |bt| bt.is_empty()) {
                    // Continuation chunk — slot already allocated, use placeholder.
                    // The actual slot is tracked in ArSequenceState.deltanet_slot.
                    slots.push(u32::MAX);
                } else {
                    let slot = dn_pool.allocate().ok_or_else(|| {
                        EngineError::Internal("batch_prefill_paged: no free DeltaNet pool slots".into())
                    })?;
                    slots.push(slot);
                }
            }
            Some(slots)
        } else {
            None
        };

        // Forward
        let prefill_start = Instant::now();
        let mut model = self
            .executor
            .model
            .lock()
            .map_err(|e| EngineError::Internal(format!("model lock poisoned: {e}")))?;

        let mut dn_pool_guard = self.cache.deltanet_pool.as_ref().map(|m| m.lock().unwrap());
        let dn_pool_ref = dn_pool_guard.as_deref_mut();

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
            max_seqlen_q,
            position_ids: &position_ids_t,
            seq_lens: &q_seq_lens,
            paged_kv: Some(&paged_kv),
            deltanet_pool: dn_pool_ref,
            deltanet_slots: deltanet_slots.as_deref(),
            deltanet_slots_gpu: None,
        };
        let needs_prompt_logprobs = items.iter().any(|item| item.request.prompt_logprobs.is_some());

        self.executor.ops.begin_forward();

        // When prompt logprobs needed: get hidden states, apply compute_logits in chunks.
        // This avoids materializing the full (total_tokens, vocab_size) logits tensor.
        let (logits, prompt_logprobs_cpu) = if needs_prompt_logprobs {
            let lm = model.as_logits_model_mut()
                .ok_or_else(|| EngineError::InvalidRequest(
                    "prompt_logprobs requested but model doesn't support LogitsSplitModel".into()
                ))?;
            let hidden = lm.forward_hidden_states(&packed_input, &mut ctx).map_err(tensor_err)?;
            let last_hidden = crate::models::commons::last_token_select(&hidden, &q_seq_lens)
                .map_err(tensor_err)?;
            let last_logits = lm.compute_logits(&last_hidden).map_err(tensor_err)?
                .unsqueeze(1).map_err(tensor_err)?;

            // Chunked prompt logprobs extraction while model lock is still held.
            let logprobs_cpu = super::generate::extract_prompt_logprobs_from_hidden_offset(
                &hidden, lm, items, &q_seq_lens, global_cached_len,
            )?;
            drop(hidden);
            (last_logits, Some(logprobs_cpu))
        } else {
            (model.forward(&packed_input, &mut ctx).map_err(tensor_err)?, None)
        };
        self.executor.ops.end_forward();
        drop(dn_pool_guard);
        drop(model);

        let prefill_ms = prefill_start.elapsed().as_secs_f32() * 1000.0;

        // Sample first token per request
        let logits_2d = logits.squeeze(1).map_err(tensor_err)?;
        let mut results: Vec<BatchPrefillResult> = Vec::with_capacity(batch_size);

        // Build prompt logprobs per item from pre-extracted CPU data.
        let prompt_logprobs_per_item: Vec<Option<Vec<TokenLogprobInfo>>> = if let Some(ref logprobs_cpu) = prompt_logprobs_cpu {
            let mut per_item = Vec::with_capacity(batch_size);
            let mut offset = 0usize;
            for (i, item) in items.iter().enumerate() {
                let q_len = q_seq_lens[i];
                let cached = per_request_cached[i];
                if item.request.prompt_logprobs.is_some() {
                    let prompt_tokens = &item.prompt_tokens[cached..];
                    let plps: Vec<TokenLogprobInfo> = (0..q_len.saturating_sub(1))
                        .map(|pos| {
                            let next_token = prompt_tokens[pos + 1];
                            TokenLogprobInfo {
                                token: self.tokenizer.decode(&[next_token], false).unwrap_or_default(),
                                token_id: next_token,
                                logprob: logprobs_cpu[offset + pos],
                                top_logprobs: Vec::new(),
                            }
                        })
                        .collect();
                    per_item.push(Some(plps));
                } else {
                    per_item.push(None);
                }
                offset += q_len;
            }
            per_item
        } else {
            vec![None; batch_size]
        };

        // Fast path: all greedy → batch GPU argmax (avoids F32 conversion + CPU transfer)
        if prefill_plan.all_greedy {
            let all_tokens = logits_2d
                .argmax(crate::tensor::D::Minus1)
                .map_err(tensor_err)?
                .to_vec1::<u32>()
                .map_err(tensor_err)?;
            for (i, token) in all_tokens.into_iter().enumerate() {
                let first_token_logprobs = if let Some(k) = items[i].request.logprobs {
                    let row = logits_2d.get(i).map_err(tensor_err)?;
                    Some(Self::extract_top_logprobs(&row, token, k, &self.tokenizer)?)
                } else {
                    None
                };
                results.push(BatchPrefillResult {
                    first_token: token,
                    block_table: block_tables[i].clone(),
                    prompt_len: seq_lens[i],
                    prefill_ms,
                    deltanet_slot: deltanet_slots.as_ref().map(|s| s[i]),
                    first_token_logprobs,
                    prompt_token_logprobs: prompt_logprobs_per_item[i].clone(),
                });
            }
        } else {
            for (i, item) in items.iter_mut().enumerate() {
                let row = logits_2d.get(i).map_err(tensor_err)?;
                let row_f32 = row.to_dtype(DType::F32).map_err(tensor_err)?;
                let token = item.logits_processor.sample(&row_f32).map_err(tensor_err)?;
                let first_token_logprobs = if let Some(k) = item.request.logprobs {
                    Some(Self::extract_top_logprobs(&row, token, k, &self.tokenizer)?)
                } else {
                    None
                };
                results.push(BatchPrefillResult {
                    first_token: token,
                    block_table: block_tables[i].clone(),
                    prompt_len: seq_lens[i],
                    prefill_ms,
                    deltanet_slot: deltanet_slots.as_ref().map(|s| s[i]),
                    first_token_logprobs,
                    prompt_token_logprobs: prompt_logprobs_per_item[i].clone(),
                });
            }
        }

        info!(
            batch_size,
            prefill_ms = format!("{:.1}", prefill_ms),
            total_prompt_tokens = total_tokens,
            "batch_prefill_paged complete"
        );

        Ok(results)
    }

    /// Allocate blocks for chunked prefill: extend existing block tables
    /// with new blocks for the additional tokens.
    fn allocate_blocks_chunked(
        &self,
        seq_lens: &[usize],
        existing_block_tables: &[Vec<u32>],
        block_size: usize,
    ) -> Result<Vec<Vec<u32>>, EngineError> {
        let bm_mutex = self.cache.block_manager.as_ref().ok_or_else(|| {
            EngineError::Internal("allocate_blocks_chunked: block manager unavailable".into())
        })?;
        let mut bm = bm_mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;

        let mut block_tables = Vec::with_capacity(seq_lens.len());
        for (i, &total_tokens) in seq_lens.iter().enumerate() {
            let total_blocks_needed = total_tokens.div_ceil(block_size);
            let existing = existing_block_tables
                .get(i)
                .cloned()
                .unwrap_or_default();
            let existing_blocks = existing.len();

            let mut bt = existing;
            for _ in existing_blocks..total_blocks_needed {
                let block = bm.allocate().ok_or_else(|| {
                    EngineError::Internal("allocate_blocks_chunked: no free blocks".into())
                })?;
                bt.push(block);
            }
            block_tables.push(bt);
        }

        Ok(block_tables)
    }
}
