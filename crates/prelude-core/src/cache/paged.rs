use crate::engine::*;
#[cfg(feature = "flash-attn-v3")]
use crate::models::layers::BatchAttnContext;
#[cfg(feature = "flash-attn-v3")]
use crate::models::layers::PagedKvBatchContext;

impl Engine {
    // ── Batch prefill + stream decode (paged attention) ──────────────────

    /// Batch prefill multiple requests using varlen paged attention.
    /// Returns first token + block table per request. Blocks are NOT freed —
    /// the caller must either call `stream_decode_with_blocks` or free them manually.
    #[cfg(feature = "flash-attn-v3")]
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
        let prefix_reuse = self.resolve_paged_prefix_reuse(prefill_plan)?;
        let cached_len = prefix_reuse.cached_len;
        let allocation_plan = self.build_cache_allocation_plan(seq_lens, &prefix_reuse)?;

        // Build packed varlen input from uncached suffix tokens.
        let mut flat_tokens: Vec<u32> = Vec::new();
        let mut cu_seqlens_q = vec![0u32];
        let mut cu_seqlens_k = vec![0u32];
        let mut q_seq_lens: Vec<usize> = Vec::with_capacity(batch_size);
        let mut max_seqlen_q = 0usize;
        let mut max_seqlen_k = 0usize;

        for item in items.iter() {
            let suffix = &item.prompt_tokens[cached_len..];
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
            .map_err(candle_err)?;
        let cu_seqlens_q_t =
            Tensor::from_vec(cu_seqlens_q, (batch_size + 1,), &self.executor.device)
                .map_err(candle_err)?;
        let cu_seqlens_k_t =
            Tensor::from_vec(cu_seqlens_k, (batch_size + 1,), &self.executor.device)
                .map_err(candle_err)?;

        // Position IDs for suffix tokens: [cached_len..cached_len+q_len_1, ...]
        let mut position_ids: Vec<u32> = Vec::with_capacity(total_tokens);
        for &q_len in &q_seq_lens {
            for pos in 0..q_len {
                position_ids.push((cached_len + pos) as u32);
            }
        }
        let position_ids_t = Tensor::from_vec(position_ids, (total_tokens,), &self.executor.device)
            .map_err(candle_err)?;

        // Allocate paged blocks for each request
        let block_tables =
            self.allocate_block_tables_from_plan(&allocation_plan, "batch_prefill_paged")?;

        // Build block_tables tensor [batch_size, max_blocks]
        let max_blocks = allocation_plan.max_total_blocks;
        let mut flat_bt: Vec<u32> = Vec::with_capacity(batch_size * max_blocks);
        for bt in &block_tables {
            flat_bt.extend_from_slice(bt);
            for _ in bt.len()..max_blocks {
                flat_bt.push(0);
            }
        }
        let block_tables_t =
            Tensor::from_vec(flat_bt, (batch_size, max_blocks), &self.executor.device)
                .map_err(candle_err)?
                .to_dtype(DType::U32)
                .map_err(candle_err)?;

        // Build slot_mapping for all tokens
        let mut slots: Vec<i64> = Vec::with_capacity(total_tokens);
        for (i, &q_len) in q_seq_lens.iter().enumerate() {
            for t in 0..q_len {
                slots.push(crate::block_manager::BlockManager::slot(
                    &block_tables[i],
                    cached_len + t,
                    pool.block_size,
                ));
            }
        }
        let slot_mapping_t =
            Tensor::new(slots.as_slice(), &self.executor.device).map_err(candle_err)?;

        // Allocate DeltaNet pool slots if this is a hybrid model
        let deltanet_slots: Option<Vec<u32>> = if let Some(ref dn_pool_mutex) =
            self.cache.deltanet_pool
        {
            let mut dn_pool = dn_pool_mutex
                .lock()
                .map_err(|e| EngineError::Internal(format!("deltanet pool lock: {e}")))?;
            let mut slots = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let slot = dn_pool.allocate().ok_or_else(|| {
                    EngineError::Internal("batch_prefill_paged: no free DeltaNet pool slots".into())
                })?;
                slots.push(slot);
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
            cu_seqlens_q: &cu_seqlens_q_t,
            max_seqlen_q,
            position_ids: &position_ids_t,
            seq_lens: &q_seq_lens,
            paged_kv: Some(&paged_kv),
            deltanet_pool: dn_pool_ref,
            deltanet_slots: deltanet_slots.as_deref(),
        };
        let logits = model.forward(&packed_input, &mut ctx).map_err(candle_err)?;
        // logits: (batch_size, 1, vocab_size)
        drop(dn_pool_guard);
        drop(model);

        let prefill_ms = prefill_start.elapsed().as_secs_f32() * 1000.0;

        // Sample first token per request
        let logits_2d = logits.squeeze(1).map_err(candle_err)?; // (batch_size, vocab_size)
        let mut results: Vec<BatchPrefillResult> = Vec::with_capacity(batch_size);

        // Fast path: all greedy → batch GPU argmax (avoids F32 conversion + CPU transfer)
        if prefill_plan.all_greedy {
            let all_tokens = logits_2d
                .argmax(candle_core::D::Minus1)
                .map_err(candle_err)?
                .to_vec1::<u32>()
                .map_err(candle_err)?;
            for (i, token) in all_tokens.into_iter().enumerate() {
                let first_token_logprobs = if let Some(k) = items[i].request.logprobs {
                    let row = logits_2d.get(i).map_err(candle_err)?;
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
                });
            }
        } else {
            for (i, item) in items.iter_mut().enumerate() {
                let row = logits_2d.get(i).map_err(candle_err)?;
                let row_f32 = row.to_dtype(DType::F32).map_err(candle_err)?;
                let token = item.logits_processor.sample(&row_f32).map_err(candle_err)?;
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

    /// Batched decode step: N sequences, each with Q=1 (one new token),
    /// different context lengths. Returns logits (N, vocab_size).
    #[cfg(feature = "flash-attn-v3")]
    pub fn batch_decode_paged(
        &self,
        seqs: &[BatchDecodeSeq],
    ) -> Result<Tensor, EngineError> {
        let pool = self.cache.paged_pool.as_ref().ok_or_else(|| {
            EngineError::Internal("batch_decode_paged requires paged attention pool".into())
        })?;

        let batch_size = seqs.len();
        if batch_size == 0 {
            return Err(EngineError::Internal("empty decode batch".into()));
        }

        // packed input: one token per sequence
        let flat_tokens: Vec<u32> = seqs.iter().map(|s| s.token).collect();
        let packed_input = Tensor::from_vec(flat_tokens, (batch_size,), &self.executor.device)
            .map_err(candle_err)?;

        // cu_seqlens_q: [0, 1, 2, ..., N] — each seq has Q=1
        let cu_seqlens_q: Vec<u32> = (0..=batch_size as u32).collect();
        let cu_seqlens_q_t =
            Tensor::from_vec(cu_seqlens_q, (batch_size + 1,), &self.executor.device)
                .map_err(candle_err)?;

        // cu_seqlens_k: [0, ctx1, ctx1+ctx2, ...] — different K lengths
        let mut cu_seqlens_k: Vec<u32> = vec![0];
        let mut max_seqlen_k = 0usize;
        for s in seqs {
            cu_seqlens_k.push(cu_seqlens_k.last().unwrap() + s.context_len as u32);
            max_seqlen_k = max_seqlen_k.max(s.context_len);
        }
        let cu_seqlens_k_t =
            Tensor::from_vec(cu_seqlens_k, (batch_size + 1,), &self.executor.device)
                .map_err(candle_err)?;

        // position_ids: position for each token
        let position_ids: Vec<u32> = seqs.iter().map(|s| s.position as u32).collect();
        let position_ids_t = Tensor::from_vec(position_ids, (batch_size,), &self.executor.device)
            .map_err(candle_err)?;

        let q_seq_lens: Vec<usize> = vec![1; batch_size];

        // slot_mapping: one slot per sequence
        let slots: Vec<i64> = seqs
            .iter()
            .map(|s| {
                crate::block_manager::BlockManager::slot(s.block_table, s.position, pool.block_size)
            })
            .collect();
        let slot_mapping_t =
            Tensor::new(slots.as_slice(), &self.executor.device).map_err(candle_err)?;

        // block_tables: padded to max_blocks
        let max_blocks = seqs.iter().map(|s| s.block_table.len()).max().unwrap_or(0);
        let mut flat_bt: Vec<u32> = Vec::with_capacity(batch_size * max_blocks);
        for s in seqs {
            flat_bt.extend_from_slice(s.block_table);
            flat_bt.resize(flat_bt.len() + max_blocks - s.block_table.len(), 0);
        }
        let block_tables_t =
            Tensor::from_vec(flat_bt, (batch_size, max_blocks), &self.executor.device)
                .map_err(candle_err)?
                .to_dtype(DType::U32)
                .map_err(candle_err)?;

        // Build deltanet_slots from BatchDecodeSeq
        let deltanet_slots: Option<Vec<u32>> = if self.cache.deltanet_pool.is_some() {
            let slots: Vec<u32> = seqs.iter().filter_map(|s| s.deltanet_slot).collect();
            if slots.len() == batch_size {
                Some(slots)
            } else {
                None
            }
        } else {
            None
        };

        // Forward
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
            cu_seqlens_q: &cu_seqlens_q_t,
            max_seqlen_q: 1,
            position_ids: &position_ids_t,
            seq_lens: &q_seq_lens,
            paged_kv: Some(&paged_kv),
            deltanet_pool: dn_pool_ref,
            deltanet_slots: deltanet_slots.as_deref(),
        };
        let logits = model.forward(&packed_input, &mut ctx).map_err(candle_err)?;
        drop(dn_pool_guard);
        drop(model);

        // logits: (batch_size, 1, vocab_size) → (batch_size, vocab_size)
        logits.squeeze(1).map_err(candle_err)
    }

    /// Continue streaming decode from a block table retained by `batch_prefill_paged`.
    /// Uses `forward` with paged KV and Q=1 for each decode step
    /// (same flash-layout KV caches as the batch prefill).
    /// Frees the block table when done.
    #[cfg(feature = "flash-attn-v3")]
    pub fn stream_decode_with_blocks(
        &self,
        request: &GenerateRequest,
        first_token: u32,
        mut block_table: Vec<u32>,
        prompt_len: usize,
        logits_processor: &mut LogitsProcessor,
        tx: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
        gen_start: Instant,
        prefill_ms: f32,
        deltanet_slot: Option<u32>,
        first_token_logprobs: Option<TokenLogprobInfo>,
    ) -> Result<GenerateResult, EngineError> {
        let pool = self.cache.paged_pool.as_ref().ok_or_else(|| {
            EngineError::Internal("stream_decode_with_blocks requires paged attention pool".into())
        })?;
        let bm_mutex = self.cache.block_manager.as_ref().ok_or_else(|| {
            EngineError::Internal("stream_decode_with_blocks requires block manager".into())
        })?;

        let max_context = self.executor.config.max_position_embeddings;
        let max_new = (request.max_new_tokens as usize).min(max_context.saturating_sub(prompt_len));
        let ttft_ms = prefill_ms;
        let logprobs_k = request.logprobs;
        let mut token_logprobs: Vec<TokenLogprobInfo> = Vec::new();

        // Send Started + first token
        let _ = tx.send(StreamEvent::Started);

        let mut output_tokens: Vec<u32> = vec![first_token];
        let mut finish_reason = FinishReason::Length;
        let mut sent_text_len = 0usize;

        if self.is_eos(first_token) {
            finish_reason = FinishReason::Eos;
        } else if self.check_stop_tokens(first_token, &request.stop.token_ids) {
            finish_reason = FinishReason::Stop;
        } else {
            // Accumulate first token logprobs
            if let Some(ref lp) = first_token_logprobs {
                token_logprobs.push(lp.clone());
            }
            // Send first token text
            let text = self
                .tokenizer
                .decode(&output_tokens, true)
                .unwrap_or_default();
            if text.len() > sent_text_len {
                let _ = tx.send(StreamEvent::Token {
                    text: text[sent_text_len..].to_string(),
                    logprobs: first_token_logprobs,
                });
                sent_text_len = text.len();
            }

            // Decode loop using forward (paged KV) with Q=1
            let decode_start = Instant::now();
            let mut pos = prompt_len; // next KV position

            for step in 1..max_new {
                // Allocate new block if needed
                if pos % pool.block_size == 0 {
                    let mut bm = bm_mutex
                        .lock()
                        .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
                    if let Some(new_block) = bm.allocate() {
                        block_table.push(new_block);
                    } else {
                        break;
                    }
                }

                // Build single-token varlen input
                let last_token = *output_tokens.last().unwrap();
                let input_t = Tensor::from_vec(vec![last_token], (1,), &self.executor.device)
                    .map_err(candle_err)?;
                let cu_q = Tensor::from_vec(vec![0u32, 1u32], (2,), &self.executor.device)
                    .map_err(candle_err)?;
                let context_len = (pos + 1) as u32;
                let cu_k = Tensor::from_vec(vec![0u32, context_len], (2,), &self.executor.device)
                    .map_err(candle_err)?;
                let pos_ids = Tensor::from_vec(vec![pos as u32], (1,), &self.executor.device)
                    .map_err(candle_err)?;

                let slot =
                    crate::block_manager::BlockManager::slot(&block_table, pos, pool.block_size);
                let slot_mapping =
                    Tensor::new(&[slot], &self.executor.device).map_err(candle_err)?;

                // Block table tensor [1, num_blocks]
                let bt_t = Tensor::new(block_table.as_slice(), &self.executor.device)
                    .map_err(candle_err)?
                    .to_dtype(DType::U32)
                    .map_err(candle_err)?
                    .unsqueeze(0)
                    .map_err(candle_err)?;

                let mut model = self
                    .executor
                    .model
                    .lock()
                    .map_err(|e| EngineError::Internal(format!("model lock poisoned: {e}")))?;

                let mut dn_pool_guard =
                    self.cache.deltanet_pool.as_ref().map(|m| m.lock().unwrap());
                let dn_pool_ref = dn_pool_guard.as_deref_mut();
                let dn_slots: Option<Vec<u32>> = deltanet_slot.map(|s| vec![s]);

                let paged_kv = PagedKvBatchContext {
                    key_caches: &pool.active_key_caches(),
                    value_caches: &pool.active_value_caches(),
                    slot_mapping: &slot_mapping,
                    block_tables: &bt_t,
                    cu_seqlens_k: &cu_k,
                    max_seqlen_k: context_len as usize,
                };
                let mut ctx = BatchAttnContext {
                    cu_seqlens_q: &cu_q,
                    max_seqlen_q: 1,
                    position_ids: &pos_ids,
                    seq_lens: &[1usize],
                    #[cfg(feature = "flash-attn-v3")]
                    paged_kv: Some(&paged_kv),
                    deltanet_pool: dn_pool_ref,
                    deltanet_slots: dn_slots.as_deref(),
                };
                let logits = model.forward(&input_t, &mut ctx).map_err(candle_err)?;
                drop(dn_pool_guard);
                drop(model);

                let logits = logits
                    .squeeze(0)
                    .map_err(candle_err)?
                    .squeeze(0)
                    .map_err(candle_err)?;

                let next_token = if request.sampling.temperature <= 1e-7 {
                    logits
                        .argmax(candle_core::D::Minus1)
                        .map_err(candle_err)?
                        .to_scalar::<u32>()
                        .map_err(candle_err)?
                } else {
                    let logits_f32 = logits.to_dtype(DType::F32).map_err(candle_err)?;
                    logits_processor.sample(&logits_f32).map_err(candle_err)?
                };
                pos += 1;

                // Extract logprobs before EOS/stop checks
                let token_lp = if let Some(k) = logprobs_k {
                    Self::extract_top_logprobs(&logits, next_token, k, &self.tokenizer).ok()
                } else {
                    None
                };

                if self.is_eos(next_token) {
                    finish_reason = FinishReason::Eos;
                    break;
                }
                if self.check_stop_tokens(next_token, &request.stop.token_ids) {
                    finish_reason = FinishReason::Stop;
                    break;
                }

                output_tokens.push(next_token);
                if let Some(ref lp) = token_lp {
                    token_logprobs.push(lp.clone());
                }

                let text = self
                    .tokenizer
                    .decode(&output_tokens, true)
                    .unwrap_or_default();
                if text.len() > sent_text_len {
                    let _ = tx.send(StreamEvent::Token {
                        text: text[sent_text_len..].to_string(),
                        logprobs: token_lp,
                    });
                    sent_text_len = text.len();
                }

                if step % 10 == 0 {
                    let elapsed = decode_start.elapsed().as_secs_f32() * 1000.0;
                    let tps = step as f32 / (elapsed / 1000.0);
                    info!(
                        rid = %request.request_id,
                        decoded = step, max_new_tokens = max_new,
                        decode_tps = format!("{:.1}", tps),
                        "paged stream decoding (batch prefill)"
                    );
                }

                if !request.stop.strings.is_empty()
                    && request.stop.strings.iter().any(|s| text.contains(s))
                {
                    finish_reason = FinishReason::Stop;
                    break;
                }
            }
        }

        // Free paged blocks
        {
            let mut bm = bm_mutex
                .lock()
                .map_err(|e| EngineError::Internal(format!("block manager lock: {e}")))?;
            bm.free(&block_table);
        }

        // Free DeltaNet pool slot
        if let (Some(slot), Some(dn_pool_mutex)) = (deltanet_slot, &self.cache.deltanet_pool)
            && let Ok(mut dn_pool) = dn_pool_mutex.lock()
        {
            dn_pool.free(slot);
        }

        let total_ms = gen_start.elapsed().as_secs_f32() * 1000.0;
        let decode_ms = total_ms - prefill_ms;
        let completion_tokens = output_tokens.len() as u32;
        let decode_tps = if decode_ms > 0.0 {
            completion_tokens as f32 / (decode_ms / 1000.0)
        } else {
            0.0
        };
        info!(
            rid = %request.request_id,
            prompt_tokens = prompt_len,
            completion_tokens,
            prefill_ms = format!("{:.1}", prefill_ms),
            decode_ms = format!("{:.1}", decode_ms),
            total_ms = format!("{:.1}", total_ms),
            decode_tps = format!("{:.1}", decode_tps),
            finish_reason = ?finish_reason,
            "stream generation finished (batch prefill)"
        );

        let usage = Usage {
            prompt_tokens: prompt_len as u32,
            completion_tokens,
            total_tokens: prompt_len as u32 + completion_tokens,
        };
        let metrics = DecodeMetrics {
            ttft_ms,
            prefill_ms,
            decode_ms,
            total_ms,
        };

        let _ = tx.send(StreamEvent::Finished {
            finish_reason: finish_reason.clone(),
            usage: usage.clone(),
            metrics: metrics.clone(),
        });

        let output_text = self
            .tokenizer
            .decode(&output_tokens, true)
            .unwrap_or_default();

        let token_logprobs_final = if token_logprobs.is_empty() {
            None
        } else {
            Some(token_logprobs)
        };
        Ok(GenerateResult {
            model: request.model.clone(),
            output_token_ids: output_tokens,
            output_text,
            finish_reason,
            usage,
            metrics,
            token_logprobs: token_logprobs_final,
        })
    }

    /// Batched streaming decode: all sequences decode together, one token per iteration.
    /// Replaces the sequential per-request `stream_decode_with_blocks`.
    #[cfg(feature = "flash-attn-v3")]
    pub fn batched_stream_decode(
        &self,
        requests: &[&GenerateRequest],
        prefill_results: Vec<BatchPrefillResult>,
        logits_processors: &mut [LogitsProcessor],
        txs: &[&tokio::sync::mpsc::UnboundedSender<StreamEvent>],
        gen_start: Instant,
    ) -> Vec<Result<GenerateResult, EngineError>> {
        let pool = match self.cache.paged_pool.as_ref() {
            Some(p) => p,
            None => {
                return requests
                    .iter()
                    .map(|_| {
                        Err(EngineError::Internal(
                            "batched_stream_decode requires paged pool".into(),
                        ))
                    })
                    .collect();
            }
        };
        let bm_mutex = match self.cache.block_manager.as_ref() {
            Some(b) => b,
            None => {
                return requests
                    .iter()
                    .map(|_| {
                        Err(EngineError::Internal(
                            "batched_stream_decode requires block manager".into(),
                        ))
                    })
                    .collect();
            }
        };

        let n = requests.len();
        let max_context = self.executor.config.max_position_embeddings;

        // Per-sequence state
        struct SeqState {
            block_table: Vec<u32>,
            output_tokens: Vec<u32>,
            sent_text_len: usize,
            prompt_len: usize,
            max_new: usize,
            pos: usize, // next KV position
            finished: bool,
            finish_reason: FinishReason,
            deltanet_slot: Option<u32>,
            logprobs_k: Option<u32>,
            token_logprobs: Vec<TokenLogprobInfo>,
        }

        let mut states: Vec<SeqState> = Vec::with_capacity(n);
        for (i, prefill) in prefill_results.into_iter().enumerate() {
            let max_new = (requests[i].max_new_tokens as usize)
                .min(max_context.saturating_sub(prefill.prompt_len));

            // Send Started + check first token
            let _ = txs[i].send(StreamEvent::Started);

            let mut finished = false;
            let mut finish_reason = FinishReason::Length;
            let mut sent_text_len = 0usize;

            let logprobs_k = requests[i].logprobs;
            let mut token_logprobs: Vec<TokenLogprobInfo> = Vec::new();

            if self.is_eos(prefill.first_token) {
                finished = true;
                finish_reason = FinishReason::Eos;
            } else if self.check_stop_tokens(prefill.first_token, &requests[i].stop.token_ids) {
                finished = true;
                finish_reason = FinishReason::Stop;
            } else {
                // Send first token text + logprobs
                let first_lp = prefill.first_token_logprobs.clone();
                if let Some(ref lp) = first_lp {
                    token_logprobs.push(lp.clone());
                }
                let text = self
                    .tokenizer
                    .decode(&[prefill.first_token], true)
                    .unwrap_or_default();
                if !text.is_empty() {
                    let _ = txs[i].send(StreamEvent::Token {
                        text: text.clone(),
                        logprobs: first_lp,
                    });
                    sent_text_len = text.len();
                }
            }

            states.push(SeqState {
                block_table: prefill.block_table,
                output_tokens: vec![prefill.first_token],
                sent_text_len,
                prompt_len: prefill.prompt_len,
                max_new,
                pos: prefill.prompt_len, // first decode writes KV at this position
                finished,
                finish_reason,
                deltanet_slot: prefill.deltanet_slot,
                logprobs_k,
                token_logprobs,
            });
        }

        // Batched decode loop
        loop {
            // Collect active sequence indices
            let active: Vec<usize> = (0..n)
                .filter(|&i| {
                    !states[i].finished && states[i].output_tokens.len() < states[i].max_new
                })
                .collect();
            if active.is_empty() {
                break;
            }

            // Allocate new blocks where needed
            {
                let mut bm = match bm_mutex.lock() {
                    Ok(bm) => bm,
                    Err(e) => {
                        for &i in &active {
                            states[i].finished = true;
                        }
                        tracing::error!(error = %e, "block manager lock failed in batched decode");
                        break;
                    }
                };
                for &i in &active {
                    if states[i].pos % pool.block_size == 0 {
                        if let Some(new_block) = bm.allocate() {
                            states[i].block_table.push(new_block);
                        } else {
                            states[i].finished = true;
                            // finish_reason stays Length
                        }
                    }
                }
            }

            // Re-filter after potential block exhaustion
            let active: Vec<usize> = active
                .into_iter()
                .filter(|&i| !states[i].finished)
                .collect();
            if active.is_empty() {
                break;
            }

            // Build batch decode input
            let decode_seqs: Vec<BatchDecodeSeq> = active
                .iter()
                .map(|&i| {
                    let s = &states[i];
                    BatchDecodeSeq {
                        token: *s.output_tokens.last().unwrap(),
                        position: s.pos,
                        context_len: s.pos + 1,
                        block_table: &s.block_table,
                        deltanet_slot: s.deltanet_slot,
                    }
                })
                .collect();

            // GPU forward
            let logits_2d = match self.batch_decode_paged(&decode_seqs) {
                Ok(l) => l,
                Err(e) => {
                    tracing::error!(error = %e, "batch_decode_paged failed");
                    for &i in &active {
                        states[i].finished = true;
                    }
                    break;
                }
            };

            // Sample tokens
            let all_greedy = active
                .iter()
                .all(|&i| requests[i].sampling.temperature <= 1e-7);
            let next_tokens: Vec<u32> = if all_greedy {
                match logits_2d
                    .argmax(candle_core::D::Minus1)
                    .and_then(|t| t.to_vec1::<u32>())
                {
                    Ok(tokens) => tokens,
                    Err(e) => {
                        tracing::error!(error = %e, "batch argmax failed");
                        for &i in &active {
                            states[i].finished = true;
                        }
                        break;
                    }
                }
            } else {
                let mut tokens = Vec::with_capacity(active.len());
                for (j, &i) in active.iter().enumerate() {
                    let row = match logits_2d.get(j) {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::error!(error = %e, "get logits row failed");
                            states[i].finished = true;
                            tokens.push(0);
                            continue;
                        }
                    };
                    let token = if requests[i].sampling.temperature <= 1e-7 {
                        match row
                            .argmax(candle_core::D::Minus1)
                            .and_then(|t| t.to_scalar::<u32>())
                        {
                            Ok(t) => t,
                            Err(e) => {
                                tracing::error!(error = %e, "argmax failed");
                                states[i].finished = true;
                                tokens.push(0);
                                continue;
                            }
                        }
                    } else {
                        let row_f32 = match row.to_dtype(DType::F32) {
                            Ok(r) => r,
                            Err(e) => {
                                tracing::error!(error = %e, "to_dtype failed");
                                states[i].finished = true;
                                tokens.push(0);
                                continue;
                            }
                        };
                        match logits_processors[i].sample(&row_f32) {
                            Ok(t) => t,
                            Err(e) => {
                                tracing::error!(error = %e, "sample failed");
                                states[i].finished = true;
                                tokens.push(0);
                                continue;
                            }
                        }
                    };
                    tokens.push(token);
                }
                tokens
            };

            // Process each sequence's new token
            for (j, &i) in active.iter().enumerate() {
                let next_token = next_tokens[j];
                states[i].pos += 1;

                // Extract logprobs before EOS/stop checks (token is still valid)
                let token_lp = if let Some(k) = states[i].logprobs_k {
                    match logits_2d.get(j) {
                        Ok(row) => {
                            Self::extract_top_logprobs(&row, next_token, k, &self.tokenizer).ok()
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                };

                if self.is_eos(next_token) {
                    states[i].finished = true;
                    states[i].finish_reason = FinishReason::Eos;
                    continue;
                }
                if self.check_stop_tokens(next_token, &requests[i].stop.token_ids) {
                    states[i].finished = true;
                    states[i].finish_reason = FinishReason::Stop;
                    continue;
                }

                states[i].output_tokens.push(next_token);
                if let Some(ref lp) = token_lp {
                    states[i].token_logprobs.push(lp.clone());
                }

                // Incremental text streaming
                let text = self
                    .tokenizer
                    .decode(&states[i].output_tokens, true)
                    .unwrap_or_default();
                if text.len() > states[i].sent_text_len {
                    let _ = txs[i].send(StreamEvent::Token {
                        text: text[states[i].sent_text_len..].to_string(),
                        logprobs: token_lp,
                    });
                    states[i].sent_text_len = text.len();
                }

                // Stop string check
                if !requests[i].stop.strings.is_empty()
                    && requests[i].stop.strings.iter().any(|s| text.contains(s))
                {
                    states[i].finished = true;
                    states[i].finish_reason = FinishReason::Stop;
                }
            }
        }

        // Free blocks and DeltaNet pool slots, build results
        let total_ms = gen_start.elapsed().as_secs_f32() * 1000.0;
        let mut results: Vec<Result<GenerateResult, EngineError>> = Vec::with_capacity(n);

        {
            let mut bm = bm_mutex.lock().ok();
            let mut dn_pool = self
                .cache
                .deltanet_pool
                .as_ref()
                .and_then(|m| m.lock().ok());
            for (i, state) in states.into_iter().enumerate() {
                if let Some(ref mut bm) = bm {
                    bm.free(&state.block_table);
                }
                if let (Some(slot), Some(pool)) = (state.deltanet_slot, &mut dn_pool) {
                    pool.free(slot);
                }

                let completion_tokens = state.output_tokens.len() as u32;
                let output_text = self
                    .tokenizer
                    .decode(&state.output_tokens, true)
                    .unwrap_or_default();

                let usage = Usage {
                    prompt_tokens: state.prompt_len as u32,
                    completion_tokens,
                    total_tokens: state.prompt_len as u32 + completion_tokens,
                };
                let metrics = DecodeMetrics {
                    ttft_ms: 0.0, // filled by caller
                    prefill_ms: 0.0,
                    decode_ms: total_ms,
                    total_ms,
                };

                let _ = txs[i].send(StreamEvent::Finished {
                    finish_reason: state.finish_reason.clone(),
                    usage: usage.clone(),
                    metrics: metrics.clone(),
                });

                let token_logprobs = if state.token_logprobs.is_empty() {
                    None
                } else {
                    Some(state.token_logprobs)
                };
                results.push(Ok(GenerateResult {
                    model: requests[i].model.clone(),
                    output_token_ids: state.output_tokens,
                    output_text,
                    finish_reason: state.finish_reason,
                    usage,
                    metrics,
                    token_logprobs,
                }));
            }
        }

        results
    }
}
