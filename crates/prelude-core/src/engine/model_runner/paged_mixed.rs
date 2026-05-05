use crate::engine::executor::{ModelOutput, StepRequest};
use crate::engine::*;
use crate::models::commons::BatchAttnContext;
use crate::models::commons::PagedKvBatchContext;

impl Engine {
    /// Unified forward pass: prefill chunks (Q>1) and decode tokens (Q=1) in one
    /// varlen batch. One model.forward() call handles everything.
    ///
    /// Returns ModelOutput with per-request logits (last token of each request's Q).
    /// Block allocation for prefill is handled here; decode block allocation must
    /// be done by the caller before building StepRequests.
    pub(crate) fn batch_mixed_paged(
        &self,
        requests: &[StepRequest],
    ) -> Result<ModelOutput, EngineError> {
        let pool = self.cache.paged_pool.as_ref().ok_or_else(|| {
            EngineError::Internal("batch_mixed_paged requires paged attention pool".into())
        })?;

        let num_requests = requests.len();
        if num_requests == 0 {
            return Ok(ModelOutput {
                logits: Tensor::zeros((0, 0), DType::F32, &Device::Cpu).map_err(tensor_err)?,
                item_seq_counts: vec![],
                prefill_results: vec![],
                sampled_tokens: None,
            });
        }

        // ── Build packed varlen input ──────────────────────────────────
        let mut flat_tokens: Vec<u32> = Vec::new();
        let mut cu_seqlens_q = vec![0u32];
        let mut cu_seqlens_k = vec![0u32];
        let mut q_seq_lens: Vec<usize> = Vec::with_capacity(num_requests);
        let mut position_ids: Vec<u32> = Vec::new();
        let mut max_seqlen_q = 0usize;
        let mut max_seqlen_k = 0usize;

        for req in requests {
            let q_len = req.tokens.len();
            flat_tokens.extend_from_slice(&req.tokens);
            cu_seqlens_q.push(cu_seqlens_q.last().unwrap() + q_len as u32);
            cu_seqlens_k.push(cu_seqlens_k.last().unwrap() + req.context_len as u32);
            q_seq_lens.push(q_len);
            max_seqlen_q = max_seqlen_q.max(q_len);
            max_seqlen_k = max_seqlen_k.max(req.context_len);

            for pos in 0..q_len {
                position_ids.push((req.position_start + pos) as u32);
            }
        }

        let total_tokens = flat_tokens.len();

        // Debug: log batch details for the first few steps
        for (i, req) in requests.iter().enumerate() {
            tracing::debug!(
                req_idx = i,
                q_len = req.tokens.len(),
                context_len = req.context_len,
                position_start = req.position_start,
                block_table_len = req.block_table.len(),
                is_prefill_final = req.is_prefill_final,
                is_prefill_partial = req.is_prefill_partial,
                needs_kv_cache = req.needs_kv_cache,
                first_token = req.tokens.first().copied().unwrap_or(0),
                last_token = req.tokens.last().copied().unwrap_or(0),
                prompt_logprobs = ?req.prompt_logprobs,
                "batch_mixed_paged request"
            );
        }

        let packed_input = Tensor::from_vec(flat_tokens, (total_tokens,), &self.executor.device)
            .map_err(tensor_err)?;
        let cu_seqlens_q_t =
            Tensor::from_vec(cu_seqlens_q, (num_requests + 1,), &self.executor.device)
                .map_err(tensor_err)?;
        let cu_seqlens_k_t =
            Tensor::from_vec(cu_seqlens_k, (num_requests + 1,), &self.executor.device)
                .map_err(tensor_err)?;
        let position_ids_t = Tensor::from_vec(position_ids, (total_tokens,), &self.executor.device)
            .map_err(tensor_err)?;

        // ── Block tables + slot mapping ───────────────────────────────
        let use_paged_kv = requests.iter().any(|r| r.needs_kv_cache);
        let block_tables_t = if use_paged_kv {
            let max_blocks = requests
                .iter()
                .map(|r| r.block_table.len())
                .max()
                .unwrap_or(0);
            let mut flat_bt: Vec<u32> = Vec::with_capacity(num_requests * max_blocks);
            for req in requests {
                flat_bt.extend_from_slice(&req.block_table);
                for _ in req.block_table.len()..max_blocks {
                    flat_bt.push(0);
                }
            }
            if max_blocks > 0 {
                Tensor::from_vec(flat_bt, (num_requests, max_blocks), &self.executor.device)
                    .map_err(tensor_err)?
                    .to_dtype(DType::U32)
                    .map_err(tensor_err)?
            } else {
                Tensor::zeros((num_requests, 0), DType::U32, &self.executor.device)
                    .map_err(tensor_err)?
            }
        } else {
            Tensor::zeros((num_requests, 0), DType::U32, &self.executor.device)
                .map_err(tensor_err)?
        };

        let slot_mapping_t = if use_paged_kv {
            let mut slots: Vec<i64> = Vec::with_capacity(total_tokens);
            for req in requests {
                for t in 0..req.tokens.len() {
                    slots.push(crate::cache::block_manager::BlockManager::slot(
                        &req.block_table,
                        req.position_start + t,
                        pool.block_size,
                    ));
                }
            }
            Tensor::new(slots.as_slice(), &self.executor.device).map_err(tensor_err)?
        } else {
            Tensor::zeros((0,), DType::I64, &self.executor.device).map_err(tensor_err)?
        };

        // ── DeltaNet slots ────────────────────────────────────────────
        let deltanet_slots: Option<Vec<u32>> = if self.cache.deltanet_pool.is_some() {
            let slots: Vec<u32> = requests.iter().filter_map(|r| r.deltanet_slot).collect();
            if slots.len() == num_requests {
                Some(slots)
            } else {
                None
            }
        } else {
            None
        };
        let deltanet_slots_gpu = match deltanet_slots.as_ref() {
            Some(slots) if self.executor.device.is_cuda() => Some(
                Tensor::from_vec(slots.clone(), (slots.len(),), &self.executor.device)
                    .map_err(tensor_err)?,
            ),
            _ => None,
        };

        // ── Forward ───────────────────────────────────────────────────
        let forward_start = Instant::now();
        let mut model = self
            .executor
            .model
            .lock()
            .map_err(|e| EngineError::Internal(format!("model lock poisoned: {e}")))?;

        let mut dn_pool_guard = self.cache.deltanet_pool.as_ref().map(|m| m.lock().unwrap());
        let dn_pool_ref = dn_pool_guard.as_deref_mut();

        let paged_kv = if use_paged_kv {
            Some(PagedKvBatchContext {
                key_caches: &pool.active_key_caches(),
                value_caches: &pool.active_value_caches(),
                slot_mapping: &slot_mapping_t,
                block_tables: &block_tables_t,
                cu_seqlens_k: &cu_seqlens_k_t,
                max_seqlen_k,
            })
        } else {
            None
        };
        let mut ctx = BatchAttnContext {
            ops: self.executor.ops,
            cu_seqlens_q: &cu_seqlens_q_t,
            max_seqlen_q,
            position_ids: &position_ids_t,
            seq_lens: &q_seq_lens,
            paged_kv: paged_kv.as_ref(),
            deltanet_pool: dn_pool_ref,
            deltanet_slots: deltanet_slots.as_deref(),
            deltanet_slots_gpu: deltanet_slots_gpu.as_ref(),
        };

        let needs_prompt_logprobs = requests.iter().any(|r| r.prompt_logprobs.is_some());

        self.executor.ops.begin_forward();

        // When prompt logprobs needed: use forward_hidden_states path to get
        // all-token hidden states, then extract per-token logprobs.
        let (logits_2d, prompt_logprobs_data) = if needs_prompt_logprobs {
            let lm = model.as_logits_model_mut().ok_or_else(|| {
                EngineError::InvalidRequest(
                    "prompt_logprobs requested but model doesn't support LogitsSplitModel".into(),
                )
            })?;
            let hidden = lm
                .forward_hidden_states(&packed_input, &mut ctx)
                .map_err(tensor_err)?;
            let last_hidden = crate::models::commons::last_token_select(&hidden, &q_seq_lens)
                .map_err(tensor_err)?;
            let last_logits = lm.compute_logits(&last_hidden).map_err(tensor_err)?;

            // Extract per-token logprobs for prefill requests from hidden
            // states.
            //
            // Pre-history: this loop used to materialise the full
            // `[q_len, vocab_size]` log_softmax matrix on the GPU, copy
            // ALL of it to the CPU via `to_vec2::<f32>()`, and then index
            // into it position-by-position. For Qwen3.5-35B-A3B with
            // q_len=1024 and vocab_size≈151_936 that's ~622 MB of D2H
            // traffic per chunk — 291 chunks of a full WikiText-2 run
            // was ~180 GB of pointless PCIe bandwidth and a ton of heap
            // churn from allocating 1024 inner `Vec<f32>`s per chunk.
            // Only the `q_len - 1` entries indexed by the next-token
            // IDs are actually used.
            //
            // Fix: route through `ops.gather_log_softmax`, which on CUDA
            // dispatches to a fused PTX kernel that mirrors vLLM's
            // `_topk_log_softmax_kernel` — two full-vocab reads, one
            // tiny scalar write per token, **no `[T, V]` log_softmax
            // temporary materialised**. The D2H copy shrinks from 622
            // MB to ~4 KB per chunk. For backends that don't implement
            // the fused op (returns `None`) we fall through to an
            // on-device gather-after-log_softmax path that still keeps
            // D2H at O(q_len) but materialises the full matrix.
            let hidden_device = hidden.device().clone();
            let mut all_logprobs: Vec<Option<Vec<f32>>> = Vec::with_capacity(num_requests);
            let mut token_offset = 0usize;
            for req in requests.iter() {
                let q_len = req.tokens.len();
                if req.prompt_logprobs.is_some() && q_len > 1 {
                    // Span of hidden states corresponding to this
                    // request's prefill tokens. Drop the last position —
                    // it has no "next token" to look up.
                    let span_hidden = hidden
                        .narrow(0, token_offset, q_len - 1)
                        .map_err(tensor_err)?;
                    let span_logits = lm.compute_logits(&span_hidden).map_err(tensor_err)?;

                    // Next-token ids as a `[q_len - 1]` U32 tensor on
                    // the same device as the logits.
                    let next_tokens: Vec<u32> = req.tokens[1..q_len].iter().copied().collect();
                    let target_ids = Tensor::from_vec(next_tokens, (q_len - 1,), &hidden_device)
                        .map_err(tensor_err)?;

                    let token_logprobs_tensor = match self
                        .executor
                        .ops
                        .gather_log_softmax(&span_logits, &target_ids)
                    {
                        Some(res) => res.map_err(tensor_err)?,
                        None => {
                            // Fallback: materialise the full log_softmax
                            // on the device, gather on GPU, tiny D2H.
                            // Slower than the fused kernel (allocates a
                            // full `[q_len - 1, vocab] F32` temp) but
                            // still keeps D2H at O(q_len).
                            let span_logits_f32 =
                                span_logits.to_dtype(DType::F32).map_err(tensor_err)?;
                            let log_probs = candle_nn::ops::log_softmax(
                                &span_logits_f32,
                                crate::tensor::D::Minus1,
                            )
                            .map_err(tensor_err)?;
                            let idx = target_ids.reshape((q_len - 1, 1)).map_err(tensor_err)?;
                            log_probs
                                .gather(&idx, 1)
                                .map_err(tensor_err)?
                                .flatten_all()
                                .map_err(tensor_err)?
                        }
                    };
                    let token_logprobs =
                        token_logprobs_tensor.to_vec1::<f32>().map_err(tensor_err)?;
                    all_logprobs.push(Some(token_logprobs));
                } else {
                    all_logprobs.push(None);
                }
                token_offset += q_len;
            }
            drop(hidden);
            (last_logits, all_logprobs)
        } else {
            let logits = model.forward(&packed_input, &mut ctx).map_err(tensor_err)?;
            let logits_2d = logits.squeeze(1).map_err(tensor_err)?;
            let empty: Vec<Option<Vec<f32>>> = vec![None; num_requests];
            (logits_2d, empty)
        };

        self.executor.ops.end_forward();
        drop(dn_pool_guard);
        drop(model);

        let forward_ms = forward_start.elapsed().as_secs_f32() * 1000.0;

        // ── Build per-request results ─────────────────────────────────
        let mut prefill_results: Vec<BatchPrefillResult> = Vec::new();
        for (i, req) in requests.iter().enumerate() {
            if req.is_prefill_final || req.is_prefill_partial {
                // Build prompt token logprobs if requested
                let prompt_token_logprobs = if let Some(raw_lps) = &prompt_logprobs_data[i] {
                    Some(
                        raw_lps
                            .iter()
                            .enumerate()
                            .map(|(pos, &lp)| {
                                let next_tok = req.tokens[pos + 1];
                                TokenLogprobInfo {
                                    token: self
                                        .tokenizer
                                        .decode(&[next_tok], false)
                                        .unwrap_or_default(),
                                    token_id: next_tok,
                                    logprob: lp,
                                    top_logprobs: Vec::new(),
                                }
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                prefill_results.push(BatchPrefillResult {
                    // The AR loop samples from ModelOutput.logits. Avoid a
                    // duplicate per-request argmax and D2H sync here.
                    first_token: 0,
                    block_table: req.block_table.clone(),
                    prompt_len: req.position_start + req.tokens.len(),
                    prefill_ms: forward_ms,
                    deltanet_slot: req.deltanet_slot,
                    first_token_logprobs: None,
                    prompt_token_logprobs,
                });
            }
        }

        info!(
            num_requests,
            total_tokens,
            forward_ms = format!("{:.1}", forward_ms),
            prefill_count = prefill_results.len(),
            "batch_mixed_paged complete"
        );

        Ok(ModelOutput {
            logits: logits_2d,
            item_seq_counts: vec![],
            prefill_results,
            sampled_tokens: None,
        })
    }
}
