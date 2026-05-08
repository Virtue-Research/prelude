use super::super::*;
use super::postprocess::{extract_prompt_logprobs_from_hidden, generate_postprocess};
use super::prefill::PrefillForwardResult;

/// Raw GPU output for generation batches (before CPU post-processing).
pub(crate) struct RawGenerateOutput {
    pub items: Vec<PreparedGenerateRequest>,
    pub forward_result: PrefillForwardResult,
    pub batch_start: Instant,
    pub prefill_ms: f32,
    /// Pre-extracted prompt logprobs (CPU-side Vec<f32>).
    /// Extracted on the GPU worker thread so that the large all-token logits tensor
    /// is freed before the next request allocates, avoiding cross-thread memory leak.
    pub prompt_logprobs_cpu: Option<Vec<f32>>,
}

impl Engine {
    pub(crate) fn prepare_generate_request(
        &self,
        request: &GenerateRequest,
        request_idx: usize,
    ) -> Result<PreparedGenerateRequest, EngineError> {
        let prompt_tokens = self.tokenize_and_validate(&request.input)?;
        let max_context = self.executor.config.max_position_embeddings;
        let max_new =
            (request.max_new_tokens as usize).min(max_context.saturating_sub(prompt_tokens.len()));
        let sampling = Self::build_sampling(request);
        let is_greedy = matches!(sampling, Sampling::ArgMax);
        let seed = request.seed.unwrap_or(DEFAULT_SEED);
        let logits_processor = LogitsProcessor::from_sampling(seed, sampling);

        Ok(PreparedGenerateRequest {
            request_idx,
            request: request.clone(),
            prompt_tokens,
            max_new,
            is_greedy,
            logits_processor,
        })
    }

    pub(crate) fn build_prefill_plan(
        &self,
        items: &[PreparedGenerateRequest],
        execution_kind: ExecutionKind,
    ) -> PrefillPlan {
        let seq_lens: Vec<usize> = items.iter().map(|item| item.prompt_tokens.len()).collect();
        let all_greedy = items.iter().all(|item| item.is_greedy);
        let prefix_reuse = self.build_prefix_reuse_candidate(items, &seq_lens);

        PrefillPlan {
            execution_kind,
            seq_lens,
            all_greedy,
            prefix_reuse,
            computed_lens: vec![],
            existing_block_tables: vec![],
        }
    }

    pub(crate) fn build_decode_plan(
        &self,
        items: &[PreparedGenerateRequest],
    ) -> Result<DecodePlan, EngineError> {
        self.ensure_multi_token_decode_ready()?;
        Ok(DecodePlan {
            initial_prefill: self.build_prefill_plan(items, ExecutionKind::MultiTokenDecode),
        })
    }

    pub(crate) fn generate_sync(
        &self,
        request: &GenerateRequest,
    ) -> Result<GenerateResult, EngineError> {
        let mut results = self.generate_batch_sync(std::slice::from_ref(request))?;
        results.pop().ok_or_else(|| {
            EngineError::Internal(
                "generate_batch_sync produced no result for single request".into(),
            )
        })
    }

    pub(crate) fn generate_stream_sync(
        &self,
        request: &GenerateRequest,
        tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<GenerateResult, EngineError> {
        self.ensure_task_supported(TaskKind::Generate)?;
        let item = self.prepare_generate_request(request, 0)?;

        if item.max_new <= 1 {
            let result = self.generate_sync(request)?;
            let _ = tx.send(StreamEvent::Started);
            if !result.output_text.is_empty() {
                let _ = tx.send(StreamEvent::Token {
                    text: result.output_text.clone(),
                    logprobs: result
                        .token_logprobs
                        .as_ref()
                        .and_then(|logprobs| logprobs.first().cloned()),
                });
            }
            let _ = tx.send(StreamEvent::Finished {
                finish_reason: result.finish_reason.clone(),
                usage: result.usage.clone(),
                metrics: result.metrics.clone(),
            });
            return Ok(result);
        }

        self.generate_stream_paged(request, item, tx)
    }

    pub(crate) fn generate_batch_sync(
        &self,
        requests: &[GenerateRequest],
    ) -> Result<Vec<GenerateResult>, EngineError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        self.ensure_task_supported(TaskKind::Generate)?;

        let mut items: Vec<PreparedGenerateRequest> = Vec::with_capacity(requests.len());
        for (idx, request) in requests.iter().enumerate() {
            items.push(self.prepare_generate_request(request, idx)?);
        }

        self.execute_generate_batch(items)
    }

    pub(crate) fn execute_generate_batch(
        &self,
        items: Vec<PreparedGenerateRequest>,
    ) -> Result<Vec<GenerateResult>, EngineError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let batch = self.plan_generate_batch(items)?;
        self.generate_prepared_batch(batch)
    }

    pub(crate) fn plan_generate_batch(
        &self,
        items: Vec<PreparedGenerateRequest>,
    ) -> Result<PreparedGenerateBatch, EngineError> {
        let all_prefill_only = items.iter().all(|item| item.max_new <= 1);

        let execution_kind = if !self.executor.device.is_cuda() {
            ExecutionKind::CpuPrefillOnly
        } else if all_prefill_only {
            ExecutionKind::CudaPrefillOnly
        } else {
            self.ensure_multi_token_decode_ready()?;
            ExecutionKind::MultiTokenDecode
        };

        let plan = if execution_kind == ExecutionKind::MultiTokenDecode {
            GenerateBatchPlan::Decode(DecodePlan {
                initial_prefill: self.build_prefill_plan(&items, execution_kind),
            })
        } else {
            GenerateBatchPlan::Prefill(self.build_prefill_plan(&items, execution_kind))
        };

        Ok(PreparedGenerateBatch { plan, items })
    }

    pub(crate) fn generate_prepared_batch(
        &self,
        batch: PreparedGenerateBatch,
    ) -> Result<Vec<GenerateResult>, EngineError> {
        let PreparedGenerateBatch { plan, items } = batch;
        if items.is_empty() {
            return Ok(Vec::new());
        }

        self.ensure_task_supported(TaskKind::Generate)?;

        match plan {
            GenerateBatchPlan::Prefill(prefill_plan) => match prefill_plan.execution_kind {
                ExecutionKind::CudaPrefillOnly => {
                    self.execute_cuda_prefill_only_batch(items, prefill_plan)
                }
                ExecutionKind::CpuPrefillOnly => self.execute_cpu_prefill_batch(items),
                ExecutionKind::MultiTokenDecode => Err(EngineError::Internal(
                    "prefill plan cannot use multi-token decode execution kind".into(),
                )),
            },
            GenerateBatchPlan::Decode(decode_plan) => {
                self.execute_multi_token_batch(items, decode_plan)
            }
        }
    }

    /// GPU-only: runs prefill_pipeline and returns raw logits tensor.
    /// Does NOT do argmax/to_vec1/logprob extraction — those are CPU work.
    ///
    /// When prompt_logprobs is requested, extracts logprobs HERE on the GPU thread
    /// so the large (total_tokens, vocab_size) tensor is freed before the next request.
    /// This prevents cross-thread GPU memory accumulation.
    pub(crate) fn prefill_forward_only(
        &self,
        items: Vec<PreparedGenerateRequest>,
    ) -> Result<RawGenerateOutput, EngineError> {
        let batch_start = Instant::now();

        let token_groups: Vec<&[Vec<u32>]> = items
            .iter()
            .map(|item| std::slice::from_ref(&item.prompt_tokens))
            .collect();

        let needs_prompt_logprobs = items
            .iter()
            .any(|item| item.request.prompt_logprobs.is_some());

        let prefill_start = Instant::now();
        let mut forward_result = if needs_prompt_logprobs {
            self.prefill_pipeline_with_hidden_states(&token_groups)?
        } else {
            self.prefill_pipeline(&token_groups)?
        }
        .ok_or_else(|| EngineError::Internal("empty prefill batch".into()))?;
        let prefill_ms = prefill_start.elapsed().as_secs_f32() * 1000.0;

        // Extract prompt logprobs on the GPU thread by applying compute_logits
        // to hidden states in chunks. This avoids materializing the full
        // (total_tokens, vocab_size) tensor and keeps GPU memory bounded.
        let prompt_logprobs_cpu = if let Some(hidden_states) = forward_result.hidden_states.take() {
            let model = self
                .executor
                .model
                .lock()
                .map_err(|e| EngineError::Internal(format!("model lock: {e}")))?;
            let logits_model = model.as_logits_model().ok_or_else(|| {
                EngineError::Internal("model does not support LogitsSplitModel".into())
            })?;
            let cpu = extract_prompt_logprobs_from_hidden(
                &hidden_states,
                logits_model,
                &items,
                &forward_result.seq_lens,
            )?;
            drop(model);
            drop(hidden_states);
            Some(cpu)
        } else {
            None
        };

        Ok(RawGenerateOutput {
            items,
            forward_result,
            batch_start,
            prefill_ms,
            prompt_logprobs_cpu,
        })
    }

    fn execute_cuda_prefill_only_batch(
        &self,
        items: Vec<PreparedGenerateRequest>,
        _prefill_plan: PrefillPlan,
    ) -> Result<Vec<GenerateResult>, EngineError> {
        let raw = self.prefill_forward_only(items)?;
        generate_postprocess(raw, &self.tokenizer, &self.eos_token_ids)
    }

    fn ensure_multi_token_decode_ready(&self) -> Result<(), EngineError> {
        if self.cache.paged_pool.is_none() || self.cache.block_manager.is_none() {
            return Err(EngineError::Unavailable(
                "multi-token decode requires paged attention pool; set PRELUDE_PAGED_ATTN_BLOCKS > 0"
                    .into(),
            ));
        }
        Ok(())
    }

    fn execute_multi_token_batch(
        &self,
        items: Vec<PreparedGenerateRequest>,
        decode_plan: DecodePlan,
    ) -> Result<Vec<GenerateResult>, EngineError> {
        let mut items = items;
        let gen_start = Instant::now();
        let request_clones: Vec<GenerateRequest> =
            items.iter().map(|item| item.request.clone()).collect();
        let request_refs: Vec<&GenerateRequest> = request_clones.iter().collect();
        let prefill_results = self.batch_prefill_paged(&mut items, &decode_plan.initial_prefill)?;
        let prefill_mss: Vec<f32> = prefill_results
            .iter()
            .map(|prefill| prefill.prefill_ms)
            .collect();
        let tx_storage: Vec<tokio::sync::mpsc::UnboundedSender<StreamEvent>> = (0..request_refs
            .len())
            .map(|_| {
                let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
                tx
            })
            .collect();
        let tx_refs: Vec<&tokio::sync::mpsc::UnboundedSender<StreamEvent>> =
            tx_storage.iter().collect();
        let mut logits_processors: Vec<LogitsProcessor> = items
            .into_iter()
            .map(|item| item.logits_processor)
            .collect();
        self.batched_stream_decode(
            &request_refs,
            prefill_results,
            &mut logits_processors,
            &tx_refs,
            gen_start,
        )
        .into_iter()
        .enumerate()
        .map(|(idx, result)| {
            let mut result = result?;
            let prefill_ms = prefill_mss[idx];
            result.metrics.ttft_ms = prefill_ms;
            result.metrics.prefill_ms = prefill_ms;
            result.metrics.decode_ms = (result.metrics.total_ms - prefill_ms).max(0.0);
            Ok(result)
        })
        .collect()
    }

    pub(crate) fn generate_stream_paged(
        &self,
        request: &GenerateRequest,
        item: PreparedGenerateRequest,
        tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<GenerateResult, EngineError> {
        let _ops_guard = crate::ops::OpsGuard::new(self.executor.ops);
        self.ensure_multi_token_decode_ready()?;
        let gen_start = Instant::now();
        let mut items = vec![item];
        let decode_plan = self.build_decode_plan(&items)?;
        let prefill_results = self.batch_prefill_paged(&mut items, &decode_plan.initial_prefill)?;
        let mut logits_processors = vec![items.pop().unwrap().logits_processor];
        let request_refs = vec![request];
        let tx_refs = vec![&tx];
        let mut results = self.batched_stream_decode(
            &request_refs,
            prefill_results,
            &mut logits_processors,
            &tx_refs,
            gen_start,
        );
        results.pop().unwrap_or_else(|| {
            Err(EngineError::Internal(
                "batched_stream_decode produced no result for single request".into(),
            ))
        })
    }

    /// CPU prefill-only generation (batch runtime path).
    ///
    /// Only handles max_new=1. Multi-token decode goes through cpu_continuous runtime.
    fn execute_cpu_prefill_batch(
        &self,
        items: Vec<PreparedGenerateRequest>,
    ) -> Result<Vec<GenerateResult>, EngineError> {
        use crate::tensor::{DType, Tensor};

        self.ensure_task_supported(TaskKind::Generate)?;

        let device = &self.executor.device;
        let mut model = self.executor.model.lock().unwrap();
        let mut results = Vec::with_capacity(items.len());

        for item in &items {
            let prompt_len = item.prompt_tokens.len();
            let mut output_token_ids = Vec::with_capacity(1);
            let mut output_logprobs: Vec<crate::types::TokenLogprobInfo> = Vec::new();
            let logprobs_k = item.request.logprobs;
            let mut finish_reason = crate::types::FinishReason::Length;
            let gen_start = std::time::Instant::now();

            let seq_len = prompt_len;
            let input = Tensor::from_vec(item.prompt_tokens.clone(), (seq_len,), device)
                .map_err(|e| EngineError::Internal(e.to_string()))?;
            let cu_seqlens = Tensor::from_vec(vec![0u32, seq_len as u32], (2,), device)
                .map_err(|e| EngineError::Internal(e.to_string()))?;
            let position_ids =
                Tensor::from_vec((0..seq_len as u32).collect::<Vec<_>>(), (seq_len,), device)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
            let seq_lens_vec = vec![seq_len];

            let mut ctx = crate::models::commons::BatchAttnContext {
                ops: crate::ops::select_ops(device),
                cu_seqlens_q: &cu_seqlens,
                max_seqlen_q: seq_len,
                position_ids: &position_ids,
                seq_lens: &seq_lens_vec,
                paged_kv: None,
                deltanet_pool: None,
                deltanet_slots: None,
                deltanet_state_is_zero: None,
                deltanet_slots_gpu: None,
            };

            let needs_prompt_logprobs = item.request.prompt_logprobs.is_some();

            // When prompt logprobs requested: get hidden states, apply lm_head separately.
            let (logits_flat, prompt_token_logprobs) = if needs_prompt_logprobs {
                let logits_model = model.as_logits_model_mut().ok_or_else(|| {
                    EngineError::InvalidRequest(
                        "prompt_logprobs requested but model doesn't support LogitsSplitModel"
                            .into(),
                    )
                })?;
                let hidden = logits_model
                    .forward_hidden_states(&input, &mut ctx)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let last_hidden = hidden
                    .get(seq_len - 1)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let last_logits = logits_model
                    .compute_logits(&last_hidden)
                    .and_then(|t| t.to_dtype(DType::F32))
                    .map_err(|e| EngineError::Internal(e.to_string()))?;

                // Reuse shared chunked extraction
                let items_slice = std::slice::from_ref(item);
                let seq_lens_slice = [seq_len];
                let logprobs_cpu = extract_prompt_logprobs_from_hidden(
                    &hidden,
                    logits_model,
                    items_slice,
                    &seq_lens_slice,
                )?;

                let prompt_tokens = &item.prompt_tokens;
                let plps: Vec<TokenLogprobInfo> = (0..seq_len.saturating_sub(1))
                    .filter_map(|pos| {
                        if pos + 1 < prompt_tokens.len() {
                            let next_token = prompt_tokens[pos + 1];
                            Some(TokenLogprobInfo {
                                token: self
                                    .tokenizer
                                    .decode(&[next_token], false)
                                    .unwrap_or_default(),
                                token_id: next_token,
                                logprob: logprobs_cpu[pos],
                                top_logprobs: Vec::new(),
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                (last_logits, if plps.is_empty() { None } else { Some(plps) })
            } else if let Some(kv) = model.as_kv_cache_model() {
                // GGUF and other models with internal KV cache
                let logits = kv
                    .forward_with_cache(&input, 0)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                // forward_with_cache returns [L, vocab]; take last token
                let last_logits = logits
                    .get(seq_len - 1)
                    .and_then(|t| t.to_dtype(DType::F32))
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                (last_logits, None)
            } else {
                let logits = model
                    .forward(&input, &mut ctx)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let flat = logits
                    .flatten_all()
                    .and_then(|t| t.to_dtype(DType::F32))
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                (flat, None)
            };
            model.clear_kv_cache();

            let prefill_ms = gen_start.elapsed().as_secs_f32() * 1000.0;

            let _ = self.sample_and_check(
                &logits_flat,
                logprobs_k,
                item,
                &mut output_token_ids,
                &mut output_logprobs,
                &mut finish_reason,
            )?;

            let total_ms = gen_start.elapsed().as_secs_f32() * 1000.0;
            let decode_ms = total_ms - prefill_ms;
            let text = self
                .tokenizer
                .decode(&output_token_ids, true)
                .unwrap_or_default();
            let text = Self::trim_stop_strings(&text, &item.request.stop.strings);
            let completion_tokens = output_token_ids.len() as u32;

            results.push(GenerateResult {
                model: self.model_id.clone(),
                output_token_ids,
                output_text: text,
                finish_reason,
                usage: crate::types::Usage::new(prompt_len as u32, completion_tokens),
                metrics: crate::types::DecodeMetrics {
                    total_ms,
                    prefill_ms,
                    decode_ms,
                    ttft_ms: prefill_ms,
                },
                token_logprobs: if output_logprobs.is_empty() {
                    None
                } else {
                    Some(output_logprobs)
                },
                prompt_token_logprobs,
            });
        }

        Ok(results)
    }

    /// Shared post-forward step: argmax → logprobs → stop check.
    /// Returns `Some(token_id)` to continue, `None` to stop generation.
    fn sample_and_check(
        &self,
        logits_flat: &Tensor,
        logprobs_k: Option<u32>,
        item: &PreparedGenerateRequest,
        output_token_ids: &mut Vec<u32>,
        output_logprobs: &mut Vec<crate::types::TokenLogprobInfo>,
        finish_reason: &mut crate::types::FinishReason,
    ) -> Result<Option<u32>, EngineError> {
        let token_id = logits_flat
            .argmax(0)
            .and_then(|t| t.to_vec0::<u32>())
            .map_err(|e| EngineError::Internal(e.to_string()))?;

        if let Some(k) = logprobs_k {
            output_logprobs.push(Engine::extract_top_logprobs(
                logits_flat,
                token_id,
                k,
                &self.tokenizer,
            )?);
        }

        output_token_ids.push(token_id);

        if self.eos_token_ids.contains(&token_id) || item.request.stop.token_ids.contains(&token_id)
        {
            *finish_reason = crate::types::FinishReason::Stop;
            return Ok(None);
        }
        if !item.request.stop.strings.is_empty() {
            let text_so_far = self
                .tokenizer
                .decode(output_token_ids, true)
                .unwrap_or_default();
            if item
                .request
                .stop
                .strings
                .iter()
                .any(|s| text_so_far.contains(s))
            {
                *finish_reason = crate::types::FinishReason::Stop;
                return Ok(None);
            }
        }
        Ok(Some(token_id))
    }
}
