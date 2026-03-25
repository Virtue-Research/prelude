use super::super::*;
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
use super::prefill::PrefillForwardResult;

/// Raw GPU output for generation batches (before CPU post-processing).
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
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

    fn build_prefill_plan(
        &self,
        items: &[PreparedGenerateRequest],
        execution_kind: ExecutionKind,
    ) -> PrefillPlan {
        let seq_lens: Vec<usize> = items.iter().map(|item| item.prompt_tokens.len()).collect();
        let all_same_len = seq_lens
            .first()
            .map(|first| seq_lens.iter().all(|len| len == first))
            .unwrap_or(true);
        let all_greedy = items.iter().all(|item| item.is_greedy);
        let prefix_reuse = self.build_prefix_reuse_candidate(items, &seq_lens);

        PrefillPlan {
            execution_kind,
            seq_lens,
            all_same_len,
            all_greedy,
            force_varlen: self.engine_config.runtime.force_varlen_prefill,
            prefix_reuse,
        }
    }

    pub(crate) fn build_decode_plan(
        &self,
        items: &[PreparedGenerateRequest],
    ) -> Result<DecodePlan, EngineError> {
        self.ensure_multi_token_decode_ready()?;
        Ok(DecodePlan {
            initial_prefill: self.build_prefill_plan(
                items,
                ExecutionKind::MultiTokenDecode,
            ),
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
                ExecutionKind::CpuPrefillOnly => {
                    self.execute_cpu_prefill_batch(items)
                }
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
    #[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
    pub(crate) fn prefill_forward_only(
        &self,
        items: Vec<PreparedGenerateRequest>,
    ) -> Result<RawGenerateOutput, EngineError> {
        let batch_start = Instant::now();

        let token_groups: Vec<&[Vec<u32>]> = items
            .iter()
            .map(|item| std::slice::from_ref(&item.prompt_tokens))
            .collect();

        let needs_prompt_logprobs = items.iter().any(|item| item.request.prompt_logprobs.is_some());

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
            let model = self.executor.model.lock()
                .map_err(|e| EngineError::Internal(format!("model lock: {e}")))?;
            let cpu = extract_prompt_logprobs_from_hidden(
                &hidden_states,
                &**model,
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

    #[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
    fn execute_cuda_prefill_only_batch(
        &self,
        items: Vec<PreparedGenerateRequest>,
        _prefill_plan: PrefillPlan,
    ) -> Result<Vec<GenerateResult>, EngineError> {
        let raw = self.prefill_forward_only(items)?;
        generate_postprocess(raw, &self.tokenizer, &self.eos_token_ids)
    }

    #[cfg(not(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer")))]
    fn ensure_multi_token_decode_ready(&self) -> Result<(), EngineError> {
        Err(EngineError::Unavailable(
            "multi-token decode requires flash-attn-v3 (cuda + paged attention)".into(),
        ))
    }

    #[cfg(not(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer")))]
    fn execute_multi_token_batch(
        &self,
        _items: Vec<PreparedGenerateRequest>,
        _decode_plan: DecodePlan,
    ) -> Result<Vec<GenerateResult>, EngineError> {
        Err(EngineError::Unavailable(
            "multi-token decode requires paged attention support".into(),
        ))
    }

    #[cfg(not(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer")))]
    fn generate_stream_paged(
        &self,
        _request: &GenerateRequest,
        _item: PreparedGenerateRequest,
        _tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<GenerateResult, EngineError> {
        Err(EngineError::Unavailable(
            "streaming decode requires paged attention support".into(),
        ))
    }

    #[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
    fn ensure_multi_token_decode_ready(&self) -> Result<(), EngineError> {
        if self.cache.paged_pool.is_none() || self.cache.block_manager.is_none() {
            return Err(EngineError::Unavailable(
                "multi-token decode requires paged attention pool; set PRELUDE_PAGED_ATTN_BLOCKS > 0"
                    .into(),
            ));
        }
        Ok(())
    }

    #[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
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
        let tx_storage: Vec<tokio::sync::mpsc::UnboundedSender<StreamEvent>> = (0
            ..request_refs.len())
            .map(|_| {
                let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
                tx
            })
            .collect();
        let tx_refs: Vec<&tokio::sync::mpsc::UnboundedSender<StreamEvent>> =
            tx_storage.iter().collect();
        let mut logits_processors: Vec<LogitsProcessor> =
            items.into_iter().map(|item| item.logits_processor).collect();
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

    #[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
    pub(crate) fn generate_stream_paged(
        &self,
        request: &GenerateRequest,
        item: PreparedGenerateRequest,
        tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<GenerateResult, EngineError> {
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
        use candle_core::{DType, Tensor};

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
            let cu_seqlens =
                Tensor::from_vec(vec![0u32, seq_len as u32], (2,), device)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
            let position_ids = Tensor::from_vec(
                (0..seq_len as u32).collect::<Vec<_>>(), (seq_len,), device,
            ).map_err(|e| EngineError::Internal(e.to_string()))?;
            let seq_lens_vec = vec![seq_len];

            let mut ctx = crate::models::common::BatchAttnContext {
                cu_seqlens_q: &cu_seqlens,
                max_seqlen_q: seq_len,
                position_ids: &position_ids,
                seq_lens: &seq_lens_vec,
                paged_kv: None,
                deltanet_pool: None,
                deltanet_slots: None,
            };

            let needs_prompt_logprobs = item.request.prompt_logprobs.is_some();

            // When prompt logprobs requested: get hidden states, apply lm_head separately.
            let (logits_flat, prompt_token_logprobs) = if needs_prompt_logprobs {
                let hidden = model.forward_hidden_states(&input, &mut ctx)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let last_hidden = hidden.get(seq_len - 1)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let last_logits = model.compute_logits(&last_hidden)
                    .and_then(|t| t.to_dtype(DType::F32))
                    .map_err(|e| EngineError::Internal(e.to_string()))?;

                // Reuse shared chunked extraction
                let items_slice = std::slice::from_ref(item);
                let seq_lens_slice = [seq_len];
                let logprobs_cpu = extract_prompt_logprobs_from_hidden(
                    &hidden, &**model, items_slice, &seq_lens_slice,
                )?;

                let prompt_tokens = &item.prompt_tokens;
                let plps: Vec<TokenLogprobInfo> = (0..seq_len.saturating_sub(1))
                    .filter_map(|pos| {
                        if pos + 1 < prompt_tokens.len() {
                            let next_token = prompt_tokens[pos + 1];
                            Some(TokenLogprobInfo {
                                token: self.tokenizer.decode(&[next_token], false).unwrap_or_default(),
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
            } else if model.supports_kv_cache() {
                // GGUF and other models with internal KV cache
                let logits = model.forward_with_cache(&input, 0)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                // forward_with_cache returns [L, vocab]; take last token
                let last_logits = logits.get(seq_len - 1)
                    .and_then(|t| t.to_dtype(DType::F32))
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                (last_logits, None)
            } else {
                let logits = model.forward(&input, &mut ctx)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let flat = logits.flatten_all()
                    .and_then(|t| t.to_dtype(DType::F32))
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                (flat, None)
            };
            model.clear_kv_cache();

            let prefill_ms = gen_start.elapsed().as_secs_f32() * 1000.0;

            let _ = self.sample_and_check(
                &logits_flat, logprobs_k, item,
                &mut output_token_ids, &mut output_logprobs, &mut finish_reason,
            )?;

            let total_ms = gen_start.elapsed().as_secs_f32() * 1000.0;
            let decode_ms = total_ms - prefill_ms;
            let text = self.tokenizer.decode(&output_token_ids, true).unwrap_or_default();
            let text = Self::trim_stop_strings(&text, &item.request.stop.strings);
            let completion_tokens = output_token_ids.len() as u32;

            results.push(GenerateResult {
                model: self.model_id.clone(),
                output_token_ids,
                output_text: text,
                finish_reason,
                usage: crate::types::Usage {
                    prompt_tokens: prompt_len as u32,
                    completion_tokens,
                    total_tokens: prompt_len as u32 + completion_tokens,
                },
                metrics: crate::types::DecodeMetrics {
                    total_ms, prefill_ms, decode_ms, ttft_ms: prefill_ms,
                },
                token_logprobs: if output_logprobs.is_empty() { None } else { Some(output_logprobs) },
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
                logits_flat, token_id, k, &self.tokenizer,
            )?);
        }

        output_token_ids.push(token_id);

        if self.eos_token_ids.contains(&token_id)
            || item.request.stop.token_ids.contains(&token_id)
        {
            *finish_reason = crate::types::FinishReason::Stop;
            return Ok(None);
        }
        if !item.request.stop.strings.is_empty() {
            let text_so_far = self.tokenizer.decode(output_token_ids, true).unwrap_or_default();
            if item.request.stop.strings.iter().any(|s| text_so_far.contains(s)) {
                *finish_reason = crate::types::FinishReason::Stop;
                return Ok(None);
            }
        }
        Ok(Some(token_id))
    }

    pub(crate) fn trim_stop_strings(text: &str, stop_strings: &[String]) -> String {
        if stop_strings.is_empty() {
            return text.to_string();
        }
        let mut trimmed = text.to_string();
        for stop_str in stop_strings {
            if let Some(pos) = trimmed.find(stop_str) {
                trimmed.truncate(pos);
            }
        }
        trimmed
    }

    pub(crate) fn build_sampling(request: &GenerateRequest) -> Sampling {
        let temperature = request.sampling.temperature as f64;
        let top_p = request.sampling.top_p as f64;
        if temperature < 1e-7 {
            return Sampling::ArgMax;
        }

        match request.sampling.top_k {
            Some(k) if k > 0 => Sampling::TopKThenTopP {
                k: k as usize,
                p: top_p,
                temperature,
            },
            _ if top_p < 1.0 => Sampling::TopP {
                p: top_p,
                temperature,
            },
            _ => Sampling::All { temperature },
        }
    }

    pub(crate) fn extract_top_logprobs(
        logits: &Tensor,
        sampled_token: u32,
        k: u32,
        tokenizer: &Tokenizer,
    ) -> Result<TokenLogprobInfo, EngineError> {
        let logits_f32 = logits
            .to_dtype(candle_core::DType::F32)
            .map_err(candle_err)?;
        let logits_vec: Vec<f32> = logits_f32.to_vec1().map_err(candle_err)?;
        let vocab_size = logits_vec.len();
        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits_vec.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_sum_exp = max_logit + sum_exp.ln();

        let k = (k as usize).min(vocab_size);
        let mut indexed: Vec<(u32, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v - log_sum_exp))
            .collect();

        if k > 0 && k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k);
        }
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_logprobs: Vec<(u32, String, f32)> = indexed
            .into_iter()
            .map(|(id, lp)| {
                let s = tokenizer.decode(&[id], false).unwrap_or_default();
                (id, s, lp)
            })
            .collect();

        let sampled_logprob = logits_vec
            .get(sampled_token as usize)
            .map(|&v| v - log_sum_exp)
            .unwrap_or(f32::NEG_INFINITY);
        let sampled_token_str = tokenizer
            .decode(&[sampled_token], false)
            .unwrap_or_default();

        Ok(TokenLogprobInfo {
            token: sampled_token_str,
            token_id: sampled_token,
            logprob: sampled_logprob,
            top_logprobs,
        })
    }

    pub(crate) fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    pub(crate) fn check_stop_tokens(&self, token_id: u32, stop_ids: &[u32]) -> bool {
        stop_ids.contains(&token_id)
    }

    // ── CPU continuous decode helpers ─────────────────────────────────

    /// CPU prefill: full prompt through forward_with_cache, returns last-token logits (F32).
    pub(crate) fn cpu_prefill_with_cache(
        &self,
        prompt_tokens: &[u32],
    ) -> Result<Tensor, EngineError> {
        let device = &self.executor.device;
        let seq_len = prompt_tokens.len();
        let input = Tensor::from_vec(prompt_tokens.to_vec(), (seq_len,), device)
            .map_err(|e| EngineError::Internal(e.to_string()))?;
        let mut model = self.executor.model.lock().unwrap();
        model.clear_kv_cache();
        let logits = model
            .forward_with_cache(&input, 0)
            .map_err(|e| EngineError::Internal(e.to_string()))?;
        drop(model);
        logits
            .get(seq_len - 1)
            .and_then(|t| t.to_dtype(DType::F32))
            .map_err(|e| EngineError::Internal(e.to_string()))
    }

    /// CPU decode step: single token through forward_with_cache, returns logits (F32).
    pub(crate) fn cpu_decode_step(
        &self,
        token: u32,
        position_offset: usize,
    ) -> Result<Tensor, EngineError> {
        let device = &self.executor.device;
        let input = Tensor::from_vec(vec![token], (1,), device)
            .map_err(|e| EngineError::Internal(e.to_string()))?;
        let mut model = self.executor.model.lock().unwrap();
        let logits = model
            .forward_with_cache(&input, position_offset)
            .map_err(|e| EngineError::Internal(e.to_string()))?;
        drop(model);
        logits
            .get(0)
            .and_then(|t| t.to_dtype(DType::F32))
            .map_err(|e| EngineError::Internal(e.to_string()))
    }

    /// CPU: clear model's internal KV cache between requests.
    pub(crate) fn cpu_clear_kv_cache(&self) {
        self.executor.model.lock().unwrap().clear_kv_cache();
    }
}

/// CPU post-processing for generation: argmax → logprob extraction → tokenizer decode → result.
/// Standalone function — no &Engine needed, can run on any thread.
///
/// Prompt logprobs arrive as pre-extracted CPU data (`raw.prompt_logprobs_cpu`),
/// computed on the GPU thread in `prefill_forward_only`. No large GPU tensors cross
/// the thread boundary.
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
pub(crate) fn generate_postprocess(
    raw: RawGenerateOutput,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
) -> Result<Vec<GenerateResult>, EngineError> {
    let RawGenerateOutput {
        mut items,
        forward_result,
        batch_start,
        prefill_ms,
        prompt_logprobs_cpu,
    } = raw;
    let batch_size = items.len();

    let logits = forward_result.output;
    let fwd_seq_lens = forward_result.seq_lens;
    let argmax_start = Instant::now();
    let logits_2d = logits.squeeze(1).map_err(candle_err)?;
    let all_tokens = logits_2d
        .argmax(candle_core::D::Minus1)
        .map_err(candle_err)?
        .to_vec1::<u32>()
        .map_err(candle_err)?;
    let argmax_ms = argmax_start.elapsed().as_secs_f32() * 1000.0;

    // Build prompt logprobs per item from pre-extracted CPU data.
    let prompt_logprobs_per_item: Vec<Option<Vec<TokenLogprobInfo>>> = if let Some(ref logprobs_cpu) = prompt_logprobs_cpu {
        let mut per_item = Vec::with_capacity(batch_size);
        let mut offset = 0usize;
        for (i, item) in items.iter().enumerate() {
            let q_len = fwd_seq_lens[i];
            if item.request.prompt_logprobs.is_some() {
                let mut plps = Vec::with_capacity(q_len.saturating_sub(1));
                let prompt_tokens = &item.prompt_tokens;
                for pos in 0..q_len.saturating_sub(1) {
                    let next_token = prompt_tokens[pos + 1];
                    let lp = logprobs_cpu[offset + pos];
                    let token_str = tokenizer.decode(&[next_token], false).unwrap_or_default();
                    plps.push(TokenLogprobInfo {
                        token: token_str,
                        token_id: next_token,
                        logprob: lp,
                        top_logprobs: Vec::new(),
                    });
                }
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

    let pack_start = Instant::now();
    let mut results: Vec<GenerateResult> = Vec::with_capacity(batch_size);
    for (i, item) in items.iter_mut().enumerate() {
        let token = all_tokens[i];
        let finish_reason = if eos_token_ids.contains(&token) {
            FinishReason::Eos
        } else {
            FinishReason::Length
        };
        let token_logprobs = if let Some(k) = item.request.logprobs {
            let row = logits_2d.get(i).map_err(candle_err)?;
            Some(vec![Engine::extract_top_logprobs(
                &row,
                token,
                k,
                tokenizer,
            )?])
        } else {
            None
        };
        let output_text = tokenizer.decode(&[token], true).unwrap_or_default();
        let total_ms = batch_start.elapsed().as_secs_f32() * 1000.0;
        let prompt_len = item.prompt_tokens.len() as u32;

        results.push(GenerateResult {
            model: item.request.model.clone(),
            output_token_ids: vec![token],
            output_text,
            finish_reason,
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                total_tokens: prompt_len + 1,
            },
            metrics: DecodeMetrics {
                ttft_ms: prefill_ms,
                prefill_ms,
                decode_ms: 0.0,
                total_ms,
            },
            token_logprobs,
            prompt_token_logprobs: prompt_logprobs_per_item[i].clone(),
        });
    }
    let pack_ms = pack_start.elapsed().as_secs_f32() * 1000.0;
    let batch_elapsed_ms = batch_start.elapsed().as_secs_f32() * 1000.0;

    tracing::trace!(
        batch_size,
        prefill_ms,
        argmax_ms,
        pack_ms,
        batch_elapsed_ms,
        "max1 varlen postprocess timing"
    );

    Ok(results)
}

/// Extract prompt logprobs from hidden states using chunked compute_logits + log_softmax + gather.
///
/// Applies lm_head in chunks (like vLLM) so the full (total_tokens, vocab_size) logits
/// tensor is never materialized. Peak GPU memory: chunk_size × vocab_size × ~6 bytes.
/// Chunk size for prompt logprobs extraction (tokens processed at a time).
/// Controls peak GPU memory: chunk_size × vocab_size × ~6 bytes.
const PROMPT_LOGPROBS_CHUNK_SIZE: usize = 512;

/// Extract prompt logprobs from hidden states using chunked compute_logits.
///
/// `token_offset`: offset into `item.prompt_tokens` for the start of hidden states
/// (non-zero when prefix caching skips a prefix).
pub(crate) fn extract_prompt_logprobs_from_hidden(
    hidden_states: &Tensor,
    model: &dyn crate::models::ModelForward,
    items: &[PreparedGenerateRequest],
    seq_lens: &[usize],
) -> Result<Vec<f32>, EngineError> {
    extract_prompt_logprobs_from_hidden_offset(hidden_states, model, items, seq_lens, 0)
}

pub(crate) fn extract_prompt_logprobs_from_hidden_offset(
    hidden_states: &Tensor,
    model: &dyn crate::models::ModelForward,
    items: &[PreparedGenerateRequest],
    seq_lens: &[usize],
    token_offset: usize,
) -> Result<Vec<f32>, EngineError> {
    let total_tokens = hidden_states.dim(0).map_err(candle_err)?;
    let device = hidden_states.device().clone();

    // Build flat token_ids: for each position, the next token being predicted.
    let mut flat_next_tokens: Vec<u32> = Vec::with_capacity(total_tokens);
    for (i, item) in items.iter().enumerate() {
        let q_len = seq_lens[i];
        let prompt_tokens = &item.prompt_tokens[token_offset..];
        for pos in 0..q_len {
            if pos + 1 < prompt_tokens.len() {
                flat_next_tokens.push(prompt_tokens[pos + 1]);
            } else {
                flat_next_tokens.push(0);
            }
        }
    }

    // Chunked: compute_logits → log_softmax → gather per chunk.
    // Only chunk_size × vocab_size logits exist at any time.
    let mut logprobs_cpu: Vec<f32> = Vec::with_capacity(total_tokens);
    for start in (0..total_tokens).step_by(PROMPT_LOGPROBS_CHUNK_SIZE) {
        let end = (start + PROMPT_LOGPROBS_CHUNK_SIZE).min(total_tokens);
        let chunk_len = end - start;

        let chunk_hidden = hidden_states.narrow(0, start, chunk_len).map_err(candle_err)?;
        let chunk_logits = model.compute_logits(&chunk_hidden).map_err(candle_err)?;
        let chunk_log_probs = candle_nn::ops::log_softmax(&chunk_logits, 1).map_err(candle_err)?;
        drop(chunk_logits); // free (chunk, vocab_size) before gather allocates

        let chunk_token_ids = Tensor::from_vec(
            flat_next_tokens[start..end].to_vec(), (chunk_len, 1), &device,
        )
        .map_err(candle_err)?
        .to_dtype(candle_core::DType::U32)
        .map_err(candle_err)?;

        let chunk_gathered = chunk_log_probs
            .gather(&chunk_token_ids, 1).map_err(candle_err)?
            .squeeze(1).map_err(candle_err)?
            .to_dtype(candle_core::DType::F32).map_err(candle_err)?;
        logprobs_cpu.extend(chunk_gathered.to_vec1::<f32>().map_err(candle_err)?);
        // chunk_log_probs freed here
    }

    Ok(logprobs_cpu)
}
