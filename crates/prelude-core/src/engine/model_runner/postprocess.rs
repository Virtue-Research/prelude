use super::super::*;
use super::generate::RawGenerateOutput;

impl Engine {
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

    /// Extract the sampled token's logprob, plus the top-`k` alternative
    /// (token_id, decoded string, logprob) entries sorted descending.
    ///
    /// `k == 0` means the caller asked for `logprobs: true` without
    /// `top_logprobs` — only the chosen token's logprob is needed.
    /// In that case we skip the full-vocab sort and tokenize step,
    /// which on a 152k-vocab Qwen3 model was the dominant hot path:
    /// previously the function fell through to `sort_by` over every
    /// vocab entry and then ran `tokenizer.decode` 152k times per
    /// request (~33 ms each on real classifier traffic with k=0),
    /// because the `if k > 0 && k < indexed.len()` guard only gated
    /// `select_nth_unstable_by` — not the sort or the decode loop.
    pub(crate) fn extract_top_logprobs(
        logits: &Tensor,
        sampled_token: u32,
        k: u32,
        tokenizer: &Tokenizer,
    ) -> Result<TokenLogprobInfo, EngineError> {
        let logits_f32 = logits
            .to_dtype(crate::tensor::DType::F32)
            .map_err(tensor_err)?;
        let logits_vec: Vec<f32> = logits_f32.to_vec1().map_err(tensor_err)?;
        let vocab_size = logits_vec.len();
        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits_vec.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_sum_exp = max_logit + sum_exp.ln();

        let sampled_logprob = logits_vec
            .get(sampled_token as usize)
            .map(|&v| v - log_sum_exp)
            .unwrap_or(f32::NEG_INFINITY);
        let sampled_token_str = tokenizer
            .decode(&[sampled_token], false)
            .unwrap_or_default();

        // Top-k path runs only when the caller asked for alternatives.
        // For k=0 the response's `top_logprobs` is an empty list and we
        // can skip the O(N log N) sort + 152k tokenizer.decode loop
        // entirely. Use `select_nth_unstable_by` then a k-sized sort —
        // that's O(N + k log k), independent of vocab_size beyond the
        // linear scan we already paid for log_sum_exp.
        let top_logprobs: Vec<(u32, String, f32)> = if k == 0 {
            Vec::new()
        } else {
            let k_capped = (k as usize).min(vocab_size);
            let mut indexed: Vec<(u32, f32)> = logits_vec
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v - log_sum_exp))
                .collect();
            if k_capped < indexed.len() {
                indexed.select_nth_unstable_by(k_capped, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                indexed.truncate(k_capped);
            }
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed
                .into_iter()
                .map(|(id, lp)| {
                    let s = tokenizer.decode(&[id], false).unwrap_or_default();
                    (id, s, lp)
                })
                .collect()
        };

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
}

/// CPU post-processing for generation: argmax → logprob extraction → tokenizer decode → result.
/// Standalone function — no &Engine needed, can run on any thread.
///
/// Prompt logprobs arrive as pre-extracted CPU data (`raw.prompt_logprobs_cpu`),
/// computed on the GPU thread in `prefill_forward_only`. No large GPU tensors cross
/// the thread boundary.
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
    let logits_2d = logits.squeeze(1).map_err(tensor_err)?;
    let all_tokens = logits_2d
        .argmax(crate::tensor::D::Minus1)
        .map_err(tensor_err)?
        .to_vec1::<u32>()
        .map_err(tensor_err)?;
    let argmax_ms = argmax_start.elapsed().as_secs_f32() * 1000.0;

    // Build prompt logprobs per item from pre-extracted CPU data.
    let prompt_logprobs_per_item: Vec<Option<Vec<TokenLogprobInfo>>> =
        if let Some(ref logprobs_cpu) = prompt_logprobs_cpu {
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
            let row = logits_2d.get(i).map_err(tensor_err)?;
            Some(vec![Engine::extract_top_logprobs(
                &row, token, k, tokenizer,
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
            usage: Usage::new(prompt_len, 1),
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
    model: &dyn crate::models::LogitsSplitModel,
    items: &[PreparedGenerateRequest],
    seq_lens: &[usize],
) -> Result<Vec<f32>, EngineError> {
    extract_prompt_logprobs_from_hidden_offset(hidden_states, model, items, seq_lens, 0)
}

pub(crate) fn extract_prompt_logprobs_from_hidden_offset(
    hidden_states: &Tensor,
    model: &dyn crate::models::LogitsSplitModel,
    items: &[PreparedGenerateRequest],
    seq_lens: &[usize],
    token_offset: usize,
) -> Result<Vec<f32>, EngineError> {
    let total_tokens = hidden_states.dim(0).map_err(tensor_err)?;
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

    // Chunked: compute_logits → gather_log_softmax per chunk.
    //
    // The chunking is retained to cap peak GPU memory: the `lm_head`
    // matmul output is `[chunk_len, vocab_size]` which at chunk_len=512
    // and vocab=152K is ~155 MB per chunk in BF16. For a full prompt
    // logprobs extract over a 32K-token context, doing it in one shot
    // would need ~20 GB of contiguous GPU memory.
    //
    // Inside each chunk we route through the fused `gather_log_softmax`
    // op instead of the naive `log_softmax → gather` chain. That
    // eliminates the per-chunk `[chunk_len, vocab_size] F32` temporary
    // (another ~311 MB per chunk on Qwen3.5-35B-A3B) and matches vLLM's
    // `_topk_log_softmax_kernel` asymptote (two full-vocab reads + one
    // scalar write per token).
    //
    // Fallback path: when the backend declines the fused op (CPU or
    // any backend without a `gather_log_softmax` impl) we drop back to
    // the old `log_softmax + gather` chain — still O(chunk_len) D2H,
    // just with the full-matrix temporary.
    let ops = crate::ops::ops_for(hidden_states.device());
    let mut logprobs_cpu: Vec<f32> = Vec::with_capacity(total_tokens);
    for start in (0..total_tokens).step_by(PROMPT_LOGPROBS_CHUNK_SIZE) {
        let end = (start + PROMPT_LOGPROBS_CHUNK_SIZE).min(total_tokens);
        let chunk_len = end - start;

        let chunk_hidden = hidden_states
            .narrow(0, start, chunk_len)
            .map_err(tensor_err)?;
        let chunk_logits = model.compute_logits(&chunk_hidden).map_err(tensor_err)?;

        let chunk_target_ids =
            Tensor::from_vec(flat_next_tokens[start..end].to_vec(), (chunk_len,), &device)
                .map_err(tensor_err)?;

        let chunk_logprobs_tensor = match ops.gather_log_softmax(&chunk_logits, &chunk_target_ids) {
            Some(res) => res.map_err(tensor_err)?,
            None => {
                // Fallback: materialise the log_softmax temporary, do
                // the on-device gather, same O(chunk_len) D2H.
                let chunk_log_probs = ops.log_softmax(&chunk_logits, 1).map_err(tensor_err)?;
                drop(chunk_logits);
                let idx = chunk_target_ids
                    .reshape((chunk_len, 1))
                    .map_err(tensor_err)?;
                chunk_log_probs
                    .gather(&idx, 1)
                    .map_err(tensor_err)?
                    .squeeze(1)
                    .map_err(tensor_err)?
                    .to_dtype(crate::tensor::DType::F32)
                    .map_err(tensor_err)?
            }
        };

        logprobs_cpu.extend(chunk_logprobs_tensor.to_vec1::<f32>().map_err(tensor_err)?);
    }

    Ok(logprobs_cpu)
}
