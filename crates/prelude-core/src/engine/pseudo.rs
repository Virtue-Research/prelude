use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use tokio::time::sleep;

use super::{EngineError, InferenceEngine};
use crate::constants::PSEUDO_ENGINE_MAX_TOKENS;
use crate::types::{
    ClassificationInputs, ClassificationResult, ClassifyRequest, ClassifyResult, DecodeMetrics,
    FinishReason, GenerateRequest, GenerateResult, ModelInfo, PromptInput, StopConfig, Usage,
};

pub struct PseudoEngine {
    model_info: ModelInfo,
    latency_ms: u64,
}

impl PseudoEngine {
    pub fn new(model_id: impl Into<String>, latency_ms: u64) -> Self {
        Self {
            model_info: ModelInfo {
                id: model_id.into(),
                created: Utc::now().timestamp(),
                owned_by: "prelude".to_string(),
            },
            latency_ms,
        }
    }

    fn generate_pseudo_text(&self, request: &GenerateRequest) -> (String, Vec<u32>, FinishReason) {
        const WORD_BANK: &[&str] = &[
            "prelude",
            "qwen3",
            "token",
            "batch",
            "decode",
            "prefill",
            "cache",
            "kv",
            "cuda",
            "latency",
            "throughput",
            "kernel",
            "scheduler",
            "context",
            "engine",
            "quant",
            "prompt",
            "response",
            "block",
            "memory",
            "pipeline",
            "stable",
            "fast",
            "compact",
            "serving",
            "model",
            "inference",
            "stream",
            "async",
            "ready",
        ];

        let max_tokens = request.max_new_tokens.clamp(1, PSEUDO_ENGINE_MAX_TOKENS) as usize;
        let temperature = request.sampling.temperature.clamp(0.0, 2.0);
        let stride = ((temperature * 10.0).round() as usize).clamp(1, 16);
        let mut seed = seed_from_prompt(
            &request.input,
            &self.model_info.id,
            temperature,
            request.seed,
        );

        let mut out = String::new();
        let mut token_ids = Vec::with_capacity(max_tokens);
        let mut finish_reason = FinishReason::Length;

        for i in 0..max_tokens {
            let idx = ((seed as usize) + i * stride + (i * i + 7 * i)) % WORD_BANK.len();
            let token = WORD_BANK[idx];
            if i > 0 {
                out.push(' ');
            }
            out.push_str(token);
            token_ids.push(idx as u32);

            if let Some(stop_pos) = find_stop_pos(&out, &request.stop) {
                out.truncate(stop_pos);
                finish_reason = FinishReason::Stop;
                break;
            }

            seed = seed.rotate_left(7) ^ (idx as u64 + 1).wrapping_mul(0x9E37_79B1_85EB_CA87);
        }

        (out.trim().to_string(), token_ids, finish_reason)
    }
}

fn seed_from_prompt(
    input: &PromptInput,
    model: &str,
    temperature: f32,
    seed_override: Option<u64>,
) -> u64 {
    if let Some(seed) = seed_override {
        return seed;
    }

    let mut acc = 0xcbf2_9ce4_8422_2325u64;
    match input {
        PromptInput::Text(prompt) => {
            for b in prompt.bytes() {
                acc ^= u64::from(b);
                acc = acc.wrapping_mul(0x100_0000_01b3);
            }
        }
        PromptInput::TokenIds(ids) => {
            for id in ids {
                acc ^= u64::from(*id);
                acc = acc.wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    for b in model.bytes() {
        acc ^= u64::from(b);
        acc = acc.wrapping_mul(0x100_0000_01b3);
    }

    acc ^ u64::from(temperature.to_bits())
}

fn find_stop_pos(text: &str, stop: &StopConfig) -> Option<usize> {
    stop.strings.iter().filter_map(|s| text.find(s)).min()
}

fn prompt_token_count(input: &PromptInput) -> u32 {
    match input {
        PromptInput::Text(text) => text
            .split_whitespace()
            .count()
            .try_into()
            .unwrap_or(u32::MAX),
        PromptInput::TokenIds(ids) => ids.len().try_into().unwrap_or(u32::MAX),
    }
}

#[async_trait]
impl InferenceEngine for PseudoEngine {
    async fn model_info(&self) -> Result<ModelInfo, EngineError> {
        Ok(self.model_info.clone())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, EngineError> {
        Ok(vec![self.model_info.clone()])
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResult, EngineError> {
        let prompt_is_empty = match &request.input {
            PromptInput::Text(text) => text.trim().is_empty(),
            PromptInput::TokenIds(ids) => ids.is_empty(),
        };
        if prompt_is_empty {
            return Err(EngineError::InvalidRequest("prompt is empty".to_string()));
        }

        let (text, token_ids, finish_reason) = self.generate_pseudo_text(&request);

        let extra = (request.max_new_tokens as u64 / 8).min(40);
        sleep(Duration::from_millis(self.latency_ms + extra)).await;

        let prompt_tokens = prompt_token_count(&request.input);
        let completion_tokens = text
            .split_whitespace()
            .count()
            .max(1)
            .try_into()
            .unwrap_or(u32::MAX);
        let total_tokens = prompt_tokens.saturating_add(completion_tokens);

        let total_ms = (self.latency_ms + extra) as f32;
        Ok(GenerateResult {
            model: request.model,
            output_token_ids: token_ids,
            output_text: text,
            finish_reason,
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            },
            metrics: DecodeMetrics {
                ttft_ms: total_ms * 0.5,
                prefill_ms: total_ms * 0.2,
                decode_ms: total_ms * 0.8,
                total_ms,
            },
            token_logprobs: None,
            prompt_token_logprobs: None,
        })
    }

    async fn cancel(&self, _request_id: &str) -> Result<bool, EngineError> {
        Ok(false)
    }

    async fn classify(&self, request: ClassifyRequest) -> Result<ClassifyResult, EngineError> {
        const NUM_CLASSES: u32 = 13;
        const LABELS: &[&str] = &[
            "LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3", "LABEL_4", "LABEL_5", "LABEL_6", "LABEL_7",
            "LABEL_8", "LABEL_9", "LABEL_10", "LABEL_11", "LABEL_12",
        ];

        let texts: Vec<String> = match &request.inputs {
            ClassificationInputs::Texts(texts) => texts.clone(),
            ClassificationInputs::TokenIds(ids) => {
                ids.iter().map(|t| format!("tokens:{}", t.len())).collect()
            }
        };

        if texts.is_empty() {
            return Err(EngineError::InvalidRequest("empty input".to_string()));
        }

        sleep(Duration::from_millis(self.latency_ms)).await;

        let mut results = Vec::with_capacity(texts.len());
        let mut total_tokens = 0u32;

        for (idx, text) in texts.iter().enumerate() {
            let mut seed = 0xcbf2_9ce4_8422_2325u64;
            for b in text.bytes() {
                seed ^= u64::from(b);
                seed = seed.wrapping_mul(0x100_0000_01b3);
            }

            let mut probs = Vec::with_capacity(NUM_CLASSES as usize);
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;

            for i in 0..NUM_CLASSES {
                seed = seed.rotate_left(7) ^ (i as u64 + 1).wrapping_mul(0x9E37_79B1_85EB_CA87);
                let logit = ((seed as f32) / (u64::MAX as f32) * 20.0) - 10.0;
                if logit > max_val {
                    max_val = logit;
                    max_idx = i as usize;
                }
                probs.push(logit);
            }

            let tokens = text.split_whitespace().count() as u32;
            total_tokens += tokens;

            results.push(ClassificationResult {
                index: idx as u32,
                label: Some(LABELS[max_idx].to_string()),
                probs,
                num_classes: NUM_CLASSES,
            });
        }

        Ok(ClassifyResult {
            model: request.model,
            results,
            prompt_tokens: total_tokens,
        })
    }
}
