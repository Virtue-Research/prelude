use std::convert::Infallible;

use axum::response::sse::Event;
use chrono::Utc;
use prelude_core::{GenerateRequest, PromptInput, SamplingParams, StopConfig, Usage};
use serde::Serialize;
use uuid::Uuid;

pub(crate) const DEFAULT_MAX_NEW_TOKENS: u32 = 4096;

pub(crate) struct GenerateRequestParams {
    pub model: String,
    pub input: PromptInput,
    pub max_new_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<u64>,
    pub logprobs: Option<u32>,
    pub prompt_logprobs: Option<u32>,
}

#[derive(Clone)]
pub(crate) struct ResponseMeta {
    pub id: String,
    pub created: i64,
    pub model: String,
}

impl ResponseMeta {
    pub fn new(prefix: &str, model: impl Into<String>) -> Self {
        Self {
            id: format!("{prefix}-{}", Uuid::new_v4().simple()),
            created: Utc::now().timestamp(),
            model: model.into(),
        }
    }
}

pub(crate) fn build_generate_request(params: GenerateRequestParams) -> GenerateRequest {
    GenerateRequest {
        request_id: format!("req-{}", Uuid::new_v4().simple()),
        model: params.model,
        input: params.input,
        sampling: SamplingParams {
            temperature: params.temperature.unwrap_or(0.7),
            top_p: params.top_p.unwrap_or(1.0),
            ..SamplingParams::default()
        },
        max_new_tokens: params.max_new_tokens,
        stop: StopConfig {
            strings: params.stop.unwrap_or_default(),
            token_ids: Vec::new(),
        },
        seed: params.seed,
        deadline_ms: None,
        logprobs: params.logprobs,
        prompt_logprobs: params.prompt_logprobs,
    }
}

pub(crate) fn empty_usage() -> Usage {
    Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    }
}

pub(crate) fn sse_json_event<T: Serialize>(value: &T) -> Result<Event, Infallible> {
    Ok(Event::default()
        .data(serde_json::to_string(value).expect("SSE payload serialization should not fail")))
}

pub(crate) fn sse_done_event() -> Result<Event, Infallible> {
    Ok(Event::default().data("[DONE]"))
}
