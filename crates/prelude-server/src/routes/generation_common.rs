use std::convert::Infallible;

use axum::response::sse::Event;
use chrono::Utc;
use prelude_core::{GenerateRequest, PromptInput, SamplingParams, StopConfig};
use serde::Serialize;
use uuid::Uuid;

pub(crate) const DEFAULT_MAX_NEW_TOKENS: u32 = 4096;

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

pub(crate) fn build_generate_request(
    model: String,
    input: PromptInput,
    max_new_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<Vec<String>>,
    seed: Option<u64>,
    logprobs: Option<u32>,
    prompt_logprobs: Option<u32>,
) -> GenerateRequest {
    GenerateRequest {
        request_id: format!("req-{}", Uuid::new_v4().simple()),
        model,
        input,
        sampling: SamplingParams {
            temperature: temperature.unwrap_or(0.7),
            top_p: top_p.unwrap_or(1.0),
            ..SamplingParams::default()
        },
        max_new_tokens,
        stop: StopConfig {
            strings: stop.unwrap_or_default(),
            token_ids: Vec::new(),
        },
        seed,
        deadline_ms: None,
        logprobs,
        prompt_logprobs,
    }
}

pub(crate) fn sse_json_event<T: Serialize>(value: &T) -> Result<Event, Infallible> {
    Ok(Event::default().data(
        serde_json::to_string(value).expect("SSE payload serialization should not fail"),
    ))
}

pub(crate) fn sse_done_event() -> Result<Event, Infallible> {
    Ok(Event::default().data("[DONE]"))
}
