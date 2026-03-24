use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use chrono::Utc;
use prelude_core::{
    CompletionChoice, CompletionPrompt, CompletionRequest, CompletionResponse, GenerateRequest,
    InferenceEngine, PromptInput, StreamEvent, Usage,
};
use tracing::info;

use super::generation_common::{
    DEFAULT_MAX_NEW_TOKENS, ResponseMeta, build_generate_request, sse_done_event, sse_json_event,
};
use crate::error::ApiError;
use crate::logprobs::to_completion_logprobs;
use crate::sse::stream_sse;
use crate::utils::{aggregate_usage, log_generation_metrics};
use crate::Server;

pub async fn completions(
    State(server): State<Server>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    request
        .validate_public_request()
        .map_err(|message| ApiError::new(StatusCode::BAD_REQUEST, message, "invalid_request_error"))?;

    let is_streaming = request.stream.unwrap_or(false);
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(false);
    let engine_requests = parse_completion_requests(&request)?;

    info!(
        batch_requests = engine_requests.len(),
        model = %engine_requests
            .first()
            .map(|r| r.model.as_str())
            .unwrap_or("unknown"),
        max_new_tokens = engine_requests.first().map(|r| r.max_new_tokens).unwrap_or(0),
        stream = is_streaming,
        "received completion request"
    );

    if is_streaming {
        if engine_requests.len() != 1 {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "streaming is not supported for batch prompts",
                "invalid_request_error",
            ));
        }
        completions_stream(
            server.engine,
            engine_requests.into_iter().next().unwrap(),
            include_usage,
        )
    } else {
        completions_batch(server.engine, engine_requests).await
    }
}

fn completions_stream(
    engine: Arc<dyn InferenceEngine>,
    request: GenerateRequest,
    include_usage: bool,
) -> Result<Response, ApiError> {
    let response_meta = ResponseMeta::new("cmpl", request.model.clone());
    let req_logprobs = request.logprobs;

    Ok(stream_sse(engine, request, move |event| match event {
        StreamEvent::Started => vec![],
        StreamEvent::Token { text, logprobs } => {
            let logprobs = logprobs.as_ref().and(req_logprobs).map(|_| {
                to_completion_logprobs(std::slice::from_ref(logprobs.as_ref().unwrap()), &text)
            });
            let chunk = CompletionResponse {
                id: response_meta.id.clone(),
                object: "text_completion".to_string(),
                created: response_meta.created,
                model: response_meta.model.clone(),
                choices: vec![CompletionChoice {
                    text,
                    index: 0,
                    finish_reason: String::new(),
                    logprobs,
                    prompt_logprobs: None,
                    prompt_token_ids: None,
                }],
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                },
                system_fingerprint: None,
            };
            vec![sse_json_event(&chunk)]
        }
        StreamEvent::Finished {
            finish_reason,
            usage,
            metrics,
        } => {
            let finish_reason = finish_reason.as_openai_str().to_string();
            log_generation_metrics(
                &usage,
                &metrics,
                &finish_reason,
                "completion stream completed",
            );

            let mut events = Vec::new();
            events.push(sse_json_event(&CompletionResponse {
                id: response_meta.id.clone(),
                object: "text_completion".to_string(),
                created: response_meta.created,
                model: response_meta.model.clone(),
                choices: vec![CompletionChoice {
                    text: String::new(),
                    index: 0,
                    finish_reason,
                    logprobs: None,
                    prompt_logprobs: None,
                    prompt_token_ids: None,
                }],
                usage: if include_usage {
                    usage.clone()
                } else {
                    Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    }
                },
                system_fingerprint: None,
            }));

            if include_usage {
                events.push(sse_json_event(&CompletionResponse {
                    id: response_meta.id.clone(),
                    object: "text_completion".to_string(),
                    created: response_meta.created,
                    model: response_meta.model.clone(),
                    choices: vec![],
                    usage,
                    system_fingerprint: None,
                }));
            }

            events.push(sse_done_event());
            events
        }
        StreamEvent::Error { message } => {
            tracing::error!(error = %message, "stream generation error");
            vec![sse_done_event()]
        }
    }))
}

async fn completions_batch(
    engine: Arc<dyn InferenceEngine>,
    requests: Vec<GenerateRequest>,
) -> Result<Response, ApiError> {
    let start = Instant::now();
    let results = engine
        .generate_batch(requests)
        .await
        .map_err(ApiError::from)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let model = results.first().map(|r| r.model.clone()).unwrap_or_default();
    let usage = aggregate_usage(&results);
    let choices: Vec<CompletionChoice> = results
        .into_iter()
        .enumerate()
        .map(|(idx, result)| {
            let logprobs = result
                .token_logprobs
                .as_ref()
                .map(|infos| to_completion_logprobs(infos, &result.output_text));
            let (prompt_logprobs, prompt_token_ids) =
                crate::logprobs::to_prompt_logprobs_response(&result);
            CompletionChoice {
                text: result.output_text,
                index: idx as u32,
                finish_reason: result.finish_reason.as_openai_str().to_string(),
                logprobs,
                prompt_logprobs,
                prompt_token_ids,
            }
        })
        .collect();

    info!(
        batch_requests = choices.len(),
        prompt_tokens = usage.prompt_tokens,
        completion_tokens = usage.completion_tokens,
        total_tokens = usage.total_tokens,
        elapsed_ms = format!("{:.1}", elapsed_ms),
        "completion batch completed"
    );

    Ok(Json(CompletionResponse {
        id: ResponseMeta::new("cmpl", model.clone()).id,
        object: "text_completion".to_string(),
        created: Utc::now().timestamp(),
        model,
        choices,
        usage,
        system_fingerprint: None,
    })
    .into_response())
}

fn parse_completion_requests(
    request: &CompletionRequest,
) -> Result<Vec<GenerateRequest>, ApiError> {
    let prompts: Vec<&str> = match &request.prompt {
        CompletionPrompt::Single(prompt) => vec![prompt.as_str()],
        CompletionPrompt::Batch(prompts) => prompts.iter().map(|s| s.as_str()).collect(),
    };
    if prompts.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "prompt batch is empty",
            "invalid_request_error",
        ));
    }

    let mut out = Vec::with_capacity(prompts.len());
    for p in prompts {
        if p.trim().is_empty() {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "prompt is empty",
                "invalid_request_error",
            ));
        }

        out.push(build_generate_request(
            request.model.clone(),
            PromptInput::Text(p.to_string()),
            request.max_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS),
            request.temperature,
            request.top_p,
            request.stop.clone(),
            request.seed,
            request.logprobs,
            request.prompt_logprobs,
        ));
    }

    Ok(out)
}
