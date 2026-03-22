use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use prelude_core::{
    ChatCompletionChoice, ChatCompletionLogprobs, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessageOut, GenerateRequest, InferenceEngine, PromptInput, StreamEvent,
};
use tracing::info;

use super::generation_common::{
    DEFAULT_MAX_NEW_TOKENS, ResponseMeta, build_generate_request, sse_done_event, sse_json_event,
};
use crate::error::ApiError;
use crate::logprobs::{to_chat_logprob_content, to_chat_logprobs};
use crate::sse::stream_sse;
use crate::utils::log_generation_metrics;
use crate::Server;

pub async fn chat_completions(
    State(server): State<Server>,
    Json(request): Json<ChatCompletionRequest>,
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
    let engine_request = parse_chat_request(&request, server.chat_template.as_deref())?;

    info!(
        rid = %engine_request.request_id,
        model = %engine_request.model,
        max_new_tokens = engine_request.max_new_tokens,
        stream = is_streaming,
        "received chat request"
    );

    if is_streaming {
        chat_stream(server.engine, engine_request, include_usage)
    } else {
        chat_batch(server.engine, engine_request).await
    }
}

fn chat_stream(
    engine: Arc<dyn InferenceEngine>,
    request: GenerateRequest,
    include_usage: bool,
) -> Result<Response, ApiError> {
    let response_meta = ResponseMeta::new("chatcmpl", request.model.clone());

    Ok(stream_sse(engine, request, move |event| match event {
        StreamEvent::Started => {
            let chunk = ChatCompletionResponse {
                id: response_meta.id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: response_meta.created,
                model: response_meta.model.clone(),
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: None,
                    delta: Some(ChatMessageOut {
                        role: "assistant".to_string(),
                        content: Some(String::new()),
                    }),
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
                system_fingerprint: None,
            };
            vec![sse_json_event(&chunk)]
        }
        StreamEvent::Token { text, logprobs } => {
            let chunk_logprobs = logprobs.as_ref().map(|info| ChatCompletionLogprobs {
                content: vec![to_chat_logprob_content(info)],
            });
            let chunk = ChatCompletionResponse {
                id: response_meta.id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: response_meta.created,
                model: response_meta.model.clone(),
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: None,
                    delta: Some(ChatMessageOut {
                        role: "assistant".to_string(),
                        content: Some(text),
                    }),
                    finish_reason: None,
                    logprobs: chunk_logprobs,
                }],
                usage: None,
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
            log_generation_metrics(&usage, &metrics, &finish_reason, "chat stream completed");

            let mut events = Vec::new();
            events.push(sse_json_event(&ChatCompletionResponse {
                id: response_meta.id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: response_meta.created,
                model: response_meta.model.clone(),
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: None,
                    delta: Some(ChatMessageOut {
                        role: "assistant".to_string(),
                        content: None,
                    }),
                    finish_reason: Some(finish_reason),
                    logprobs: None,
                }],
                usage: if include_usage {
                    Some(usage.clone())
                } else {
                    None
                },
                system_fingerprint: None,
            }));

            if include_usage {
                events.push(sse_json_event(&ChatCompletionResponse {
                    id: response_meta.id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: response_meta.created,
                    model: response_meta.model.clone(),
                    choices: vec![],
                    usage: Some(usage),
                    system_fingerprint: None,
                }));
            }

            events.push(sse_done_event());
            events
        }
    }))
}

async fn chat_batch(
    engine: Arc<dyn InferenceEngine>,
    request: GenerateRequest,
) -> Result<Response, ApiError> {
    let result = engine.generate(request).await.map_err(ApiError::from)?;
    let finish_reason = result.finish_reason.as_openai_str().to_string();
    log_generation_metrics(
        &result.usage,
        &result.metrics,
        &finish_reason,
        "chat request completed",
    );

    let chat_logprobs = result
        .token_logprobs
        .as_ref()
        .map(|infos| to_chat_logprobs(infos));

    Ok(Json(ChatCompletionResponse {
        id: ResponseMeta::new("chatcmpl", result.model.clone()).id,
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: result.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: Some(ChatMessageOut {
                role: "assistant".to_string(),
                content: Some(result.output_text),
            }),
            delta: None,
            finish_reason: Some(finish_reason),
            logprobs: chat_logprobs,
        }],
        usage: Some(result.usage),
        system_fingerprint: None,
    })
    .into_response())
}

fn parse_chat_request(
    request: &ChatCompletionRequest,
    chat_template: Option<&crate::chat_template::ModelChatTemplate>,
) -> Result<GenerateRequest, ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "messages is empty",
            "invalid_request_error",
        ));
    }

    let chat_template = chat_template.ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "model tokenizer does not define a chat template; use /v1/completions or provide tokenizer_config.json/chat_template.jinja",
            "invalid_request_error",
        )
    })?;
    let prompt = chat_template.render(&request.messages).map_err(|e| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("failed to render chat template: {e}"),
            "invalid_request_error",
        )
    })?;

    let logprobs = if request.logprobs.unwrap_or(false) {
        Some(request.top_logprobs.unwrap_or(0))
    } else {
        None
    };

    Ok(build_generate_request(
        request.model.clone(),
        PromptInput::Text(prompt),
        request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(DEFAULT_MAX_NEW_TOKENS),
        request.temperature,
        request.top_p,
        request.stop.clone(),
        request.seed,
        logprobs,
    ))
}
