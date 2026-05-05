use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use prelude_core::{
    ChatCompletionChoice, ChatCompletionLogprobs, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessageOut, GenerateRequest, InferenceEngine, PromptInput, StreamEvent,
};
use tracing::info;

use super::generation_common::{
    DEFAULT_MAX_NEW_TOKENS, ResponseMeta, build_generate_request, sse_done_event, sse_json_event,
};
use crate::Server;
use crate::error::ApiError;
use crate::logprobs::{to_chat_logprob_content, to_chat_logprobs};
use crate::sse::stream_sse;
use crate::utils::log_generation_metrics;

pub async fn chat_completions(
    State(server): State<Server>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    request.validate_public_request().map_err(|message| {
        ApiError::new(StatusCode::BAD_REQUEST, message, "invalid_request_error")
    })?;

    let is_streaming = request.stream.unwrap_or(false);
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(false);
    let (engine_request, thinking_in_prompt) =
        parse_chat_request(&request, server.chat_template.as_deref())?;

    info!(
        rid = %engine_request.request_id,
        model = %engine_request.model,
        max_new_tokens = engine_request.max_new_tokens,
        stream = is_streaming,
        thinking_in_prompt,
        "received chat request"
    );

    if is_streaming {
        chat_stream(
            server.engine,
            engine_request,
            include_usage,
            thinking_in_prompt,
        )
    } else {
        chat_batch(server.engine, engine_request, thinking_in_prompt).await
    }
}

/// Peek at the end of a rendered chat prompt to see if the template
/// injected a `<think>\n` opener. Matches vLLM's `Qwen3ReasoningParser`
/// behavior: when the prompt ends in an open `<think>` block, the
/// generation stream is assumed to start inside a reasoning span.
fn prompt_ends_in_open_think(prompt: &str) -> bool {
    // Accept both with and without trailing newline to be resilient to
    // template variations (stream-style templates sometimes omit the `\n`).
    prompt.ends_with("<think>\n") || prompt.ends_with("<think>")
}

/// Split a model output into (reasoning, content) matching vLLM's
/// `Qwen3ReasoningParser`:
///
/// - If `thinking_in_prompt`, the generation started inside an open
///   `<think>` block. Look for `</think>`: everything before is reasoning,
///   everything after is content. If the block never closed (short
///   generation), the whole output is reasoning and content is empty.
/// - If the output itself starts with `<think>` (model self-emitted the
///   opener because the prompt didn't), handle the same way on the
///   trimmed suffix.
/// - Otherwise, no reasoning split — return `(None, output)`.
fn split_reasoning(output: &str, thinking_in_prompt: bool) -> (Option<String>, String) {
    const END: &str = "</think>";

    // Matches vLLM's `str.partition(end_token)` — no leading/trailing
    // whitespace trimming, first occurrence only. The leading `\n` that
    // the chat template puts after `</think>` stays on the content side,
    // matching vLLM's byte-faithful output.
    if thinking_in_prompt {
        if let Some(idx) = output.find(END) {
            let reasoning = output[..idx].to_string();
            let content = output[idx + END.len()..].to_string();
            return (Some(reasoning), content);
        }
        // Unfinished reasoning block — entire output is reasoning.
        return (Some(output.to_string()), String::new());
    }

    // Prompt didn't inject <think>. If the model itself emitted <think>
    // (e.g. a reasoning model was prompted without the thinking template),
    // still try to split on </think>. `str.partition` in vLLM is
    // equivalent to `find + slice` here.
    let trimmed = output
        .strip_prefix("<think>\n")
        .or_else(|| output.strip_prefix("<think>"));
    if let Some(rest) = trimmed {
        if let Some(idx) = rest.find(END) {
            let reasoning = rest[..idx].to_string();
            let content = rest[idx + END.len()..].to_string();
            return (Some(reasoning), content);
        }
        return (Some(rest.to_string()), String::new());
    }

    (None, output.to_string())
}

fn chat_stream(
    engine: Arc<dyn InferenceEngine>,
    request: GenerateRequest,
    include_usage: bool,
    _thinking_in_prompt: bool,
) -> Result<Response, ApiError> {
    // TODO(reasoning-stream): vLLM's `Qwen3ReasoningParser` tracks the
    // `</think>` boundary incrementally in the stream and fills
    // `DeltaMessage.reasoning` vs `.content` per chunk. We currently send
    // every chunk as `content` regardless of whether the model is still
    // inside the `<think>` block. That's fine for clients that accumulate
    // and post-process, but doesn't match vLLM's streaming contract.
    // Revisit when a streaming-reasoning client actually asks for it.
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
                        reasoning: None,
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
                        reasoning: None,
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
                        reasoning: None,
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
        StreamEvent::Error { message } => {
            tracing::error!(error = %message, "chat stream generation error");
            vec![sse_done_event()]
        }
    }))
}

async fn chat_batch(
    engine: Arc<dyn InferenceEngine>,
    request: GenerateRequest,
    thinking_in_prompt: bool,
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

    let (reasoning, content) = split_reasoning(&result.output_text, thinking_in_prompt);

    Ok(Json(ChatCompletionResponse {
        id: ResponseMeta::new("chatcmpl", result.model.clone()).id,
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: result.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: Some(ChatMessageOut {
                role: "assistant".to_string(),
                content: Some(content),
                reasoning,
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
) -> Result<(GenerateRequest, bool), ApiError> {
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

    // Reasoning-mode templates (Qwen3, Qwen3.5, ...) append `<think>\n` to
    // the assistant turn so the model starts generating inside a reasoning
    // span. We detect that here so the response builder can split the
    // output on `</think>` and populate `reasoning` vs `content`.
    //
    // Note on trigger choice: vLLM keys this off an explicit
    // `chat_template_kwargs.enable_thinking` flag, while we inspect the
    // rendered prompt's suffix. For Qwen3.5's template the two are
    // equivalent: `enable_thinking=False` causes the template to append
    // `<think>\n\n</think>\n\n` (prompt ends with `</think>\n\n`) and
    // `enable_thinking=True` appends `<think>\n` (prompt ends with
    // `<think>\n`). Our `prompt_ends_in_open_think` matches the latter
    // but not the former, so the outcome agrees with vLLM for this
    // template. If we ever add a template that injects `<think>` via a
    // different path, swap this for an explicit kwarg.
    let thinking_in_prompt = prompt_ends_in_open_think(&prompt);

    let logprobs = if request.logprobs.unwrap_or(false) {
        Some(request.top_logprobs.unwrap_or(0))
    } else {
        None
    };

    Ok((
        build_generate_request(
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
            None, // prompt_logprobs not supported for chat completions
        ),
        thinking_in_prompt,
    ))
}
