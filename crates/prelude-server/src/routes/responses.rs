// TODO: Implement OpenAI Responses API (/v1/responses)
//
// The Responses API is OpenAI's newer interface that supports:
// - Multi-turn conversation state management (conversation IDs)
// - Built-in tools (web search, code interpreter, file search)
// - Structured output / function calling
// - Streaming via SSE with event types (response.created, response.output_item.delta, response.done)
//
// Endpoints:
//   POST /v1/responses              — create a response
//   GET  /v1/responses/{id}         — retrieve a response
//   POST /v1/responses/{id}/cancel  — cancel an in-progress response
//
// Reference: https://platform.openai.com/docs/api-reference/responses
//
// For now, return 501 Not Implemented.

use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

pub async fn create_response() -> Response {
    not_implemented("/v1/responses")
}

pub async fn get_response(Path(_id): Path<String>) -> Response {
    not_implemented("/v1/responses/{id}")
}

pub async fn cancel_response(Path(_id): Path<String>) -> Response {
    not_implemented("/v1/responses/{id}/cancel")
}

fn not_implemented(endpoint: &str) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": format!("{} is not yet implemented", endpoint),
                "type": "not_implemented"
            }
        })),
    )
        .into_response()
}
