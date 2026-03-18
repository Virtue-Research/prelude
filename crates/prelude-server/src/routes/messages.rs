// TODO: Implement Anthropic-compatible Messages API (/v1/messages)
//
// SGLang supports this for Anthropic SDK compatibility.
//
// Endpoints:
//   POST /v1/messages              — create a message (Anthropic format)
//   POST /v1/messages/count_tokens — count tokens for a message
//
// Reference: https://docs.anthropic.com/en/api/messages

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

pub async fn create_message() -> Response {
    not_implemented("/v1/messages")
}

pub async fn count_tokens() -> Response {
    not_implemented("/v1/messages/count_tokens")
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
