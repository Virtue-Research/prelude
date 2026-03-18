// TODO: Implement /v1/tokenize and /v1/detokenize
//
// Requires exposing tokenize/detokenize on the InferenceEngine trait.
// CandleEngine already has pub fn tokenize(), needs trait plumbing.
//
// Request format (OpenAI/vLLM/SGLang):
//   POST /v1/tokenize   { "model": "...", "prompt": "Hello world" }
//   POST /v1/detokenize { "model": "...", "tokens": [9906, 1917] }
//
// Response format:
//   { "tokens": [9906, 1917], "count": 2 }
//   { "prompt": "Hello world" }

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

pub async fn tokenize() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "/v1/tokenize is not yet implemented",
                "type": "not_implemented"
            }
        })),
    )
        .into_response()
}

pub async fn detokenize() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "/v1/detokenize is not yet implemented",
                "type": "not_implemented"
            }
        })),
    )
        .into_response()
}
