// TODO: Implement /v1/score (text similarity scoring)
//
// Used by vLLM and SGLang for cross-encoder / reward models.
//
// Request:  POST /v1/score { "model": "...", "text_1": "...", "text_2": "..." }
// Response: { "id": "...", "data": [{ "index": 0, "score": 0.95 }], "usage": {...} }

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

pub async fn score() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "/v1/score is not yet implemented",
                "type": "not_implemented"
            }
        })),
    )
        .into_response()
}
