// TODO: Implement OpenAI Moderations API (/v1/moderations)
//
// Reference: https://platform.openai.com/docs/api-reference/moderations
//
// Request:  POST /v1/moderations { "model": "...", "input": "text" }
// Response: { "id": "...", "model": "...", "results": [{ "flagged": bool, "categories": {...}, "category_scores": {...} }] }

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

pub async fn moderations() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "/v1/moderations is not yet implemented",
                "type": "not_implemented"
            }
        })),
    )
        .into_response()
}
