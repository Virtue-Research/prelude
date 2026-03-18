// TODO: Implement /v1/audio/transcriptions (speech-to-text)
//
// OpenAI Whisper-compatible endpoint.
//
// Request:  POST /v1/audio/transcriptions (multipart form: file + model)
// Response: { "text": "transcribed text" }

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

pub async fn transcriptions() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "/v1/audio/transcriptions is not yet implemented",
                "type": "not_implemented"
            }
        })),
    )
        .into_response()
}
