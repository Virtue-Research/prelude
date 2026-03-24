use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use prelude_core::EngineError;
use serde::Serialize;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub kind: String,
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            kind: kind.into(),
        }
    }
}

impl From<EngineError> for ApiError {
    fn from(err: EngineError) -> Self {
        match err {
            EngineError::InvalidRequest(msg) => {
                Self::new(StatusCode::BAD_REQUEST, msg, "invalid_request_error")
            }
            EngineError::Unavailable(msg) => {
                Self::new(StatusCode::SERVICE_UNAVAILABLE, msg, "service_unavailable")
            }
            EngineError::Internal(msg) => {
                Self::new(StatusCode::INTERNAL_SERVER_ERROR, msg, "internal_error")
            }
        }
    }
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let payload = Json(ErrorEnvelope {
            error: ErrorBody {
                message: self.message,
                kind: self.kind,
            },
        });
        (self.status, payload).into_response()
    }
}
