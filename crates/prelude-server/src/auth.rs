use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;

#[derive(Clone)]
pub struct ApiKeys(pub Arc<Vec<String>>);

pub async fn auth_middleware(
    State(keys): State<ApiKeys>,
    request: Request,
    next: Next,
) -> Response {
    // No keys configured — auth disabled
    if keys.0.is_empty() {
        return next.run(request).await;
    }

    // Only skip auth for explicitly public paths
    let path = request.uri().path();
    match path {
        "/health" | "/metrics" => return next.run(request).await,
        _ => {}
    }

    // Check Authorization: Bearer <key>
    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header[7..];
            if keys.0.iter().any(|k| k == token) {
                next.run(request).await
            } else {
                unauthorized("Invalid API key")
            }
        }
        _ => unauthorized("Missing API key. Expected header: Authorization: Bearer <key>"),
    }
}

fn unauthorized(message: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "authentication_error"
            }
        })),
    )
        .into_response()
}
