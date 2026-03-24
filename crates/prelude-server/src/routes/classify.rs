use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use chrono::Utc;
use prelude_core::{ClassificationRequest, ClassificationResponse, ClassificationUsage, ClassifyRequest};
use tracing::info;
use uuid::Uuid;

use crate::error::ApiError;
use crate::Server;

pub async fn classify(
    State(server): State<Server>,
    Json(request): Json<ClassificationRequest>,
) -> Result<Json<ClassificationResponse>, ApiError> {
    let inputs = request
        .get_inputs()
        .map_err(|msg| ApiError::new(StatusCode::BAD_REQUEST, msg, "invalid_request_error"))?;

    let classify_request = ClassifyRequest {
        request_id: format!("classify-{}", Uuid::new_v4().simple()),
        model: request.model.clone(),
        inputs,
    };

    info!(
        rid = %classify_request.request_id,
        model = %classify_request.model,
        "received classify request"
    );

    let result = server
        .engine
        .classify(classify_request)
        .await
        .map_err(ApiError::from)?;

    info!(
        model = %result.model,
        num_results = result.results.len(),
        prompt_tokens = result.prompt_tokens,
        "classify request completed"
    );

    Ok(Json(ClassificationResponse {
        id: format!("classify-{}", Uuid::new_v4().simple()),
        object: "list".to_string(),
        created: Utc::now().timestamp(),
        model: result.model,
        data: result.results,
        usage: ClassificationUsage {
            prompt_tokens: result.prompt_tokens,
            total_tokens: result.prompt_tokens,
            completion_tokens: 0,
        },
    }))
}
