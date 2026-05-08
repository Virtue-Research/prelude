use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use prelude_core::{
    ClassificationRequest, ClassificationResponse, ClassificationUsage, ClassifyRequest,
};
use tracing::info;

use super::generation_common::{api_id, unix_timestamp};
use crate::Server;
use crate::error::ApiError;

pub async fn classify(
    State(server): State<Server>,
    Json(request): Json<ClassificationRequest>,
) -> Result<Json<ClassificationResponse>, ApiError> {
    let inputs = request
        .get_inputs()
        .map_err(|msg| ApiError::new(StatusCode::BAD_REQUEST, msg, "invalid_request_error"))?;

    let classify_request = ClassifyRequest {
        request_id: api_id("classify"),
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
        id: api_id("classify"),
        object: "list".to_string(),
        created: unix_timestamp(),
        model: result.model,
        data: result.results,
        usage: ClassificationUsage::from_prompt_tokens(result.prompt_tokens),
    }))
}
