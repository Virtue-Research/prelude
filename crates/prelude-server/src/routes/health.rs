use axum::extract::State;
use axum::Json;
use prelude_core::HealthResponse;

use crate::error::ApiError;
use crate::Server;

pub async fn health(State(server): State<Server>) -> Result<Json<HealthResponse>, ApiError> {
    let model = server.engine.model_info().await.map_err(ApiError::from)?;
    Ok(Json(HealthResponse {
        status: "ready".to_string(),
        model: model.id,
        uptime_s: server.started_at.elapsed().as_secs_f64(),
    }))
}
