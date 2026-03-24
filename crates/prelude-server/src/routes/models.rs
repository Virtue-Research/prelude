use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use prelude_core::{ModelCard, ModelListResponse};

use crate::error::ApiError;
use crate::Server;

pub async fn list_models(
    State(server): State<Server>,
) -> Result<Json<ModelListResponse>, ApiError> {
    let models = server.engine.list_models().await.map_err(ApiError::from)?;
    let data = models
        .into_iter()
        .map(|m| ModelCard {
            id: m.id,
            object: "model".to_string(),
            created: m.created,
            owned_by: m.owned_by,
        })
        .collect();

    Ok(Json(ModelListResponse {
        object: "list".to_string(),
        data,
    }))
}

pub async fn get_model(
    State(server): State<Server>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelCard>, ApiError> {
    let models = server.engine.list_models().await.map_err(ApiError::from)?;
    let model = models.into_iter().find(|m| m.id == model_id);

    match model {
        Some(m) => Ok(Json(ModelCard {
            id: m.id,
            object: "model".to_string(),
            created: m.created,
            owned_by: m.owned_by,
        })),
        None => Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("model '{}' not found", model_id),
            "not_found",
        )),
    }
}
