use axum::extract::State;
use axum::Json;
use base64::Engine as _;
use prelude_core::{
    EmbedRequest, EmbeddingObject, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    EmbeddingValue,
};
use tracing::info;
use uuid::Uuid;

use crate::error::ApiError;
use crate::Server;

/// Encode a Vec<f32> as a base64 string of raw little-endian f32 bytes.
/// OpenAI-compatible: each f32 → 4 bytes LE, then base64-encode the whole buffer.
fn encode_embedding_base64(embedding: &[f32]) -> String {
    // Safety: f32 slice → u8 slice is always valid for LE platforms
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(embedding.as_ptr() as *const u8, embedding.len() * 4) };
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub async fn embeddings(
    State(server): State<Server>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    request
        .validate_public_request()
        .map_err(|message| ApiError::new(axum::http::StatusCode::BAD_REQUEST, message, "invalid_request_error"))?;

    let inputs = request.get_inputs();
    let use_base64 = request
        .encoding_format
        .as_deref()
        .is_some_and(|f| f.eq_ignore_ascii_case("base64"));

    let embed_request = EmbedRequest {
        request_id: format!("embed-{}", Uuid::new_v4().simple()),
        model: request.model.clone(),
        inputs,
    };

    info!(
        rid = %embed_request.request_id,
        model = %embed_request.model,
        encoding_format = if use_base64 { "base64" } else { "float" },
        "received embed request"
    );

    let result = server
        .engine
        .embed(embed_request)
        .await
        .map_err(ApiError::from)?;

    info!(
        model = %result.model,
        num_embeddings = result.data.len(),
        dimensions = result.dimensions,
        prompt_tokens = result.prompt_tokens,
        "embed request completed"
    );

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data: result
            .data
            .into_iter()
            .map(|d| EmbeddingObject {
                object: "embedding".to_string(),
                index: d.index,
                embedding: if use_base64 {
                    EmbeddingValue::Base64(encode_embedding_base64(&d.embedding))
                } else {
                    EmbeddingValue::Float(d.embedding)
                },
            })
            .collect(),
        model: result.model,
        usage: EmbeddingUsage {
            prompt_tokens: result.prompt_tokens,
            total_tokens: result.prompt_tokens,
        },
    }))
}
