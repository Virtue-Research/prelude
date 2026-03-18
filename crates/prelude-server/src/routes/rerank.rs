// TODO: Implement /v1/rerank (document reranking)
//
// Used by vLLM for reranker models (e.g., Cohere rerank, BGE reranker).
//
// Request:  POST /v1/rerank { "model": "...", "query": "...", "documents": ["...", "..."] }
// Response: { "results": [{ "index": 0, "relevance_score": 0.95 }] }

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

pub async fn rerank() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "/v1/rerank is not yet implemented",
                "type": "not_implemented"
            }
        })),
    )
        .into_response()
}
