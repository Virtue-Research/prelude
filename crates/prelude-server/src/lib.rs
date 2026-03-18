mod auth;
pub mod chat_template;
mod error;
mod logprobs;
mod routes;
mod sse;
mod utils;

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::Router;
use axum::http::{HeaderValue, Method};
use axum::middleware;
use axum::routing::{get, post};
use prelude_core::InferenceEngine;
use tower_http::cors::{AllowHeaders, AllowOrigin, CorsLayer};
use tower_http::trace::TraceLayer;

use chat_template::ModelChatTemplate;
use routes::{chat_completions, classify, completions, embeddings, get_model, health, list_models};

#[derive(Clone)]
pub struct Server {
    pub engine: Arc<dyn InferenceEngine>,
    pub chat_template: Option<Arc<ModelChatTemplate>>,
    pub started_at: Instant,
}

#[derive(Clone, Debug, Default)]
pub struct RouterOptions {
    pub cors_allowed_origins: Vec<String>,
}

/// Build an axum Router with all OpenAI-compatible routes.
///
/// The returned router is backend-agnostic — it only uses the `InferenceEngine` trait.
/// Pass API keys to enable authentication on `/v1/*` routes; pass empty vec to disable.
pub fn build_router(
    engine: Arc<dyn InferenceEngine>,
    chat_template: Option<Arc<ModelChatTemplate>>,
    api_keys: Vec<String>,
) -> Router {
    build_router_with_options(engine, chat_template, api_keys, RouterOptions::default())
        .expect("default router options should be valid")
}

pub fn build_router_with_options(
    engine: Arc<dyn InferenceEngine>,
    chat_template: Option<Arc<ModelChatTemplate>>,
    api_keys: Vec<String>,
    options: RouterOptions,
) -> Result<Router> {
    let api_keys = auth::ApiKeys(Arc::new(api_keys));

    let server = Server {
        engine,
        chat_template,
        started_at: Instant::now(),
    };

    let router = Router::new()
        // Health & models
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{model}", get(get_model))
        // Core generation
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        // Embeddings & classification
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/classify", post(classify))
        .layer(middleware::from_fn_with_state(
            api_keys,
            auth::auth_middleware,
        ))
        .layer(TraceLayer::new_for_http())
        .with_state(server);

    Ok(apply_cors(router, &options)?)
}

fn apply_cors(router: Router, options: &RouterOptions) -> Result<Router> {
    if options.cors_allowed_origins.is_empty() {
        return Ok(router);
    }

    let allowed_origins = options
        .cors_allowed_origins
        .iter()
        .map(|origin| {
            origin
                .parse::<HeaderValue>()
                .with_context(|| format!("invalid CORS origin {origin:?}"))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(router.layer(
        CorsLayer::new()
            .allow_origin(AllowOrigin::list(allowed_origins))
            .allow_methods([Method::GET, Method::POST])
            .allow_headers(AllowHeaders::mirror_request()),
    ))
}
