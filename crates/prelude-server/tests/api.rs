mod common;

use std::sync::Arc;

use axum::body::Body;
use axum::http::header::{
    ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
    ACCESS_CONTROL_REQUEST_HEADERS, ACCESS_CONTROL_REQUEST_METHOD, ORIGIN,
};
use axum::http::{Request, StatusCode};
use serde_json::json;
use tower::ServiceExt;

use common::{
    ContractTestEngine, bearer_request, empty_request, make_app, mock_chat_template, send_json,
    send_text, sse_data_lines,
};
use prelude_server::{RouterOptions, build_router_with_options};

// ── Health & Models ──

#[tokio::test]
async fn health_returns_ready() {
    let app = make_app(true, vec![]);
    let (status, body) = send_json(&app, empty_request("GET", "/health")).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ready");
    assert_eq!(body["model"], "test-model");
    assert!(body["uptime_s"].as_f64().unwrap() >= 0.0);
}

#[tokio::test]
async fn models_list_and_lookup_work() {
    let app = make_app(true, vec![]);

    let (list_status, list_body) = send_json(&app, empty_request("GET", "/v1/models")).await;
    assert_eq!(list_status, StatusCode::OK);
    assert_eq!(list_body["object"], "list");
    assert_eq!(list_body["data"][0]["id"], "test-model");

    let (get_status, get_body) =
        send_json(&app, empty_request("GET", "/v1/models/test-model")).await;
    assert_eq!(get_status, StatusCode::OK);
    assert_eq!(get_body["id"], "test-model");

    let (missing_status, missing_body) =
        send_json(&app, empty_request("GET", "/v1/models/missing")).await;
    assert_eq!(missing_status, StatusCode::NOT_FOUND);
    assert_eq!(missing_body["error"]["type"], "not_found");
}

#[tokio::test]
async fn cors_preflight_allows_allowlisted_origins_for_api_requests() {
    let app = build_router_with_options(
        Arc::new(ContractTestEngine::new("test-model")),
        Some(mock_chat_template()),
        vec![],
        RouterOptions {
            cors_allowed_origins: vec!["https://app.example.com".to_string()],
        },
    )
    .unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/completions")
                .header(ORIGIN, "https://app.example.com")
                .header(ACCESS_CONTROL_REQUEST_METHOD, "POST")
                .header(ACCESS_CONTROL_REQUEST_HEADERS, "authorization,content-type")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(ACCESS_CONTROL_ALLOW_ORIGIN).unwrap(),
        "https://app.example.com"
    );

    let allow_methods = response
        .headers()
        .get(ACCESS_CONTROL_ALLOW_METHODS)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(allow_methods.contains("GET"));
    assert!(allow_methods.contains("POST"));

    assert_eq!(
        response
            .headers()
            .get(ACCESS_CONTROL_ALLOW_HEADERS)
            .unwrap(),
        "authorization,content-type"
    );
}

#[tokio::test]
async fn completions_batch_preserves_order_and_usage() {
    let app = make_app(true, vec![]);
    let request = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": ["alpha prompt", "beta prompt"],
            "max_tokens": 2,
            "seed": 7
        }),
    );

    let (status, body) = send_json(&app, request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["index"], 0);
    assert_eq!(body["choices"][1]["index"], 1);
    assert_ne!(body["choices"][0]["text"], body["choices"][1]["text"]);
    assert_eq!(body["usage"]["prompt_tokens"], 4);
    assert_eq!(body["usage"]["completion_tokens"], 4);
    assert_eq!(body["usage"]["total_tokens"], 8);
}

#[tokio::test]
async fn completions_support_stop_strings_and_seed_determinism() {
    let app = make_app(true, vec![]);
    let first = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "deterministic prompt",
            "max_tokens": 3,
            "seed": 11
        }),
    );
    let second = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "deterministic prompt",
            "max_tokens": 3,
            "seed": 11
        }),
    );
    let third = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "deterministic prompt",
            "max_tokens": 3,
            "seed": 12
        }),
    );
    let (_, first_body) = send_json(&app, first).await;
    let (_, second_body) = send_json(&app, second).await;
    let (_, third_body) = send_json(&app, third).await;

    let baseline_stop = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "stop prompt",
            "max_tokens": 4,
            "seed": 3
        }),
    );
    let (_, baseline_stop_body) = send_json(&app, baseline_stop).await;
    let words: Vec<&str> = baseline_stop_body["choices"][0]["text"]
        .as_str()
        .unwrap()
        .split_whitespace()
        .collect();
    let stop_target = format!(" {}", words[1]);
    let stop_request = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "stop prompt",
            "max_tokens": 4,
            "seed": 3,
            "stop": [stop_target]
        }),
    );
    let (_, stop_body) = send_json(&app, stop_request).await;

    let cross_boundary_stop = format!(
        "{} {}",
        words[0].chars().last().unwrap(),
        words[1].chars().next().unwrap()
    );
    let cross_boundary_request = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "stop prompt",
            "max_tokens": 4,
            "seed": 3,
            "stop": [cross_boundary_stop]
        }),
    );
    let (_, cross_boundary_body) = send_json(&app, cross_boundary_request).await;

    assert_eq!(
        first_body["choices"][0]["text"],
        second_body["choices"][0]["text"]
    );
    assert_ne!(
        first_body["choices"][0]["text"],
        third_body["choices"][0]["text"]
    );
    assert_eq!(stop_body["choices"][0]["finish_reason"], "stop");
    assert!(
        stop_body["choices"][0]["text"]
            .as_str()
            .unwrap()
            .split_whitespace()
            .count()
            < words.len()
    );
    assert_eq!(cross_boundary_body["choices"][0]["finish_reason"], "stop");
    assert!(
        cross_boundary_body["choices"][0]["text"]
            .as_str()
            .unwrap()
            .len()
            < words[0].len()
    );
}

#[tokio::test]
async fn completions_logprobs_shape_is_stable() {
    let app = make_app(true, vec![]);
    let request = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "logprob prompt",
            "max_tokens": 2,
            "logprobs": 3
        }),
    );

    let (status, body) = send_json(&app, request).await;

    assert_eq!(status, StatusCode::OK);
    let logprobs = &body["choices"][0]["logprobs"];
    assert_eq!(logprobs["tokens"].as_array().unwrap().len(), 2);
    assert_eq!(logprobs["token_logprobs"].as_array().unwrap().len(), 2);
    assert_eq!(
        logprobs["top_logprobs"].as_array().unwrap()[0]
            .as_object()
            .unwrap()
            .len(),
        3
    );
}

#[tokio::test]
async fn completion_streaming_emits_sse_chunks_and_done() {
    let app = make_app(true, vec![]);
    let request = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "stream prompt",
            "max_tokens": 2,
            "stream": true,
            "stream_options": {"include_usage": true},
            "logprobs": 2
        }),
    );

    let (status, body, content_type) = send_text(&app, request).await;
    let events = sse_data_lines(&body);

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.unwrap().contains("text/event-stream"));
    assert_eq!(events.last().unwrap(), "[DONE]");
    assert!(events.len() >= 4);

    let token_chunk: serde_json::Value = serde_json::from_str(&events[0]).unwrap();
    assert_eq!(token_chunk["object"], "text_completion");
    assert!(token_chunk["choices"][0]["logprobs"].is_object());

    let finish_chunk: serde_json::Value = serde_json::from_str(&events[2]).unwrap();
    assert_eq!(finish_chunk["choices"][0]["finish_reason"], "length");

    let usage_chunk: serde_json::Value = serde_json::from_str(&events[3]).unwrap();
    assert_eq!(usage_chunk["choices"], json!([]));
    assert!(usage_chunk["usage"]["total_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn completion_streaming_rejects_prompt_batches() {
    let app = make_app(true, vec![]);
    let request = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": ["one", "two"],
            "max_tokens": 2,
            "stream": true
        }),
    );

    let (status, body) = send_json(&app, request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn completion_rejects_empty_prompt() {
    let app = make_app(true, vec![]);
    let request = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "   ",
            "max_tokens": 2
        }),
    );

    let (status, body) = send_json(&app, request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn completion_rejects_unsupported_fields() {
    let app = make_app(true, vec![]);
    let request = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "hello",
            "n": 2,
            "user": "user-123"
        }),
    );

    let (status, body) = send_json(&app, request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"].as_str().unwrap().contains("n=2"));
}

#[tokio::test]
async fn chat_requires_template() {
    let app = make_app(false, vec![]);
    let request = bearer_request(
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2
        }),
    );

    let (status, body) = send_json(&app, request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn chat_honors_max_completion_tokens_over_max_tokens() {
    let app = make_app(true, vec![]);
    let request = bearer_request(
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "count to three"}],
            "max_tokens": 1,
            "max_completion_tokens": 3,
            "seed": 21
        }),
    );

    let (status, body) = send_json(&app, request).await;
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(content.split_whitespace().count(), 3);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
}

#[tokio::test]
async fn chat_streaming_matches_non_streaming_output() {
    let app = make_app(true, vec![]);
    let batch_request = bearer_request(
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "stream this"}],
            "max_tokens": 3,
            "seed": 99
        }),
    );
    let stream_request = bearer_request(
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "stream this"}],
            "max_tokens": 3,
            "seed": 99,
            "stream": true,
            "stream_options": {"include_usage": true}
        }),
    );

    let (_, batch_body) = send_json(&app, batch_request).await;
    let (_, stream_body, _) = send_text(&app, stream_request).await;
    let events = sse_data_lines(&stream_body);

    let stream_text = events[..events.len() - 1]
        .iter()
        .filter_map(|event| serde_json::from_str::<serde_json::Value>(event).ok())
        .filter_map(|event| {
            event["choices"][0]["delta"]["content"]
                .as_str()
                .map(|content| content.to_string())
        })
        .collect::<String>();

    assert_eq!(
        stream_text,
        batch_body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
    );
    assert_eq!(events.last().unwrap(), "[DONE]");
}

#[tokio::test]
async fn chat_logprobs_shape_is_stable() {
    let app = make_app(true, vec![]);
    let request = bearer_request(
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "show logprobs"}],
            "max_tokens": 2,
            "logprobs": true,
            "top_logprobs": 2
        }),
    );

    let (status, body) = send_json(&app, request).await;
    let content = &body["choices"][0]["logprobs"]["content"];

    assert_eq!(status, StatusCode::OK);
    assert_eq!(content.as_array().unwrap().len(), 2);
    assert_eq!(content[0]["top_logprobs"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn chat_rejects_unsupported_fields_and_non_text_messages() {
    let app = make_app(true, vec![]);

    let unsupported_field = bearer_request(
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"}
        }),
    );
    let (unsupported_status, unsupported_body) = send_json(&app, unsupported_field).await;
    assert_eq!(unsupported_status, StatusCode::BAD_REQUEST);
    assert_eq!(unsupported_body["error"]["type"], "invalid_request_error");
    assert!(
        unsupported_body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("response_format")
    );

    let multimodal = bearer_request(
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hi"}]
            }]
        }),
    );
    let (multimodal_status, multimodal_body) = send_json(&app, multimodal).await;
    assert_eq!(multimodal_status, StatusCode::BAD_REQUEST);
    assert_eq!(multimodal_body["error"]["type"], "invalid_request_error");
    assert!(
        multimodal_body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("plain string")
    );
}

#[tokio::test]
async fn classify_supports_text_batches_token_ids_and_messages() {
    let app = make_app(true, vec![]);

    let (batch_status, batch_body) = send_json(
        &app,
        bearer_request(
            "POST",
            "/v1/classify",
            json!({
                "model": "test-model",
                "input": ["one input", "two inputs"]
            }),
        ),
    )
    .await;
    assert_eq!(batch_status, StatusCode::OK);
    assert_eq!(batch_body["data"].as_array().unwrap().len(), 2);
    assert_eq!(batch_body["data"][0]["index"], 0);
    assert_eq!(batch_body["data"][1]["num_classes"], 3);

    let (token_status, token_body) = send_json(
        &app,
        bearer_request(
            "POST",
            "/v1/classify",
            json!({
                "model": "test-model",
                "input": [[1, 2, 3], [4, 5]]
            }),
        ),
    )
    .await;
    assert_eq!(token_status, StatusCode::OK);
    assert_eq!(token_body["usage"]["prompt_tokens"], 5);

    let (message_status, message_body) = send_json(
        &app,
        bearer_request(
            "POST",
            "/v1/classify",
            json!({
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "Be strict"},
                    {"role": "user", "content": "Rate this"}
                ]
            }),
        ),
    )
    .await;
    assert_eq!(message_status, StatusCode::OK);
    assert_eq!(message_body["data"].as_array().unwrap().len(), 1);
    assert!(
        message_body["data"][0]["label"]
            .as_str()
            .unwrap()
            .starts_with("LABEL_")
    );
}

#[tokio::test]
async fn classify_rejects_input_and_messages_together() {
    let app = make_app(true, vec![]);
    let request = bearer_request(
        "POST",
        "/v1/classify",
        json!({
            "model": "test-model",
            "input": "hello",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    );

    let (status, body) = send_json(&app, request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn embeddings_support_text_and_token_inputs() {
    let app = make_app(true, vec![]);

    let (single_status, single_body) = send_json(
        &app,
        bearer_request(
            "POST",
            "/v1/embeddings",
            json!({
                "model": "test-model",
                "input": "embed this"
            }),
        ),
    )
    .await;
    assert_eq!(single_status, StatusCode::OK);
    assert_eq!(single_body["data"].as_array().unwrap().len(), 1);
    assert_eq!(
        single_body["data"][0]["embedding"]
            .as_array()
            .unwrap()
            .len(),
        4
    );

    let (batch_status, batch_body) = send_json(
        &app,
        bearer_request(
            "POST",
            "/v1/embeddings",
            json!({
                "model": "test-model",
                "input": [[1, 2], [3, 4, 5]]
            }),
        ),
    )
    .await;
    assert_eq!(batch_status, StatusCode::OK);
    assert_eq!(batch_body["data"][0]["index"], 0);
    assert_eq!(batch_body["data"][1]["index"], 1);
    assert_eq!(batch_body["usage"]["prompt_tokens"], 5);
}

#[tokio::test]
async fn embeddings_reject_unsupported_options() {
    let app = make_app(true, vec![]);
    let request = bearer_request(
        "POST",
        "/v1/embeddings",
        json!({
            "model": "test-model",
            "input": "embed this",
            "encoding_format": "hex",
            "dimensions": 2
        }),
    );

    let (status, body) = send_json(&app, request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("dimensions")
            || body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("encoding_format")
    );
}

#[tokio::test]
async fn auth_protects_v1_routes_and_skips_non_v1() {
    let app = make_app(true, vec!["sk-secret".to_string()]);

    let (health_status, _) = send_json(&app, empty_request("GET", "/health")).await;
    assert_eq!(health_status, StatusCode::OK);

    let (missing_status, missing_body) = send_json(&app, empty_request("GET", "/v1/models")).await;
    assert_eq!(missing_status, StatusCode::UNAUTHORIZED);
    assert_eq!(missing_body["error"]["type"], "authentication_error");

    let classify_req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/classify")
        .header("authorization", "Bearer sk-secret")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            json!({"model": "test-model", "input": "allowed"}).to_string(),
        ))
        .unwrap();
    let (classify_status, _) = send_json(&app, classify_req).await;
    assert_eq!(classify_status, StatusCode::OK);

    // /v1/classify without auth should be rejected
    let (noauth_classify, _) = send_json(&app, empty_request("POST", "/v1/classify")).await;
    assert_eq!(noauth_classify, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn engine_errors_map_to_expected_status_codes() {
    let app = make_app(true, vec![]);
    let unavailable_req = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "unavailable-model",
            "prompt": "hello",
            "max_tokens": 1
        }),
    );
    let internal_req = bearer_request(
        "POST",
        "/v1/completions",
        json!({
            "model": "internal-model",
            "prompt": "hello",
            "max_tokens": 1
        }),
    );

    let (unavailable_status, unavailable_body) = send_json(&app, unavailable_req).await;
    let (internal_status, internal_body) = send_json(&app, internal_req).await;

    assert_eq!(unavailable_status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(unavailable_body["error"]["type"], "service_unavailable");
    assert_eq!(internal_status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(internal_body["error"]["type"], "internal_error");
}

#[tokio::test]
async fn unsupported_placeholder_routes_are_not_mounted() {
    let app = make_app(true, vec![]);
    let stubs = [
        ("POST", "/v1/responses"),
        ("GET", "/v1/responses/resp_123"),
        ("POST", "/v1/responses/resp_123/cancel"),
        ("POST", "/v1/tokenize"),
        ("POST", "/v1/detokenize"),
        ("POST", "/v1/moderations"),
        ("POST", "/v1/score"),
        ("POST", "/v1/rerank"),
        ("POST", "/v1/audio/transcriptions"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/messages/count_tokens"),
    ];

    for (method, path) in stubs {
        let (status, body) = send_json(&app, empty_request(method, path)).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{method} {path}");
        assert!(body.is_null(), "{method} {path}");
    }
}
