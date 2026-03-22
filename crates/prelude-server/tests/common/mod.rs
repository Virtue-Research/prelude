#![allow(dead_code)]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::net::TcpListener;
use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode, header::CONTENT_TYPE};
use http_body_util::BodyExt;
use prelude_core::{
    ClassificationInputs, ClassificationResult, ClassifyRequest, ClassifyResult, DecodeMetrics,
    EmbedRequest, EmbedResult, EmbeddingData, EngineError, FinishReason, GenerateRequest,
    GenerateResult, InferenceEngine, ModelInfo, PromptInput, StreamEvent, TokenLogprobInfo, Usage,
};
use prelude_server::build_router;
use prelude_server::chat_template::ModelChatTemplate;
use tower::ServiceExt;

const WORD_BANK: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
];

#[derive(Clone)]
pub struct ContractTestEngine {
    model_info: ModelInfo,
}

impl ContractTestEngine {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_info: ModelInfo {
                id: model_id.into(),
                created: 1_700_000_000,
                owned_by: "prelude-tests".to_string(),
            },
        }
    }

    fn maybe_fail(&self, model: &str) -> Result<(), EngineError> {
        if model.contains("unavailable") {
            return Err(EngineError::Unavailable(
                "test backend unavailable".to_string(),
            ));
        }
        if model.contains("internal") {
            return Err(EngineError::Internal("test backend crashed".to_string()));
        }
        if model.contains("invalid") {
            return Err(EngineError::InvalidRequest(
                "test backend rejected request".to_string(),
            ));
        }
        Ok(())
    }

    fn hash_value<T: Hash>(&self, value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn prompt_key(&self, input: &PromptInput) -> String {
        match input {
            PromptInput::Text(text) => text.clone(),
            PromptInput::TokenIds(ids) => format!("{ids:?}"),
        }
    }

    fn prompt_tokens(&self, input: &PromptInput) -> u32 {
        match input {
            PromptInput::Text(text) => text
                .split_whitespace()
                .count()
                .try_into()
                .unwrap_or(u32::MAX),
            PromptInput::TokenIds(ids) => ids.len().try_into().unwrap_or(u32::MAX),
        }
    }

    fn render_generation(
        &self,
        request: &GenerateRequest,
    ) -> Result<RenderedGeneration, EngineError> {
        self.maybe_fail(&request.model)?;

        let prompt_key = self.prompt_key(&request.input);
        if prompt_key.trim().is_empty() {
            return Err(EngineError::InvalidRequest("prompt is empty".to_string()));
        }

        let max_tokens = request.max_new_tokens.clamp(1, 8) as usize;
        let mut seed =
            self.hash_value(&(prompt_key.as_str(), request.model.as_str(), request.seed));
        let top_k = request.logprobs.unwrap_or(0).max(1) as usize;

        let mut text = String::new();
        let mut pieces = Vec::with_capacity(max_tokens);
        let mut token_ids = Vec::with_capacity(max_tokens);
        let mut token_logprobs = Vec::with_capacity(max_tokens);
        let mut finish_reason = FinishReason::Length;

        for step in 0..max_tokens {
            let bank_idx = ((seed as usize) + step * 5) % WORD_BANK.len();
            let word = WORD_BANK[bank_idx];
            let piece = if step == 0 {
                word.to_string()
            } else {
                format!(" {word}")
            };
            let candidate = format!("{text}{piece}");
            if let Some(stop_pos) = find_stop_pos(&candidate, &request.stop.strings) {
                let delta_start = text.len().min(stop_pos);
                let delta = &candidate[delta_start..stop_pos];
                if !delta.is_empty() {
                    pieces.push(delta.to_string());
                    token_ids.push(bank_idx as u32);
                    token_logprobs.push(self.build_logprob(delta, bank_idx, top_k));
                }
                text = candidate[..stop_pos].trim_end().to_string();
                finish_reason = FinishReason::Stop;
                break;
            }

            text = candidate;
            pieces.push(piece.clone());
            token_ids.push(bank_idx as u32);
            token_logprobs.push(self.build_logprob(&piece, bank_idx, top_k));
            seed = seed.rotate_left(7) ^ ((bank_idx as u64 + 1) * 0x9E37_79B1);
        }

        let prompt_tokens = self.prompt_tokens(&request.input);
        let completion_tokens = text
            .split_whitespace()
            .count()
            .try_into()
            .unwrap_or(u32::MAX);
        let usage = Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        };

        Ok(RenderedGeneration {
            text,
            pieces,
            token_ids,
            token_logprobs: request.logprobs.map(|_| token_logprobs),
            finish_reason,
            usage,
        })
    }

    fn build_logprob(&self, token: &str, token_id: usize, top_k: usize) -> TokenLogprobInfo {
        let mut top_logprobs = Vec::with_capacity(top_k);
        for offset in 0..top_k {
            let idx = (token_id + offset) % WORD_BANK.len();
            top_logprobs.push((idx as u32, WORD_BANK[idx].to_string(), -0.1 - offset as f32));
        }
        TokenLogprobInfo {
            token: token.to_string(),
            token_id: token_id as u32,
            logprob: -0.1,
            top_logprobs,
        }
    }

    fn classify_inputs(&self, inputs: &ClassificationInputs) -> Vec<String> {
        match inputs {
            ClassificationInputs::Texts(texts) => texts.clone(),
            ClassificationInputs::TokenIds(token_ids) => token_ids
                .iter()
                .map(|ids| format!("tokens:{}", ids.len()))
                .collect(),
        }
    }

    fn embedding_inputs(&self, inputs: &ClassificationInputs) -> Vec<String> {
        self.classify_inputs(inputs)
    }
}

struct RenderedGeneration {
    text: String,
    pieces: Vec<String>,
    token_ids: Vec<u32>,
    token_logprobs: Option<Vec<TokenLogprobInfo>>,
    finish_reason: FinishReason,
    usage: Usage,
}

#[async_trait]
impl InferenceEngine for ContractTestEngine {
    async fn model_info(&self) -> Result<ModelInfo, EngineError> {
        Ok(self.model_info.clone())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, EngineError> {
        Ok(vec![self.model_info.clone()])
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResult, EngineError> {
        let rendered = self.render_generation(&request)?;
        Ok(GenerateResult {
            model: request.model,
            output_token_ids: rendered.token_ids,
            output_text: rendered.text,
            finish_reason: rendered.finish_reason,
            usage: rendered.usage,
            metrics: fixed_metrics(),
            token_logprobs: rendered.token_logprobs,
            prompt_token_logprobs: None,
        })
    }

    async fn generate_stream(
        &self,
        request: GenerateRequest,
        tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<(), EngineError> {
        let rendered = self.render_generation(&request)?;
        let logprobs = rendered.token_logprobs.unwrap_or_default();

        let _ = tx.send(StreamEvent::Started);
        for (idx, piece) in rendered.pieces.into_iter().enumerate() {
            let _ = tx.send(StreamEvent::Token {
                text: piece,
                logprobs: logprobs.get(idx).cloned(),
            });
        }
        let _ = tx.send(StreamEvent::Finished {
            finish_reason: rendered.finish_reason,
            usage: rendered.usage,
            metrics: fixed_metrics(),
        });
        Ok(())
    }

    async fn cancel(&self, _request_id: &str) -> Result<bool, EngineError> {
        Ok(false)
    }

    async fn classify(&self, request: ClassifyRequest) -> Result<ClassifyResult, EngineError> {
        self.maybe_fail(&request.model)?;
        let inputs = self.classify_inputs(&request.inputs);
        if inputs.is_empty() {
            return Err(EngineError::InvalidRequest("empty input".to_string()));
        }

        let results = inputs
            .iter()
            .enumerate()
            .map(|(idx, input)| {
                let base = self.hash_value(&(request.model.as_str(), input.as_str()));
                let probs = vec![
                    (base & 0xff) as f32 / 255.0,
                    ((base >> 8) & 0xff) as f32 / 255.0,
                    ((base >> 16) & 0xff) as f32 / 255.0,
                ];
                let label_idx = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap_or(0);
                ClassificationResult {
                    index: idx as u32,
                    label: Some(format!("LABEL_{label_idx}")),
                    probs,
                    num_classes: 3,
                }
            })
            .collect();

        let prompt_tokens = match &request.inputs {
            ClassificationInputs::Texts(texts) => texts
                .iter()
                .map(|text| text.split_whitespace().count() as u32)
                .sum(),
            ClassificationInputs::TokenIds(token_ids) => {
                token_ids.iter().map(|ids| ids.len() as u32).sum()
            }
        };

        Ok(ClassifyResult {
            model: request.model,
            results,
            prompt_tokens,
        })
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResult, EngineError> {
        self.maybe_fail(&request.model)?;
        let inputs = self.embedding_inputs(&request.inputs);
        if inputs.is_empty() {
            return Err(EngineError::InvalidRequest("empty input".to_string()));
        }

        let data = inputs
            .iter()
            .enumerate()
            .map(|(idx, input)| {
                let base = self.hash_value(&(request.model.as_str(), input.as_str()));
                EmbeddingData {
                    index: idx as u32,
                    embedding: vec![
                        (base & 0xff) as f32 / 255.0,
                        ((base >> 8) & 0xff) as f32 / 255.0,
                        ((base >> 16) & 0xff) as f32 / 255.0,
                        ((base >> 24) & 0xff) as f32 / 255.0,
                    ],
                }
            })
            .collect();

        let prompt_tokens = match &request.inputs {
            ClassificationInputs::Texts(texts) => texts
                .iter()
                .map(|text| text.split_whitespace().count() as u32)
                .sum(),
            ClassificationInputs::TokenIds(token_ids) => {
                token_ids.iter().map(|ids| ids.len() as u32).sum()
            }
        };

        Ok(EmbedResult {
            model: request.model,
            data,
            prompt_tokens,
            dimensions: 4,
        })
    }
}

pub fn make_app(with_chat_template: bool, api_keys: Vec<String>) -> axum::Router {
    let chat_template = with_chat_template.then(mock_chat_template);
    build_router(
        Arc::new(ContractTestEngine::new("test-model")),
        chat_template,
        api_keys,
    )
}

pub fn mock_chat_template() -> Arc<ModelChatTemplate> {
    Arc::new(ModelChatTemplate::from_template_string(
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{{ '<|im_start|>assistant\\n' }}".to_string(),
    ))
}

pub async fn send_json(app: &axum::Router, req: Request<Body>) -> (StatusCode, serde_json::Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&body).unwrap_or(serde_json::json!(null));
    (status, json)
}

pub async fn send_text(
    app: &axum::Router,
    req: Request<Body>,
) -> (StatusCode, String, Option<String>) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string());
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        String::from_utf8(body.to_vec()).unwrap(),
        content_type,
    )
}

pub fn sse_data_lines(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|line| line.to_string())
        .collect()
}

pub fn find_free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

pub fn bearer_request(method: &str, path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

pub fn empty_request(method: &str, path: &str) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .body(Body::empty())
        .unwrap()
}

fn fixed_metrics() -> DecodeMetrics {
    DecodeMetrics {
        ttft_ms: 5.0,
        prefill_ms: 2.0,
        decode_ms: 3.0,
        total_ms: 8.0,
    }
}

fn find_stop_pos(text: &str, stops: &[String]) -> Option<usize> {
    stops.iter().filter_map(|stop| text.find(stop)).min()
}
