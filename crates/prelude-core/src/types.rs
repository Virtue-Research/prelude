use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// Re-export shared types from the scheduler module (single source of truth).
pub use crate::scheduler::{FinishReason, SamplingParams};

// ── Logprobs Core Types ──

/// Per-token logprob info returned by the engine (internal representation).
#[derive(Debug, Clone)]
pub struct TokenLogprobInfo {
    pub token: String,
    pub token_id: u32,
    pub logprob: f32,
    /// Top-k (token_id, token_string, logprob) entries, sorted descending by logprob.
    pub top_logprobs: Vec<(u32, String, f32)>,
}

// ── Logprobs API Types (OpenAI-compatible) ──

/// Logprobs object for `/v1/completions` (flat dict format per OpenAI spec).
#[derive(Debug, Clone, Serialize)]
pub struct CompletionLogprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    pub text_offset: Vec<u32>,
    /// Each entry maps token string → logprob for the top candidates.
    pub top_logprobs: Vec<HashMap<String, f32>>,
}

/// Logprobs object for `/v1/chat/completions` (structured list format).
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionLogprobs {
    pub content: Vec<ChatCompletionLogprobContent>,
}

/// A single token's logprob info in chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionLogprobContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<ChatCompletionTopLogprob>,
}

/// A candidate token in the top_logprobs list for chat completions.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionTopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: CompletionPrompt,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub user: Option<String>,
    /// Number of top log probabilities to return per token (null = don't return).
    #[serde(default)]
    pub logprobs: Option<u32>,
    /// Number of top log probabilities to return per prompt token (vLLM extension).
    #[serde(default)]
    pub prompt_logprobs: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    // TODO: implement n > 1 (multiple choices)
    #[serde(default)]
    pub n: Option<u32>,
    // TODO: implement frequency/presence penalty in sampling
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
}

impl CompletionRequest {
    pub fn validate_public_request(&self) -> Result<(), String> {
        validate_single_choice(self.n)?;
        reject_if_present("user", self.user.is_some())?;
        reject_if_present("frequency_penalty", self.frequency_penalty.is_some())?;
        reject_if_present("presence_penalty", self.presence_penalty.is_some())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogprobs>,
    /// Per-prompt-token logprobs (vLLM extension). First element is always None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<Vec<Option<HashMap<u32, PromptLogprobEntry>>>>,
    /// Prompt token IDs (returned when prompt_logprobs is requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_ids: Option<Vec<u32>>,
}

/// A single logprob entry in the prompt_logprobs response (vLLM-compatible).
#[derive(Debug, Clone, Serialize)]
pub struct PromptLogprobEntry {
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoded_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

// ── Chat Completions (OpenAI-compatible) ──

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Preferred over `max_tokens` (OpenAI's new name).
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    /// Deprecated by OpenAI in favor of `max_completion_tokens`, kept for compatibility.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub user: Option<String>,
    /// Whether to return log probabilities of the output tokens.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top log probabilities to return per token (requires logprobs=true).
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    // TODO: implement n > 1 (multiple choices)
    #[serde(default)]
    pub n: Option<u32>,
    // TODO: implement frequency/presence penalty in sampling
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    // TODO: implement response_format (json_object mode)
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

impl ChatCompletionRequest {
    pub fn validate_public_request(&self) -> Result<(), String> {
        validate_single_choice(self.n)?;
        reject_if_present("user", self.user.is_some())?;
        reject_if_present("frequency_penalty", self.frequency_penalty.is_some())?;
        reject_if_present("presence_penalty", self.presence_penalty.is_some())?;
        reject_if_present("response_format", self.response_format.is_some())?;

        if self.top_logprobs.is_some() && !self.logprobs.unwrap_or(false) {
            return Err("top_logprobs requires logprobs=true".to_string());
        }

        for (index, message) in self.messages.iter().enumerate() {
            message.validate_text_only(index)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: ChatMessageContent,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Value>,
}

impl ChatMessage {
    pub fn text_content(&self) -> Option<&str> {
        self.content.as_text()
    }

    pub fn validate_text_only(&self, index: usize) -> Result<(), String> {
        reject_if_present(&format!("messages[{index}].name"), self.name.is_some())?;
        reject_if_present(
            &format!("messages[{index}].tool_call_id"),
            self.tool_call_id.is_some(),
        )?;
        reject_if_present(
            &format!("messages[{index}].tool_calls"),
            self.tool_calls.is_some(),
        )?;

        if self.text_content().is_none() {
            return Err(format!(
                "messages[{index}].content must be a plain string; multimodal/tool-style content is not supported"
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Other(Value),
}

impl ChatMessageContent {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text.as_str()),
            Self::Other(_) => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

// TODO: implement json_object response format
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<ChatMessageOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<ChatMessageOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatCompletionLogprobs>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessageOut {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ── Health / Models ──

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub uptime_s: f64,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Clone)]
pub enum PromptInput {
    Text(String),
    TokenIds(Vec<u32>),
}


#[derive(Debug, Clone, Default)]
pub struct StopConfig {
    pub strings: Vec<String>,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub request_id: String,
    pub model: String,
    pub input: PromptInput,
    pub sampling: SamplingParams,
    pub max_new_tokens: u32,
    pub stop: StopConfig,
    pub seed: Option<u64>,
    pub deadline_ms: Option<u64>,
    /// If Some(k), return top-k log probabilities per generated token.
    pub logprobs: Option<u32>,
    /// If Some(k), return top-k log probabilities per prompt token (vLLM extension).
    pub prompt_logprobs: Option<u32>,
}


#[derive(Debug, Clone)]
pub struct DecodeMetrics {
    pub ttft_ms: f32,
    pub prefill_ms: f32,
    pub decode_ms: f32,
    pub total_ms: f32,
}

#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub model: String,
    pub output_token_ids: Vec<u32>,
    pub output_text: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub metrics: DecodeMetrics,
    /// Per-token logprob info (populated when logprobs was requested).
    pub token_logprobs: Option<Vec<TokenLogprobInfo>>,
    /// Per-prompt-token logprob info (populated when prompt_logprobs was requested).
    pub prompt_token_logprobs: Option<Vec<TokenLogprobInfo>>,
}

/// Events emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Generation started (after prefill). Maps to SSE role chunk.
    Started,
    /// A new token's text delta was decoded.
    Token {
        text: String,
        logprobs: Option<TokenLogprobInfo>,
    },
    /// Generation finished with final metrics.
    Finished {
        finish_reason: FinishReason,
        usage: Usage,
        metrics: DecodeMetrics,
    },
    /// Generation failed with an error.
    Error {
        message: String,
    },
}

// ── Classification API (Virtue Guardrail) ──

/// Input formats for classification - uses untagged deserialization
/// to auto-detect format from JSON
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ClassificationInput {
    /// Batch of pre-tokenized inputs: [[2980, 374], [9241, 3366]]
    BatchTokenIds(Vec<Vec<u32>>),
    /// Single pre-tokenized input: [2980, 374, 264, 1273]
    TokenIds(Vec<u32>),
    /// Batch of text strings: ["text1", "text2"]
    Batch(Vec<String>),
    /// Single text string: "How do I make a bomb?"
    Single(String),
}

/// Request body for POST /v1/classify
#[derive(Debug, Clone, Deserialize)]
pub struct ClassificationRequest {
    /// Model name/path (e.g., "./classifier_glean_int8")
    pub model: String,
    /// Text input(s) - completion style
    #[serde(default)]
    pub input: Option<ClassificationInput>,
    /// Chat-style input - will be formatted as "role: content\n..."
    #[serde(default)]
    pub messages: Option<Vec<ChatMessage>>,
}

impl ClassificationRequest {
    /// Convert messages to a formatted string for classification
    pub fn format_messages(messages: &[ChatMessage]) -> String {
        messages
            .iter()
            .filter_map(|message| {
                message
                    .text_content()
                    .map(|content| format!("{}: {}", message.role, content))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get the inputs as a list of strings or token IDs
    pub fn get_inputs(&self) -> Result<ClassificationInputs, &'static str> {
        match (&self.input, &self.messages) {
            (Some(_), Some(_)) => Err("Cannot provide both 'input' and 'messages'"),
            (None, None) => Err("Either 'input' or 'messages' must be provided"),
            (Some(input), None) => Ok(match input {
                ClassificationInput::Single(s) => ClassificationInputs::Texts(vec![s.clone()]),
                ClassificationInput::Batch(v) => ClassificationInputs::Texts(v.clone()),
                ClassificationInput::TokenIds(ids) => {
                    ClassificationInputs::TokenIds(vec![ids.clone()])
                }
                ClassificationInput::BatchTokenIds(ids) => {
                    ClassificationInputs::TokenIds(ids.clone())
                }
            }),
            (None, Some(messages)) => {
                for (index, message) in messages.iter().enumerate() {
                    message
                        .validate_text_only(index)
                        .map_err(|_| "Only plain string chat messages are supported")?;
                }
                let formatted = Self::format_messages(messages);
                Ok(ClassificationInputs::Texts(vec![formatted]))
            }
        }
    }
}

/// Normalized inputs for classification
#[derive(Debug, Clone)]
pub enum ClassificationInputs {
    Texts(Vec<String>),
    TokenIds(Vec<Vec<u32>>),
}

/// Single classification result
#[derive(Debug, Clone, Serialize)]
pub struct ClassificationResult {
    /// Position in input batch (0-based)
    pub index: u32,
    /// Predicted label from model's id2label mapping (argmax of probs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Logits for each class (raw scores)
    pub probs: Vec<f32>,
    /// Number of classes
    pub num_classes: u32,
}

/// Token usage for classification
#[derive(Debug, Clone, Serialize)]
pub struct ClassificationUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
    /// Always 0 for classification
    pub completion_tokens: u32,
}

/// Response body for POST /v1/classify
#[derive(Debug, Clone, Serialize)]
pub struct ClassificationResponse {
    /// Unique response ID (format: "classify-<hex>")
    pub id: String,
    /// Always "list"
    pub object: String,
    /// Unix timestamp
    pub created: i64,
    /// Model name used
    pub model: String,
    /// Classification results (one per input)
    pub data: Vec<ClassificationResult>,
    /// Token usage
    pub usage: ClassificationUsage,
}

/// Internal request for classification engine
#[derive(Debug, Clone)]
pub struct ClassifyRequest {
    pub request_id: String,
    pub model: String,
    pub inputs: ClassificationInputs,
}

/// Internal result from classification engine
#[derive(Debug, Clone)]
pub struct ClassifyResult {
    pub model: String,
    pub results: Vec<ClassificationResult>,
    pub prompt_tokens: u32,
}

// ── Embedding Types ─────────────────────────────────────────────────────

/// Internal request for embedding engine
#[derive(Debug, Clone)]
pub struct EmbedRequest {
    pub request_id: String,
    pub model: String,
    pub inputs: ClassificationInputs, // Reuse: can be texts or token_ids
}

/// Single embedding result
#[derive(Debug, Clone)]
pub struct EmbeddingData {
    /// Index in the input batch
    pub index: u32,
    /// Embedding vector
    pub embedding: Vec<f32>,
}

/// Internal result from embedding engine
#[derive(Debug, Clone)]
pub struct EmbedResult {
    pub model: String,
    pub data: Vec<EmbeddingData>,
    pub prompt_tokens: u32,
    /// Embedding dimension
    pub dimensions: usize,
}

// ── Embedding API Types (OpenAI-compatible) ─────────────────────────────

/// Input formats for embeddings - uses untagged deserialization
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Batch of strings: ["text1", "text2"]
    Batch(Vec<String>),
    /// Single string: "text"
    Single(String),
    /// Batch of token arrays: [[123, 456], [789]]
    BatchTokenIds(Vec<Vec<u32>>),
    /// Single token array: [123, 456, 789]
    TokenIds(Vec<u32>),
}

/// Request body for POST /v1/embeddings
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRequest {
    /// Input text(s) to embed
    pub input: EmbeddingInput,
    /// Model name/path
    pub model: String,
    /// Encoding format (default: "float")
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Number of dimensions (optional, for dimensionality reduction)
    #[serde(default)]
    pub dimensions: Option<usize>,
}

impl EmbeddingRequest {
    pub fn validate_public_request(&self) -> Result<(), String> {
        reject_if_present("dimensions", self.dimensions.is_some())?;

        if let Some(format) = self.encoding_format.as_deref()
            && !format.eq_ignore_ascii_case("float")
            && !format.eq_ignore_ascii_case("base64")
        {
            return Err(format!(
                "encoding_format={format:?} is not supported; use \"float\" or \"base64\""
            ));
        }

        Ok(())
    }

    /// Convert to internal ClassificationInputs format
    pub fn get_inputs(&self) -> ClassificationInputs {
        match &self.input {
            EmbeddingInput::Single(s) => ClassificationInputs::Texts(vec![s.clone()]),
            EmbeddingInput::Batch(v) => ClassificationInputs::Texts(v.clone()),
            EmbeddingInput::TokenIds(ids) => ClassificationInputs::TokenIds(vec![ids.clone()]),
            EmbeddingInput::BatchTokenIds(ids) => ClassificationInputs::TokenIds(ids.clone()),
        }
    }
}

/// Embedding value: either a float array or a base64-encoded string of raw f32 bytes.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Float(Vec<f32>),
    Base64(String),
}

/// Single embedding object in response
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingObject {
    /// Always "embedding"
    pub object: String,
    /// Index in the input list
    pub index: u32,
    /// The embedding vector (float array or base64 string)
    pub embedding: EmbeddingValue,
}

/// Response body for POST /v1/embeddings
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponse {
    /// Always "list"
    pub object: String,
    /// List of embedding objects
    pub data: Vec<EmbeddingObject>,
    /// Model used
    pub model: String,
    /// Token usage
    pub usage: EmbeddingUsage,
}

/// Token usage for embeddings
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

fn validate_single_choice(n: Option<u32>) -> Result<(), String> {
    if let Some(value) = n && value != 1 {
        return Err(format!("n={value} is not supported; only n=1 is supported"));
    }
    Ok(())
}

fn reject_if_present(field: &str, is_present: bool) -> Result<(), String> {
    if is_present {
        return Err(format!("{field} is not supported in this release"));
    }
    Ok(())
}
