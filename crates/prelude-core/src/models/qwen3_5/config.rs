use crate::models::commons::HybridAttentionPattern;
use crate::models::resolve_or_warn;

// ── Config ──────────────────────────────────────────────────────────────

/// Custom deserializer that handles nested `text_config` for VL models
/// and `rope_parameters.base` for rope_theta.
#[derive(Debug, Clone)]
pub struct Qwen3_5Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub full_attention_interval: usize,
    pub attn_output_gate: bool,
    // DeltaNet
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub tie_word_embeddings: bool,
    // MoE (None for dense models)
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub shared_expert_intermediate_size: Option<usize>,
    pub norm_topk_prob: bool,
}

/// Raw serde struct — all defaultable fields are Option<T> so we can warn on fallback.
#[derive(serde::Deserialize)]
struct RawQwen3_5Config {
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    full_attention_interval: Option<usize>,
    #[serde(default)]
    attn_output_gate: Option<bool>,
    // DeltaNet
    #[serde(default)]
    linear_num_key_heads: Option<usize>,
    #[serde(default)]
    linear_num_value_heads: Option<usize>,
    #[serde(default)]
    linear_key_head_dim: Option<usize>,
    #[serde(default)]
    linear_value_head_dim: Option<usize>,
    #[serde(default)]
    linear_conv_kernel_dim: Option<usize>,
    #[serde(default)]
    tie_word_embeddings: bool,
    // MoE fields (None for dense models)
    #[serde(default)]
    num_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    moe_intermediate_size: Option<usize>,
    #[serde(default)]
    shared_expert_intermediate_size: Option<usize>,
    #[serde(default)]
    norm_topk_prob: Option<bool>,
    // rope_parameters for extracting rope_theta
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
}

const MODEL: &str = "Qwen3.5";

impl<'de> serde::Deserialize<'de> for Qwen3_5Config {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw: serde_json::Value = serde::Deserialize::deserialize(deserializer)?;

        // `tie_word_embeddings` can live at the top level (Qwen3.5-35B-A3B) or
        // inside `text_config` (dense Qwen3.5 variants). Read top-level first.
        let top_tie = raw.get("tie_word_embeddings").and_then(|v| v.as_bool());

        // If this is a VL model with text_config, extract the sub-object
        let text_val = if let Some(tc) = raw.get("text_config") {
            tc.clone()
        } else {
            raw.clone()
        };

        let r: RawQwen3_5Config =
            serde_json::from_value(text_val).map_err(serde::de::Error::custom)?;

        // Extract rope_theta: try direct field, then rope_parameters.rope_theta/base, then rope_scaling
        let rope_theta = r.rope_theta.or_else(|| {
            r.rope_parameters
                .as_ref()
                .and_then(|v| {
                    v.get("rope_theta")
                        .or_else(|| v.get("base"))
                        .and_then(|b| b.as_f64())
                })
                .or_else(|| {
                    r.rope_scaling
                        .as_ref()
                        .and_then(|v| v.get("base"))
                        .and_then(|b| b.as_f64())
                })
        });

        // Extract partial_rotary_factor: try direct field, then nested in rope_parameters
        // (Qwen3.5-35B-A3B stores it there; dense variants store it flat).
        let partial_rotary_factor = r.partial_rotary_factor.or_else(|| {
            r.rope_parameters
                .as_ref()
                .and_then(|v| v.get("partial_rotary_factor"))
                .and_then(|b| b.as_f64())
        });

        // tie_word_embeddings fallback: top-level JSON → text_config field → default false.
        let tie_word_embeddings = top_tie.unwrap_or(r.tie_word_embeddings);

        Ok(Qwen3_5Config {
            vocab_size: resolve_or_warn!(r.vocab_size, 248320, "vocab_size", MODEL),
            hidden_size: resolve_or_warn!(r.hidden_size, 2048, "hidden_size", MODEL),
            intermediate_size: resolve_or_warn!(
                r.intermediate_size,
                6144,
                "intermediate_size",
                MODEL
            ),
            num_hidden_layers: resolve_or_warn!(
                r.num_hidden_layers,
                24,
                "num_hidden_layers",
                MODEL
            ),
            num_attention_heads: resolve_or_warn!(
                r.num_attention_heads,
                16,
                "num_attention_heads",
                MODEL
            ),
            num_key_value_heads: resolve_or_warn!(
                r.num_key_value_heads,
                2,
                "num_key_value_heads",
                MODEL
            ),
            head_dim: resolve_or_warn!(r.head_dim, 256, "head_dim", MODEL),
            max_position_embeddings: resolve_or_warn!(
                r.max_position_embeddings,
                262144,
                "max_position_embeddings",
                MODEL
            ),
            rms_norm_eps: resolve_or_warn!(r.rms_norm_eps, 1e-6, "rms_norm_eps", MODEL),
            rope_theta: resolve_or_warn!(rope_theta, 10_000_000.0, "rope_theta", MODEL),
            partial_rotary_factor: resolve_or_warn!(
                partial_rotary_factor,
                0.25,
                "partial_rotary_factor",
                MODEL
            ),
            full_attention_interval: resolve_or_warn!(
                r.full_attention_interval,
                4,
                "full_attention_interval",
                MODEL
            ),
            attn_output_gate: resolve_or_warn!(r.attn_output_gate, true, "attn_output_gate", MODEL),
            linear_num_key_heads: resolve_or_warn!(
                r.linear_num_key_heads,
                16,
                "linear_num_key_heads",
                MODEL
            ),
            linear_num_value_heads: resolve_or_warn!(
                r.linear_num_value_heads,
                16,
                "linear_num_value_heads",
                MODEL
            ),
            linear_key_head_dim: resolve_or_warn!(
                r.linear_key_head_dim,
                128,
                "linear_key_head_dim",
                MODEL
            ),
            linear_value_head_dim: resolve_or_warn!(
                r.linear_value_head_dim,
                128,
                "linear_value_head_dim",
                MODEL
            ),
            linear_conv_kernel_dim: resolve_or_warn!(
                r.linear_conv_kernel_dim,
                4,
                "linear_conv_kernel_dim",
                MODEL
            ),
            tie_word_embeddings,
            num_experts: r.num_experts,
            num_experts_per_tok: r.num_experts_per_tok,
            moe_intermediate_size: r.moe_intermediate_size,
            shared_expert_intermediate_size: r.shared_expert_intermediate_size,
            norm_topk_prob: resolve_or_warn!(r.norm_topk_prob, true, "norm_topk_prob", MODEL),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LayerType {
    LinearAttention,
    FullAttention,
}

impl Qwen3_5Config {
    pub(super) fn attention_pattern(&self) -> HybridAttentionPattern {
        HybridAttentionPattern::new(self.num_hidden_layers, self.full_attention_interval)
    }

    pub(super) fn layer_type(&self, idx: usize) -> LayerType {
        if self.attention_pattern().is_full_attention_layer(idx) {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        }
    }

    pub(super) fn full_attention_layer_count(&self) -> usize {
        self.attention_pattern().full_attention_layers()
    }

    pub(super) fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub(super) fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Convolution dimension: Q + K + V flattened (Z is separate, not convolved).
    pub(super) fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }

    pub(super) fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    pub(super) fn is_moe(&self) -> bool {
        self.num_experts.is_some()
    }
}
