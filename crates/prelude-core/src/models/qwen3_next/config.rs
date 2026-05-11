use crate::models::commons::HybridAttentionPattern;
use crate::models::model_config;

// ── Config ──────────────────────────────────────────────────────────────

model_config! {
    pub struct Qwen3NextConfig("Qwen3Next") {
        required {
            vocab_size: usize,
            hidden_size: usize,
            intermediate_size: usize,
            num_hidden_layers: usize,
            num_attention_heads: usize,
            num_key_value_heads: usize,
            head_dim: usize,
            max_position_embeddings: usize,
        }
        serde_default {
            norm_topk_prob: bool,
            tie_word_embeddings: bool,
        }
        warn_default {
            rms_norm_eps: f64 = 1e-6,
            rope_theta: f64 = 10_000_000.0,
            partial_rotary_factor: f64 = 0.25,
            full_attention_interval: usize = 4,
            // DeltaNet
            linear_num_key_heads: usize = 16,
            linear_num_value_heads: usize = 32,
            linear_key_head_dim: usize = 128,
            linear_value_head_dim: usize = 128,
            linear_conv_kernel_dim: usize = 4,
            // MoE
            num_experts: usize = 512,
            num_experts_per_tok: usize = 10,
            moe_intermediate_size: usize = 512,
            shared_expert_intermediate_size: usize = 512,
            decoder_sparse_step: usize = 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LayerType {
    LinearAttention,
    FullAttention,
}

impl Qwen3NextConfig {
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
}
