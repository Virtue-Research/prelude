use std::collections::HashMap;
use std::sync::Arc;

use crate::tensor::{DType, Device, Module, Result, Tensor};
use crate::models::commons::activation::Activation;
use crate::models::commons::embedding::Embedding;
use crate::loading::var_builder::VarBuilder;
use serde::Deserialize;

use crate::models::commons::{
    BatchState, Linear, RmsNorm,
};
use crate::ops::{MaskType, VarlenParams};

use crate::engine::{EmbeddingActivation, EmbeddingSemantics};

use crate::models::model_config;

// ── Gemma3 Config ────────────────────────────────────────────────────────

model_config! {
    /// Gemma3 text model configuration
    pub struct Gemma3Config("Gemma3") {
        required {
            hidden_size: usize,
            intermediate_size: usize,
            num_hidden_layers: usize,
            num_attention_heads: usize,
            num_key_value_heads: usize,
            head_dim: usize,
        }
        serde_default {
            sliding_window: Option<usize>,
            final_logit_softcapping: Option<f64>,
            attn_logit_softcapping: Option<f64>,
            attention_bias: bool,
            layer_types: Option<Vec<String>>,
            use_bidirectional_attention: bool,
        }
        warn_default {
            vocab_size: usize = 262144,
            max_position_embeddings: usize = 32768,
            hidden_activation: Activation = Activation::GeluPytorchTanh,
            rms_norm_eps: f64 = 1e-6,
            rope_theta: f64 = 1_000_000.0,
            rope_local_base_freq: f64 = 10_000.0,
            sliding_window_pattern: usize = 6,
            query_pre_attn_scalar: usize = 256,
            tie_word_embeddings: bool = true,
        }
    }
}

impl Gemma3Config {
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Configuration for Gemma3 sequence classification model
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma3ClassifierConfig {
    #[serde(flatten)]
    pub base: Gemma3Config,
    pub num_labels: usize,
    #[serde(default)]
    pub label2id: Option<HashMap<String, usize>>,
    #[serde(default)]
    pub id2label: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma3AttentionMode {
    FullCausal,
    SlidingCausal { window_size: usize },
    FullBidirectional,
    SlidingBidirectional { window_size: usize },
}

impl Gemma3AttentionMode {
    fn is_sliding(self) -> bool {
        matches!(
            self,
            Self::SlidingCausal { .. } | Self::SlidingBidirectional { .. }
        )
    }
}

fn attention_mode_for_layer(cfg: &Gemma3Config, layer_idx: usize) -> Gemma3AttentionMode {
    let is_sliding = if let Some(layer_types) = cfg.layer_types.as_ref() {
        matches!(
            layer_types.get(layer_idx).map(String::as_str),
            Some("sliding_attention")
        )
    } else if cfg.sliding_window.is_some() {
        (layer_idx + 1) % cfg.sliding_window_pattern != 0
    } else {
        false
    };

    let is_bidirectional = cfg.use_bidirectional_attention;
    let window_size = cfg.sliding_window.unwrap_or(cfg.max_position_embeddings);

    match (is_sliding, is_bidirectional) {
        (false, false) => Gemma3AttentionMode::FullCausal,
        (true, false) => Gemma3AttentionMode::SlidingCausal { window_size },
        (false, true) => Gemma3AttentionMode::FullBidirectional,
        (true, true) => Gemma3AttentionMode::SlidingBidirectional { window_size },
    }
}

// ── Rotary Embedding ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Gemma3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Gemma3RotaryEmbedding {
    fn new(dtype: DType, cfg: &Gemma3Config, dev: &Device, rope_theta: f64) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.broadcast_mul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply_varlen(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.index_select(position_ids, 0)?;
        let sin = self.sin.index_select(position_ids, 0)?;
        let (total, h_q, d) = q.dims3()?;
        let h_k = k.dim(1)?;
        let q4 = q.reshape((1, total, h_q, d))?;
        let k4 = k.reshape((1, total, h_k, d))?;
        let q_embed = crate::ops::rope_thd(&q4, &cos, &sin)?;
        let k_embed = crate::ops::rope_thd(&k4, &cos, &sin)?;
        Ok((
            q_embed.reshape((total, h_q, d))?,
            k_embed.reshape((total, h_k, d))?,
        ))
    }
}

// ── MLP ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Gemma3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Gemma3Mlp {
    fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::load(
                vb.pp("gate_proj"),
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
            )?,
            up_proj: Linear::load(
                vb.pp("up_proj"),
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
            )?,
            down_proj: Linear::load(
                vb.pp("down_proj"),
                cfg.intermediate_size,
                cfg.hidden_size,
                false,
            )?,
            act_fn: cfg.hidden_activation,
        })
    }

    fn forward(&self, ops: &dyn crate::ops::Ops, x: &Tensor) -> Result<Tensor> {
        let bs = BatchState::no_lora();
        let gate = self.gate_proj.forward(x, &bs, ops)?;
        let up = self.up_proj.forward(x, &bs, ops)?;
        self.down_proj.forward(&(gate.apply(&self.act_fn)? * up)?, &bs, ops)
    }
}

// ── Attention ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Gemma3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<Gemma3RotaryEmbedding>,
    softmax_scale: f32,
    attention_mode: Gemma3AttentionMode,
}

impl Gemma3Attention {
    fn new(
        cfg: &Gemma3Config,
        rotary_emb: Arc<Gemma3RotaryEmbedding>,
        vb: VarBuilder,
        attention_mode: Gemma3AttentionMode,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        // Gemma3 uses query_pre_attn_scalar for scaling
        let softmax_scale = 1.0 / (cfg.query_pre_attn_scalar as f32).sqrt();

        Ok(Self {
            q_proj: Linear::load(vb.pp("q_proj"), cfg.hidden_size, num_heads * head_dim, cfg.attention_bias)?,
            k_proj: Linear::load(vb.pp("k_proj"), cfg.hidden_size, num_kv_heads * head_dim, cfg.attention_bias)?,
            v_proj: Linear::load(vb.pp("v_proj"), cfg.hidden_size, num_kv_heads * head_dim, cfg.attention_bias)?,
            o_proj: Linear::load(vb.pp("o_proj"), num_heads * head_dim, cfg.hidden_size, cfg.attention_bias)?,
            q_norm: {
                let weight = vb.pp("q_norm").get(head_dim, "weight")?;
                let weight = (&weight + 1.0)?;
                RmsNorm::from_weight(weight, cfg.rms_norm_eps)
            },
            k_norm: {
                let weight = vb.pp("k_norm").get(head_dim, "weight")?;
                let weight = (&weight + 1.0)?;
                RmsNorm::from_weight(weight, cfg.rms_norm_eps)
            },
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            softmax_scale,
            attention_mode,
        })
    }

    fn forward(
        &mut self,
        ops: &dyn crate::ops::Ops,
        packed_input: &Tensor,
        cu_seqlens: &Tensor,
        max_seqlen: usize,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (total_tokens, _) = packed_input.dims2()?;
        let bs = BatchState::no_lora();

        let q = self.q_proj.forward(packed_input, &bs, ops)?;
        let k = self.k_proj.forward(packed_input, &bs, ops)?;
        let v = self.v_proj.forward(packed_input, &bs, ops)?;

        let q = q.reshape((total_tokens, self.num_heads, self.head_dim))?;
        let k = k.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (q, k) = self.rotary_emb.apply_varlen(&q, &k, position_ids)?;

        let mask = match self.attention_mode {
            Gemma3AttentionMode::FullCausal => MaskType::Causal,
            Gemma3AttentionMode::FullBidirectional => MaskType::Bidirectional,
            Gemma3AttentionMode::SlidingCausal { window_size } => MaskType::SlidingWindow {
                left: window_size.saturating_sub(1), right: 0,
            },
            Gemma3AttentionMode::SlidingBidirectional { window_size } => MaskType::SlidingWindow {
                left: window_size.saturating_sub(1), right: window_size.saturating_sub(1),
            },
        };
        let attn_out = ops.varlen_attention(&q, &k, &v, &VarlenParams {
            cu_seqlens_q: cu_seqlens, cu_seqlens_k: cu_seqlens,
            max_seqlen_q: max_seqlen, max_seqlen_k: max_seqlen,
            scale: self.softmax_scale, mask, softcap: None,
        })?;

        let attn_dim = self.num_heads * self.head_dim;
        self.o_proj.forward(&attn_out.reshape((total_tokens, attn_dim))?, &bs, ops)
    }

    fn clear_kv_cache(&mut self) {}
}

// ── Decoder Layer ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Gemma3DecoderLayer {
    self_attn: Gemma3Attention,
    mlp: Gemma3Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl Gemma3DecoderLayer {
    fn new(
        cfg: &Gemma3Config,
        rotary_emb: Arc<Gemma3RotaryEmbedding>,
        vb: VarBuilder,
        attention_mode: Gemma3AttentionMode,
    ) -> Result<Self> {
        let self_attn = Gemma3Attention::new(cfg, rotary_emb, vb.pp("self_attn"), attention_mode)?;
        let mlp = Gemma3Mlp::new(cfg, vb.pp("mlp"))?;

        // Load weights and create adjusted weights (+1 for Gemma)
        let input_ln_weight = vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?;
        let post_attn_ln_weight = vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")?;
        let pre_ffn_ln_weight = vb
            .pp("pre_feedforward_layernorm")
            .get(cfg.hidden_size, "weight")?;
        let post_ffn_ln_weight = vb
            .pp("post_feedforward_layernorm")
            .get(cfg.hidden_size, "weight")?;

        let input_ln_adjusted = (&input_ln_weight + 1.0)?;
        let post_attn_ln_adjusted = (&post_attn_ln_weight + 1.0)?;
        let pre_ffn_ln_adjusted = (&pre_ffn_ln_weight + 1.0)?;
        let post_ffn_ln_adjusted = (&post_ffn_ln_weight + 1.0)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: RmsNorm::from_weight(input_ln_adjusted, cfg.rms_norm_eps),
            post_attention_layernorm: RmsNorm::from_weight(post_attn_ln_adjusted, cfg.rms_norm_eps),
            pre_feedforward_layernorm: RmsNorm::from_weight(pre_ffn_ln_adjusted, cfg.rms_norm_eps),
            post_feedforward_layernorm: RmsNorm::from_weight(post_ffn_ln_adjusted, cfg.rms_norm_eps),
        })
    }

    fn forward(
        &mut self,
        ops: &dyn crate::ops::Ops,
        xs: &Tensor,
        cu_seqlens: &Tensor,
        max_seqlen: usize,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(xs)?;
        let attn_output =
            self.self_attn
                .forward(ops, &normed, cu_seqlens, max_seqlen, position_ids)?;

        let post_attn_normed = self.post_attention_layernorm.forward(&attn_output)?;
        let xs = ops.add_or_fused(&post_attn_normed, xs)?;

        let pre_ffn_normed = self.pre_feedforward_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(ops, &pre_ffn_normed)?;

        let post_ffn_normed = self.post_feedforward_layernorm.forward(&mlp_output)?;
        ops.add_or_fused(&post_ffn_normed, &xs)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Base Model ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Gemma3Model {
    embed_tokens: Embedding,
    layers: Vec<Gemma3DecoderLayer>,
    norm: RmsNorm,
    hidden_size: usize,
}

impl Gemma3Model {
    fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = {
            let emb_vb = vb.pp("embed_tokens");
            let weight = emb_vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            Embedding::new(weight, cfg.hidden_size)
        };

        // Create global rotary embedding (full context)
        let global_rotary = Arc::new(Gemma3RotaryEmbedding::new(
            vb.dtype(),
            cfg,
            vb.device(),
            cfg.rope_theta,
        )?);

        // Create local rotary embedding (for sliding window layers)
        let local_rotary = Arc::new(Gemma3RotaryEmbedding::new(
            vb.dtype(),
            cfg,
            vb.device(),
            cfg.rope_local_base_freq,
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let attention_mode = attention_mode_for_layer(cfg, i);
            let rotary = if attention_mode.is_sliding() {
                local_rotary.clone()
            } else {
                global_rotary.clone()
            };

            layers.push(Gemma3DecoderLayer::new(
                cfg,
                rotary,
                vb.pp(&format!("layers.{}", i)),
                attention_mode,
            )?);
        }

        let norm_weight_raw = vb.pp("norm").get(cfg.hidden_size, "weight")?;
        let norm_weight = (&norm_weight_raw + 1.0)?; // Gemma adds 1 to weights
        let norm = RmsNorm::from_weight(norm_weight.clone(), cfg.rms_norm_eps);

        let _ = norm_weight; // consumed by norm
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(
        &mut self,
        ops: &dyn crate::ops::Ops,
        packed_input: &Tensor,
        cu_seqlens: &Tensor,
        max_seqlen: usize,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let embed_scale = (self.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(packed_input)? * embed_scale)?;

        for layer in &mut self.layers {
            xs = layer.forward(ops, &xs, cu_seqlens, max_seqlen, position_ids)?;
        }

        self.norm.forward(&xs)
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

fn pool_mean_varlen(hidden: &Tensor, seq_lens: &[usize]) -> Result<Tensor> {
    let mut pooled = Vec::with_capacity(seq_lens.len());
    let mut start = 0usize;
    for &len in seq_lens {
        let seq = hidden.narrow(0, start, len)?;
        pooled.push(seq.mean(0)?);
        start += len;
    }
    Tensor::stack(&pooled, 0)
}

// ── Causal LM Model ──────────────────────────────────────────────────────

/// Gemma3 model for causal language modeling
#[derive(Debug, Clone)]
pub struct Gemma3ForCausalLM {
    base: Gemma3Model,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
}

impl Gemma3ForCausalLM {
    pub fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with_parts(cfg, vb.pp("model"), vb)
    }

    pub fn new_with_parts(
        cfg: &Gemma3Config,
        model_vb: VarBuilder,
        head_vb: VarBuilder,
    ) -> Result<Self> {
        let base = Gemma3Model::new(cfg, model_vb.clone())?;

        // For tied embeddings, use embedding weights as lm_head
        // Gemma3 defaults to tie_word_embeddings=true
        let lm_head = if cfg.tie_word_embeddings {
            let embed_weight = model_vb
                .pp("embed_tokens")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            Linear::from_weight(embed_weight, None)?
        } else {
            Linear::load(head_vb.pp("lm_head"), cfg.hidden_size, cfg.vocab_size, false)?
        };

        Ok(Self {
            base,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
        })
    }

    pub fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut crate::models::commons::BatchAttnContext,
    ) -> Result<Tensor> {
        let hidden = self.base.forward(
            ctx.ops,
            packed_input,
            ctx.cu_seqlens_q,
            ctx.max_seqlen_q,
            ctx.position_ids,
        )?;
        let last_hidden =
            crate::models::commons::last_token_select(&hidden, ctx.seq_lens)?.contiguous()?;

        let logits = self.lm_head.forward(&last_hidden.unsqueeze(1)?, &BatchState::no_lora(), ctx.ops)?;

        if let Some(cap) = self.final_logit_softcapping {
            let scaled = (&logits / cap)?;
            let tanh = scaled.tanh()?;
            tanh * cap
        } else {
            Ok(logits)
        }
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}

impl crate::models::LogitsSplitModel for Gemma3ForCausalLM {
    fn forward_hidden_states(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut crate::models::commons::BatchAttnContext,
    ) -> crate::tensor::Result<Tensor> {
        self.base.forward(
            ctx.ops,
            packed_input,
            ctx.cu_seqlens_q,
            ctx.max_seqlen_q,
            ctx.position_ids,
        )
    }

    fn compute_logits(&self, hidden: &Tensor) -> crate::tensor::Result<Tensor> {
        let logits = self.lm_head.forward(hidden, &BatchState::no_lora(), crate::ops::select_ops(hidden.device()))?;
        if let Some(cap) = self.final_logit_softcapping {
            let scaled = (&logits / cap)?;
            let tanh = scaled.tanh()?;
            tanh * cap
        } else {
            Ok(logits)
        }
    }
}

impl crate::models::ModelForward for Gemma3ForCausalLM {
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut crate::models::commons::BatchAttnContext,
    ) -> crate::tensor::Result<Tensor> {
        self.forward(packed_input, ctx)
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache();
    }

    fn as_logits_model(&self) -> Option<&dyn crate::models::LogitsSplitModel> {
        Some(self)
    }

    fn as_logits_model_mut(&mut self) -> Option<&mut dyn crate::models::LogitsSplitModel> {
        Some(self)
    }
}

// ── Sequence Classification Model ────────────────────────────────────────

/// Gemma3 model for sequence classification
#[derive(Debug, Clone)]
pub struct Gemma3ForSequenceClassification {
    base: Gemma3Model,
    score: Linear,
    num_labels: usize,
    id2label: Option<HashMap<usize, String>>,
}

impl Gemma3ForSequenceClassification {
    pub fn new(cfg: &Gemma3ClassifierConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_parts(cfg, vb.pp("model"), vb)
    }

    pub fn new_with_parts(
        cfg: &Gemma3ClassifierConfig,
        model_vb: VarBuilder,
        head_vb: VarBuilder,
    ) -> Result<Self> {
        let base = Gemma3Model::new(&cfg.base, model_vb)?;
        let score =
            Linear::load(head_vb.pp("score"), cfg.base.hidden_size, cfg.num_labels, false)?;

        // Convert id2label from String keys to usize keys
        let id2label = cfg.id2label.as_ref().map(|m| {
            m.iter()
                .filter_map(|(k, v)| k.parse::<usize>().ok().map(|id| (id, v.clone())))
                .collect()
        });

        Ok(Self {
            base,
            score,
            num_labels: cfg.num_labels,
            id2label,
        })
    }

    pub fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut crate::models::commons::BatchAttnContext,
    ) -> Result<Tensor> {
        let hidden_states = self.base.forward(
            ctx.ops,
            packed_input,
            ctx.cu_seqlens_q,
            ctx.max_seqlen_q,
            ctx.position_ids,
        )?;
        let last_hidden = crate::models::commons::last_token_select(&hidden_states, ctx.seq_lens)?;
        self.score.forward(&last_hidden, &BatchState::no_lora(), ctx.ops)
    }

    pub fn get_label(&self, class_idx: usize) -> Option<String> {
        self.id2label
            .as_ref()
            .and_then(|m| m.get(&class_idx).cloned())
            .or_else(|| Some(format!("LABEL_{}", class_idx)))
    }

    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
struct Gemma3EmbeddingDenseLayer {
    linear: Linear,
    activation: EmbeddingActivation,
}

impl Gemma3EmbeddingDenseLayer {
    fn new(spec: &crate::engine::EmbeddingDenseLayerSpec, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(vb.pp("linear"), spec.in_features, spec.out_features, spec.bias)?,
            activation: spec.activation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear.forward(x, &BatchState::no_lora(), crate::ops::select_ops(x.device()))?;
        match self.activation {
            EmbeddingActivation::Identity => Ok(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Gemma3ForEmbedding {
    base: Gemma3Model,
    pooling: crate::engine::EmbeddingPooling,
    dense_layers: Vec<Gemma3EmbeddingDenseLayer>,
    output_dim: usize,
}

impl Gemma3ForEmbedding {
    fn new(
        cfg: &Gemma3Config,
        model_vb: VarBuilder,
        semantics: &EmbeddingSemantics,
        auxiliary: &[Gemma3EmbeddingDenseLayer],
    ) -> Result<Self> {
        Ok(Self {
            base: Gemma3Model::new(cfg, model_vb)?,
            pooling: semantics.pooling,
            dense_layers: auxiliary.to_vec(),
            output_dim: semantics.output_dim(cfg.hidden_size),
        })
    }

    pub fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut crate::models::commons::BatchAttnContext,
    ) -> Result<Tensor> {
        let hidden_states = self.base.forward(
            ctx.ops,
            packed_input,
            ctx.cu_seqlens_q,
            ctx.max_seqlen_q,
            ctx.position_ids,
        )?;
        let hidden_states = hidden_states.to_dtype(DType::F32)?;
        let mut pooled = match self.pooling {
            crate::engine::EmbeddingPooling::LastToken => {
                crate::models::commons::last_token_select(&hidden_states, ctx.seq_lens)?
            }
            crate::engine::EmbeddingPooling::Mean => {
                pool_mean_varlen(&hidden_states, ctx.seq_lens)?
            }
            crate::engine::EmbeddingPooling::Cls => {
                crate::models::commons::first_token_select(&hidden_states, ctx.seq_lens)?
            }
        };

        for layer in &self.dense_layers {
            pooled = layer.forward(&pooled)?;
        }

        pooled.contiguous()
    }

    pub fn hidden_size(&self) -> usize {
        self.output_dim
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}

impl crate::models::ClassifierModel for Gemma3ForSequenceClassification {
    fn num_labels(&self) -> usize {
        Gemma3ForSequenceClassification::num_labels(self)
    }

    fn get_label(&self, class_idx: usize) -> Option<String> {
        Gemma3ForSequenceClassification::get_label(self, class_idx)
    }
}

impl crate::models::ModelForward for Gemma3ForSequenceClassification {
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut crate::models::commons::BatchAttnContext,
    ) -> crate::tensor::Result<Tensor> {
        self.forward(packed_input, ctx)
    }

    fn clear_kv_cache(&mut self) {
        Gemma3ForSequenceClassification::clear_kv_cache(self);
    }

    fn as_classifier(&self) -> Option<&dyn crate::models::ClassifierModel> {
        Some(self)
    }
}

impl crate::models::EmbeddingModel for Gemma3ForEmbedding {
    fn embedding_dim(&self) -> usize {
        self.hidden_size()
    }
}

impl crate::models::ModelForward for Gemma3ForEmbedding {
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut crate::models::commons::BatchAttnContext,
    ) -> crate::tensor::Result<Tensor> {
        Gemma3ForEmbedding::forward(self, packed_input, ctx)
    }

    fn clear_kv_cache(&mut self) {
        Gemma3ForEmbedding::clear_kv_cache(self);
    }

    fn as_embedding(&self) -> Option<&dyn crate::models::EmbeddingModel> {
        Some(self)
    }
}

// ── Registry / meta ─────────────────────────────────────────────────────

pub(crate) mod meta {
    use super::{
        Gemma3ClassifierConfig, Gemma3Config, Gemma3EmbeddingDenseLayer, Gemma3ForCausalLM,
        Gemma3ForEmbedding, Gemma3ForSequenceClassification,
    };
    use crate::loading::var_builder::VarBuilder;
    use crate::engine::EngineError;
    use crate::engine::{CommonModelConfig, EmbeddingSemantics, RuntimeCaps, TaskKind, WeightsBackend};
    use crate::models::registry::{
        ArchSpec, AuxiliaryVarBuilder, ParsedModelConfig, candle_model_err,
        inject_num_labels_if_missing, parse_value,
    };

    const ARCHITECTURE_ALIASES: &[&str] = &["Gemma3", "Gemma3Text"];
    const MODEL_TYPE_ALIASES: &[&str] = &["gemma3", "gemma3_text", "gemma2", "gemma"];
    const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate, TaskKind::Classify, TaskKind::Embed];

    #[derive(Debug, Clone, Copy)]
    enum Gemma3WeightLayout {
        /// Causal-LM checkpoint: `model.embed_tokens.weight`, `model.layers.*`, …
        FlatText,
        /// Multi-modal checkpoint with `language_model` wrapper:
        /// `model.language_model.model.embed_tokens.weight`, …
        NestedLanguageModel,
        /// Bare text-model checkpoint (e.g. `Gemma3TextModel` /
        /// `embeddinggemma-300m`): backbone weights live at the root —
        /// `embed_tokens.weight`, `layers.*` — with no `model.` prefix.
        BareText,
    }

    /// Opaque config stored in `ParsedModelConfig.arch_config` for Gemma3.
    enum Gemma3ArchConfig {
        Dense {
            cfg: Gemma3Config,
            layout: Gemma3WeightLayout,
        },
        Classifier {
            cfg: Gemma3ClassifierConfig,
            layout: Gemma3WeightLayout,
        },
        Embedding {
            cfg: Gemma3Config,
            layout: Gemma3WeightLayout,
        },
    }

    fn common_from_gemma3(cfg: &Gemma3Config) -> CommonModelConfig {
        CommonModelConfig {
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            max_position_embeddings: cfg.max_position_embeddings,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        }
    }

    fn infer_weight_layout(raw: &serde_json::Value) -> Gemma3WeightLayout {
        if raw.get("text_config").is_some() {
            return Gemma3WeightLayout::NestedLanguageModel;
        }
        // Bare text-model checkpoints (Gemma3TextModel, EmbeddingGemma) ship
        // the backbone at the root — keys are `embed_tokens.weight`,
        // `layers.*` etc. with no `model.` prefix. Detect them by the
        // architectures string only — `model_type == "gemma3_text"` is shared
        // with `Gemma3ForCausalLM`, which DOES use the `model.` prefix.
        let is_bare_text = raw
            .get("architectures")
            .and_then(|v| v.as_array())
            .is_some_and(|arr| {
                arr.iter()
                    .filter_map(|a| a.as_str())
                    .any(|s| s == "Gemma3TextModel")
            });
        if is_bare_text {
            Gemma3WeightLayout::BareText
        } else {
            Gemma3WeightLayout::FlatText
        }
    }

    fn parse_gemma_text_config(
        raw: &serde_json::Value,
        description: &str,
    ) -> Result<Gemma3Config, EngineError> {
        if let Some(text_config) = raw.get("text_config") {
            parse_value(text_config.clone(), description)
        } else {
            parse_value(raw.clone(), description)
        }
    }

    pub(crate) struct Gemma3ModelBuildContext<'a> {
        pub main_vb: VarBuilder<'a>,
        pub embedding: Option<&'a EmbeddingSemantics>,
        pub auxiliary: &'a [AuxiliaryVarBuilder],
    }

    impl<'a> Gemma3ModelBuildContext<'a> {
        fn auxiliary_vb(&self, module_path: &str) -> Option<VarBuilder<'static>> {
            self.auxiliary
                .iter()
                .find(|aux| aux.module_path == module_path)
                .map(|aux| aux.vb.clone())
        }
    }

    fn embedding_semantics_or_default(ctx: &Gemma3ModelBuildContext<'_>) -> EmbeddingSemantics {
        ctx.embedding.cloned().unwrap_or_default()
    }

    pub(crate) fn build_gemma3_model_with_context(
        arch_config: &dyn std::any::Any,
        ctx: &Gemma3ModelBuildContext<'_>,
    ) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
        let cfg = arch_config
            .downcast_ref::<Gemma3ArchConfig>()
            .ok_or_else(|| EngineError::Internal("unexpected arch config type for Gemma3".into()))?;

        let backbone_vb = |layout| match layout {
            Gemma3WeightLayout::FlatText => ctx.main_vb.clone().pp("model"),
            Gemma3WeightLayout::NestedLanguageModel => ctx
                .main_vb
                .clone()
                .pp("model")
                .pp("language_model")
                .pp("model"),
            Gemma3WeightLayout::BareText => ctx.main_vb.clone(),
        };

        match cfg {
            Gemma3ArchConfig::Dense { cfg, layout } => Ok(Box::new(
                Gemma3ForCausalLM::new_with_parts(cfg, backbone_vb(*layout), ctx.main_vb.clone())
                    .map_err(candle_model_err)?,
            )),
            Gemma3ArchConfig::Classifier { cfg, layout } => Ok(Box::new(
                Gemma3ForSequenceClassification::new_with_parts(
                    cfg,
                    backbone_vb(*layout),
                    ctx.main_vb.clone(),
                )
                .map_err(candle_model_err)?,
            )),
            Gemma3ArchConfig::Embedding { cfg, layout } => {
                let semantics = embedding_semantics_or_default(ctx);
                let mut dense_layers = Vec::with_capacity(semantics.dense_layers.len());
                for dense in &semantics.dense_layers {
                    let vb = ctx.auxiliary_vb(&dense.module_path).ok_or_else(|| {
                        EngineError::Internal(format!(
                            "missing embedding weights for Gemma3 dense module {}",
                            dense.module_path
                        ))
                    })?;
                    dense_layers
                        .push(Gemma3EmbeddingDenseLayer::new(dense, vb).map_err(candle_model_err)?);
                }
                Ok(Box::new(
                    Gemma3ForEmbedding::new(cfg, backbone_vb(*layout), &semantics, &dense_layers)
                        .map_err(candle_model_err)?,
                ))
            }
        }
    }

    pub(crate) struct Gemma3ArchSpec;

    pub(crate) static GEMMA3_ARCH_SPEC: Gemma3ArchSpec = Gemma3ArchSpec;
    inventory::submit!(crate::models::registry::ArchSpecEntry::new(&GEMMA3_ARCH_SPEC));

    impl ArchSpec for Gemma3ArchSpec {
        fn name(&self) -> &'static str {
            "gemma3"
        }

        fn architecture_aliases(&self) -> &'static [&'static str] {
            ARCHITECTURE_ALIASES
        }

        fn model_type_aliases(&self) -> &'static [&'static str] {
            MODEL_TYPE_ALIASES
        }

        fn supported_tasks(&self) -> &'static [TaskKind] {
            SUPPORTED_TASKS
        }

        fn parse_config(
            &self,
            task: TaskKind,
            raw: &serde_json::Value,
            _content: &str,
        ) -> Result<ParsedModelConfig, EngineError> {
            let layout = infer_weight_layout(raw);
            match task {
                TaskKind::Generate => {
                    let cfg = parse_gemma_text_config(raw, "Gemma3 config")?;
                    let common = common_from_gemma3(&cfg);
                    Ok(ParsedModelConfig {
                        common,
                        deltanet: None,
                        arch_config: Box::new(Gemma3ArchConfig::Dense { cfg, layout }),
                    })
                }
                TaskKind::Classify => {
                    let json = inject_num_labels_if_missing(raw);
                    let base = parse_gemma_text_config(&json, "Gemma3 classifier text config")?;
                    let num_labels = json
                        .get("num_labels")
                        .and_then(|value| value.as_u64())
                        .ok_or_else(|| {
                            EngineError::InvalidRequest(
                                "Gemma3 classifier config is missing `num_labels`".into(),
                            )
                        })? as usize;
                    let cfg = Gemma3ClassifierConfig {
                        base,
                        num_labels,
                        label2id: json
                            .get("label2id")
                            .and_then(|value| serde_json::from_value(value.clone()).ok()),
                        id2label: json
                            .get("id2label")
                            .and_then(|value| serde_json::from_value(value.clone()).ok()),
                    };
                    let common = common_from_gemma3(&cfg.base);
                    Ok(ParsedModelConfig {
                        common,
                        deltanet: None,
                        arch_config: Box::new(Gemma3ArchConfig::Classifier { cfg, layout }),
                    })
                }
                TaskKind::Embed => {
                    let cfg = parse_gemma_text_config(raw, "Gemma3 embedding config")?;
                    let common = common_from_gemma3(&cfg);
                    Ok(ParsedModelConfig {
                        common,
                        deltanet: None,
                        arch_config: Box::new(Gemma3ArchConfig::Embedding { cfg, layout }),
                    })
                }
            }
        }

        fn build_model(
            &self,
            arch_config: &dyn std::any::Any,
            vb: VarBuilder<'_>,
        ) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
            let ctx = Gemma3ModelBuildContext {
                main_vb: vb,
                embedding: None,
                auxiliary: &[],
            };
            build_gemma3_model_with_context(arch_config, &ctx)
        }

        fn runtime_caps(
            &self,
            task: TaskKind,
            backend: WeightsBackend,
            device: &crate::tensor::Device,
        ) -> RuntimeCaps {
            let is_safetensors = backend == WeightsBackend::Safetensors;
            let is_generate = task == TaskKind::Generate;

            RuntimeCaps {
                supports_kv_cache: is_safetensors && is_generate,
                supports_prefix_cache: false,
                supports_paged_attn: false,
                supports_varlen: device.is_cuda() && is_safetensors,
                supports_deltanet: false,
                supports_cuda_graph: false,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::embedding_semantics_or_default;
        use crate::engine::{
            EmbeddingDenseLayerSpec, EmbeddingNormalization, EmbeddingPooling, EmbeddingSemantics,
        };
        use super::Gemma3ModelBuildContext;
        use crate::loading::var_builder::VarBuilder;

        #[test]
        fn gemma3_embedding_defaults_to_last_token_without_modules_metadata() {
            let ctx = Gemma3ModelBuildContext {
                main_vb: VarBuilder::zeros(crate::tensor::DType::F32, &crate::tensor::Device::Cpu),
                embedding: None,
                auxiliary: &[],
            };

            let semantics = embedding_semantics_or_default(&ctx);

            assert_eq!(semantics.pooling, EmbeddingPooling::LastToken);
            assert_eq!(semantics.normalization, EmbeddingNormalization::None);
            assert!(semantics.dense_layers.is_empty());
        }

        #[test]
        fn gemma3_embedding_uses_provided_semantics_when_available() {
            let provided = EmbeddingSemantics {
                pooling: EmbeddingPooling::Mean,
                normalization: EmbeddingNormalization::L2,
                dense_layers: vec![EmbeddingDenseLayerSpec {
                    module_path: "2_Dense".into(),
                    in_features: 10,
                    out_features: 12,
                    bias: true,
                    activation: Default::default(),
                }],
            };
            let ctx = Gemma3ModelBuildContext {
                main_vb: VarBuilder::zeros(crate::tensor::DType::F32, &crate::tensor::Device::Cpu),
                embedding: Some(&provided),
                auxiliary: &[],
            };

            let semantics = embedding_semantics_or_default(&ctx);

            assert_eq!(semantics, provided);
        }
    }
}
