pub(crate) mod meta;

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use crate::nn_ops::{Activation, Embedding};
use crate::loading::var_builder::VarBuilder;
use serde::Deserialize;

use crate::models::common::{
    Linear, RmsNorm,
    varlen_attention, varlen_attention_bidirectional, varlen_attention_windowed,
};

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
        let freqs = t.matmul(&inv_freq)?;
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
        let q_embed = crate::nn_ops::rotary_emb::rope_thd(&q4, &cos, &sin)?;
        let k_embed = crate::nn_ops::rotary_emb::rope_thd(&k4, &cos, &sin)?;
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        (gate.apply(&self.act_fn)? * up)?.apply(&self.down_proj)
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
        packed_input: &Tensor,
        cu_seqlens: &Tensor,
        max_seqlen: usize,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (total_tokens, _) = packed_input.dims2()?;

        let q = self.q_proj.forward(packed_input)?;
        let k = self.k_proj.forward(packed_input)?;
        let v = self.v_proj.forward(packed_input)?;

        let q = q.reshape((total_tokens, self.num_heads, self.head_dim))?;
        let k = k.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (q, k) = self.rotary_emb.apply_varlen(&q, &k, position_ids)?;

        let attn_out = match self.attention_mode {
            Gemma3AttentionMode::FullCausal => varlen_attention(
                &q, &k, &v,
                cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                self.softmax_scale, None,
            )?,
            Gemma3AttentionMode::FullBidirectional => varlen_attention_bidirectional(
                &q, &k, &v,
                cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                self.softmax_scale,
            )?,
            Gemma3AttentionMode::SlidingCausal { window_size } => varlen_attention_windowed(
                &q, &k, &v,
                cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                self.softmax_scale,
                Some(window_size.saturating_sub(1)),
                Some(0),
            )?,
            Gemma3AttentionMode::SlidingBidirectional { window_size } => varlen_attention_windowed(
                &q, &k, &v,
                cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                self.softmax_scale,
                Some(window_size.saturating_sub(1)),
                Some(window_size.saturating_sub(1)),
            )?,
        };

        let attn_dim = self.num_heads * self.head_dim;
        attn_out.reshape((total_tokens, attn_dim))?.apply(&self.o_proj)
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
        xs: &Tensor,
        cu_seqlens: &Tensor,
        max_seqlen: usize,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(xs)?;
        let attn_output =
            self.self_attn
                .forward(&normed, cu_seqlens, max_seqlen, position_ids)?;

        let post_attn_normed = self.post_attention_layernorm.forward(&attn_output)?;
        let xs = crate::models::common::fast_add(&post_attn_normed, xs)?;

        let pre_ffn_normed = self.pre_feedforward_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&pre_ffn_normed)?;

        let post_ffn_normed = self.post_feedforward_layernorm.forward(&mlp_output)?;
        crate::models::common::fast_add(&post_ffn_normed, &xs)
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
        packed_input: &Tensor,
        cu_seqlens: &Tensor,
        max_seqlen: usize,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let embed_scale = (self.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(packed_input)? * embed_scale)?;

        for layer in &mut self.layers {
            xs = layer.forward(&xs, cu_seqlens, max_seqlen, position_ids)?;
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
        ctx: &mut crate::models::common::BatchAttnContext,
    ) -> Result<Tensor> {
        let hidden = self.base.forward(
            packed_input,
            ctx.cu_seqlens_q,
            ctx.max_seqlen_q,
            ctx.position_ids,
        )?;
        let last_hidden =
            crate::models::common::last_token_select(&hidden, ctx.seq_lens)?.contiguous()?;

        let logits = last_hidden.unsqueeze(1)?.apply(&self.lm_head)?;

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
        ctx: &mut crate::models::common::BatchAttnContext,
    ) -> candle_core::Result<Tensor> {
        self.base.forward(
            packed_input,
            ctx.cu_seqlens_q,
            ctx.max_seqlen_q,
            ctx.position_ids,
        )
    }

    fn compute_logits(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let logits = hidden.apply(&self.lm_head)?;
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
        ctx: &mut crate::models::common::BatchAttnContext,
    ) -> candle_core::Result<Tensor> {
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
        ctx: &mut crate::models::common::BatchAttnContext,
    ) -> Result<Tensor> {
        let hidden_states = self.base.forward(
            packed_input,
            ctx.cu_seqlens_q,
            ctx.max_seqlen_q,
            ctx.position_ids,
        )?;
        let last_hidden = crate::models::common::last_token_select(&hidden_states, ctx.seq_lens)?;
        last_hidden.apply(&self.score)
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
        let x = x.apply(&self.linear)?;
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
        ctx: &mut crate::models::common::BatchAttnContext,
    ) -> Result<Tensor> {
        let hidden_states = self.base.forward(
            packed_input,
            ctx.cu_seqlens_q,
            ctx.max_seqlen_q,
            ctx.position_ids,
        )?;
        let hidden_states = hidden_states.to_dtype(DType::F32)?;
        let mut pooled = match self.pooling {
            crate::engine::EmbeddingPooling::LastToken => {
                crate::models::common::last_token_select(&hidden_states, ctx.seq_lens)?
            }
            crate::engine::EmbeddingPooling::Mean => {
                pool_mean_varlen(&hidden_states, ctx.seq_lens)?
            }
            crate::engine::EmbeddingPooling::Cls => {
                crate::models::common::first_token_select(&hidden_states, ctx.seq_lens)?
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
        ctx: &mut crate::models::common::BatchAttnContext,
    ) -> candle_core::Result<Tensor> {
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
        ctx: &mut crate::models::common::BatchAttnContext,
    ) -> candle_core::Result<Tensor> {
        Gemma3ForEmbedding::forward(self, packed_input, ctx)
    }

    fn clear_kv_cache(&mut self) {
        Gemma3ForEmbedding::clear_kv_cache(self);
    }

    fn as_embedding(&self) -> Option<&dyn crate::models::EmbeddingModel> {
        Some(self)
    }
}
