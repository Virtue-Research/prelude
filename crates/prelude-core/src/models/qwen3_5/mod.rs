//! Qwen3.5: Hybrid attention (Gated DeltaNet + Gated Attention) with dense MLP.
//!
//! Architecture:
//! - Layers where (i+1) % full_attention_interval == 0: standard gated softmax attention
//! - All other layers: Gated DeltaNet (linear attention with delta rule recurrence)
//! - Every layer has a standard dense MLP (gate/up/down with SiLU)
//!
//! Dense variants: 0.8B, 2B, 4B, 9B, 27B.
//!
//! Portions of this implementation are derived from:
//! - SGLang: <https://github.com/sgl-project/sglang/blob/78ddf05a/python/sglang/srt/models/qwen3_5.py>
//! - HuggingFace `modeling_qwen3_5.py`
//! SGLang is licensed under the Apache License, Version 2.0.

pub mod gguf;
pub(crate) mod meta;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use crate::nn_ops::{CandleLinear, Embedding};
use crate::loading::var_builder::VarBuilder;

use crate::models::common::varlen_attention;

use crate::models::common::{
    fast_rms_norm, last_token_select, BatchAttnContext,
    LayerAttnContext, Linear, RmsNorm,
};
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

        Ok(Qwen3_5Config {
            vocab_size: resolve_or_warn!(r.vocab_size, 248320, "vocab_size", MODEL),
            hidden_size: resolve_or_warn!(r.hidden_size, 2048, "hidden_size", MODEL),
            intermediate_size: resolve_or_warn!(r.intermediate_size, 6144, "intermediate_size", MODEL),
            num_hidden_layers: resolve_or_warn!(r.num_hidden_layers, 24, "num_hidden_layers", MODEL),
            num_attention_heads: resolve_or_warn!(r.num_attention_heads, 16, "num_attention_heads", MODEL),
            num_key_value_heads: resolve_or_warn!(r.num_key_value_heads, 2, "num_key_value_heads", MODEL),
            head_dim: resolve_or_warn!(r.head_dim, 256, "head_dim", MODEL),
            max_position_embeddings: resolve_or_warn!(r.max_position_embeddings, 262144, "max_position_embeddings", MODEL),
            rms_norm_eps: resolve_or_warn!(r.rms_norm_eps, 1e-6, "rms_norm_eps", MODEL),
            rope_theta: resolve_or_warn!(rope_theta, 10_000_000.0, "rope_theta", MODEL),
            partial_rotary_factor: resolve_or_warn!(r.partial_rotary_factor, 0.25, "partial_rotary_factor", MODEL),
            full_attention_interval: resolve_or_warn!(r.full_attention_interval, 4, "full_attention_interval", MODEL),
            attn_output_gate: resolve_or_warn!(r.attn_output_gate, true, "attn_output_gate", MODEL),
            linear_num_key_heads: resolve_or_warn!(r.linear_num_key_heads, 16, "linear_num_key_heads", MODEL),
            linear_num_value_heads: resolve_or_warn!(r.linear_num_value_heads, 16, "linear_num_value_heads", MODEL),
            linear_key_head_dim: resolve_or_warn!(r.linear_key_head_dim, 128, "linear_key_head_dim", MODEL),
            linear_value_head_dim: resolve_or_warn!(r.linear_value_head_dim, 128, "linear_value_head_dim", MODEL),
            linear_conv_kernel_dim: resolve_or_warn!(r.linear_conv_kernel_dim, 4, "linear_conv_kernel_dim", MODEL),
            tie_word_embeddings: r.tie_word_embeddings,
            num_experts: r.num_experts,
            num_experts_per_tok: r.num_experts_per_tok,
            moe_intermediate_size: r.moe_intermediate_size,
            shared_expert_intermediate_size: r.shared_expert_intermediate_size,
            norm_topk_prob: resolve_or_warn!(r.norm_topk_prob, true, "norm_topk_prob", MODEL),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerType {
    LinearAttention,
    FullAttention,
}

impl Qwen3_5Config {
    fn layer_type(&self, idx: usize) -> LayerType {
        if (idx + 1) % self.full_attention_interval == 0 {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        }
    }

    fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Convolution dimension: Q + K + V flattened (Z is separate, not convolved).
    fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }

    fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    fn is_moe(&self) -> bool {
        self.num_experts.is_some()
    }
}

// ── RoPE with partial rotary factor ─────────────────────────────────────

struct PartialRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl PartialRotaryEmbedding {
    fn new(cfg: &Qwen3_5Config, dtype: DType, device: &Device) -> Result<Self> {
        let rotary_dim = cfg.rotary_dim();
        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1.0 / cfg.rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;
        let positions = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self {
            cos,
            sin,
            rotary_dim,
        })
    }

    /// Apply partial RoPE with per-token position_ids for varlen paths.
    /// q, k shape: [total_tokens, num_heads, head_dim]
    fn apply_varlen(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // position_ids: [total_tokens] → index_select cos/sin
        let cos = self.cos.index_select(position_ids, 0)?.unsqueeze(1)?; // [T, 1, rotary_dim/2]
        let sin = self.sin.index_select(position_ids, 0)?.unsqueeze(1)?;

        let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?;
        let q_pass = q.narrow(
            D::Minus1,
            self.rotary_dim,
            q.dim(D::Minus1)? - self.rotary_dim,
        )?;
        let q_rot = apply_rotary_emb(&q_rot, &cos, &sin)?;
        let q = Tensor::cat(&[q_rot, q_pass], D::Minus1)?;

        let k_rot = k.narrow(D::Minus1, 0, self.rotary_dim)?;
        let k_pass = k.narrow(
            D::Minus1,
            self.rotary_dim,
            k.dim(D::Minus1)? - self.rotary_dim,
        )?;
        let k_rot = apply_rotary_emb(&k_rot, &cos, &sin)?;
        let k = Tensor::cat(&[k_rot, k_pass], D::Minus1)?;

        Ok((q, k))
    }
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(
        &[
            (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?,
            (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?,
        ],
        D::Minus1,
    )?;
    Ok(rotated)
}

// ── RMSNormGated ────────────────────────────────────────────────────────

struct RmsNormGated {
    weight: Tensor,
    eps: f64,
    num_heads: usize,
    head_dim: usize,
}

impl RmsNormGated {
    fn new(head_dim: usize, num_heads: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(head_dim, "weight")?;
        Ok(Self {
            weight,
            eps,
            num_heads,
            head_dim,
        })
    }

    /// Apply per-head RMS normalization then gate with SiLU(z).
    /// x and z: [..., num_heads * head_dim], weight: [head_dim] (broadcast over heads).
    fn forward(&self, x: &Tensor, z: &Tensor) -> Result<Tensor> {
        let orig_shape = x.shape().clone();
        let leading: Vec<usize> = orig_shape.dims()[..orig_shape.dims().len() - 1].to_vec();
        let mut new_shape = leading.clone();
        new_shape.push(self.num_heads);
        new_shape.push(self.head_dim);

        // Reshape to [..., num_heads, head_dim] for per-head norm
        let x = x.reshape(new_shape.as_slice())?;
        let z = z.reshape(new_shape.as_slice())?;

        // RMS norm on last dimension (head_dim)
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let normed = normed.to_dtype(x.dtype())?.broadcast_mul(&self.weight)?;
        let gate = crate::nn_ops::Activation::Silu.forward(&z)?;
        let result = normed.broadcast_mul(&gate)?;

        // Reshape back to [..., num_heads * head_dim]
        result.reshape(orig_shape)
    }
}

// ── Gated DeltaNet (Linear Attention) ────────────────────────────────────

struct Qwen3_5GatedDeltaNet {
    // Qwen3.5 uses split projections (not fused like Qwen3-Next)
    in_proj_qkv: Linear, // hidden → key_dim*2 + value_dim
    in_proj_z: Linear,   // hidden → value_dim
    in_proj_b: Linear,   // hidden → num_v_heads
    in_proj_a: Linear,   // hidden → num_v_heads
    conv_weight: Tensor,     // [conv_dim, kernel_size] reshaped for dot product
    dt_bias: Tensor,         // [num_v_heads]
    a_log: Tensor,           // [num_v_heads]
    norm: RmsNormGated,
    out_proj: Linear,
    // State
    conv_state: Option<Tensor>,      // [conv_dim, kernel-1]
    recurrent_state: Option<Tensor>, // [num_v_heads, k_dim, v_dim] in f32
    // Config
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel: usize,
}

impl Qwen3_5GatedDeltaNet {
    fn new(cfg: &Qwen3_5Config, vb: VarBuilder) -> Result<Self> {
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();

        // Split projections: QKV separate from Z, B, A
        let in_proj_qkv = Linear::load(
            vb.pp("in_proj_qkv"),
            cfg.hidden_size,
            key_dim * 2 + value_dim, // Q + K + V
            false,
        )?;
        let in_proj_z = Linear::load(
            vb.pp("in_proj_z"),
            cfg.hidden_size,
            value_dim,
            false,
        )?;
        let in_proj_b = Linear::load(
            vb.pp("in_proj_b"),
            cfg.hidden_size,
            cfg.linear_num_value_heads,
            false,
        )?;
        let in_proj_a = Linear::load(
            vb.pp("in_proj_a"),
            cfg.hidden_size,
            cfg.linear_num_value_heads,
            false,
        )?;

        // Conv1d weight: stored as [conv_dim, 1, kernel_size], reshape to [conv_dim, kernel_size]
        let conv_weight_raw = vb.get((conv_dim, 1, cfg.linear_conv_kernel_dim), "conv1d.weight")?;
        let conv_weight = conv_weight_raw.squeeze(1)?;

        let dt_bias = vb.get(cfg.linear_num_value_heads, "dt_bias")?;
        let a_log = vb.get(cfg.linear_num_value_heads, "A_log")?;

        let norm = RmsNormGated::new(
            cfg.linear_value_head_dim,
            cfg.linear_num_value_heads,
            cfg.rms_norm_eps,
            vb.pp("norm"),
        )?;
        let out_proj = Linear::load(
            vb.pp("out_proj"),
            value_dim,
            cfg.hidden_size,
            false,
        )?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv_weight,
            dt_bias,
            a_log,
            norm,
            out_proj,
            conv_state: None,
            recurrent_state: None,
            num_k_heads: cfg.linear_num_key_heads,
            num_v_heads: cfg.linear_num_value_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel: cfg.linear_conv_kernel_dim,
        })
    }

    fn clear_state(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }

    /// Forward pass for a single token (decode) or a sequence (prefill).
    fn forward(&mut self, x: &Tensor, _offset: usize) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        assert_eq!(b, 1, "Qwen3.5 DeltaNet only supports batch_size=1");

        // Project with split projections
        let qkv = x.apply(&self.in_proj_qkv)?; // [1, L, key_dim*2 + value_dim]
        let z = x.apply(&self.in_proj_z)?; // [1, L, value_dim]
        let b_param = x.apply(&self.in_proj_b)?; // [1, L, num_v_heads]
        let a_param = x.apply(&self.in_proj_a)?; // [1, L, num_v_heads]

        // Split QKV: simple concat layout [Q(key_dim) | K(key_dim) | V(value_dim)]
        let q_cat = qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k_cat = qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v_cat = qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        // QKV goes through conv1d, Z does not
        let qkv_for_conv = Tensor::cat(&[&q_cat, &k_cat, &v_cat], D::Minus1)?; // [B, L, conv_dim]

        // Apply causal conv1d
        let qkv_conv = if seq_len == 1 {
            self.conv1d_decode(&qkv_for_conv.squeeze(0)?.squeeze(0)?)?
                .unsqueeze(0)?
                .unsqueeze(0)?
        } else {
            self.conv1d_prefill(&qkv_for_conv)?
        };

        // Apply SiLU activation after conv
        let qkv_conv = crate::nn_ops::Activation::Silu.forward(&qkv_conv)?;

        // Split into q, k, v
        let q = qkv_conv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = qkv_conv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = qkv_conv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        // Process each timestep
        let device = x.device();
        let mut outputs = Vec::with_capacity(seq_len);

        // Squeeze batch dim: [1, L, dim] -> [L, dim]
        let q = q.get(0)?;
        let k = k.get(0)?;
        let v = v.get(0)?;
        let b_param = b_param.get(0)?;
        let a_param = a_param.get(0)?;

        for t in 0..seq_len {
            let q_t = q.get(t)?.contiguous()?; // [key_dim]
            let k_t = k.get(t)?.contiguous()?; // [key_dim]
            let v_t = v.get(t)?.contiguous()?; // [value_dim]
            let b_t = b_param.get(t)?.contiguous()?; // [num_v_heads]
            let a_t = a_param.get(t)?.contiguous()?; // [num_v_heads]

            let out_t = self.delta_rule_step(&q_t, &k_t, &v_t, &b_t, &a_t, device)?;
            outputs.push(out_t);
        }

        // Stack outputs: [1, L, value_dim]
        let output = Tensor::stack(&outputs, 0)?.unsqueeze(0)?;

        // Reshape z for gated norm: [1, L, value_dim]
        let z = z.contiguous()?;

        // Gated RMSNorm + output projection
        let normed = self.norm.forward(&output, &z)?;
        normed.apply(&self.out_proj)
    }

    /// Single-step delta rule update (batched across all v_heads).
    fn delta_rule_step(
        &mut self,
        q: &Tensor, // [key_dim]
        k: &Tensor, // [key_dim]
        v: &Tensor, // [value_dim]
        b: &Tensor, // [num_v_heads]
        a: &Tensor, // [num_v_heads]
        device: &Device,
    ) -> Result<Tensor> {
        let kv_ratio = self.num_v_heads / self.num_k_heads;

        // L2-normalize q and k per head, scale q by 1/sqrt(head_k_dim)
        let q = q.reshape((self.num_k_heads, self.head_k_dim))?;
        let q = l2_normalize_last_dim(&q)?;
        let k = k.reshape((self.num_k_heads, self.head_k_dim))?;
        let k = l2_normalize_last_dim(&k)?;

        // v: [num_v_heads, head_v_dim]
        let v = v.reshape((self.num_v_heads, self.head_v_dim))?;

        // Gating computation (all in f32 for numerical stability)
        let dt_bias = self.dt_bias.to_dtype(DType::F32)?;
        let a_log = self.a_log.to_dtype(DType::F32)?;
        let a_f32 = a.to_dtype(DType::F32)?;
        let b_f32 = b.to_dtype(DType::F32)?;

        // g = -exp(A_log) * softplus(a + dt_bias) per head
        let neg_a_exp = a_log.exp()?.neg()?;
        let a_plus_dt = (a_f32 + dt_bias)?;
        let softplus_val = softplus(&a_plus_dt)?;
        let g = (neg_a_exp * softplus_val)?; // [num_v_heads]

        // beta = sigmoid(b) per head
        let beta = crate::nn_ops::ops::sigmoid(&b_f32)?; // [num_v_heads]

        // decay = exp(g)
        let decay = g.exp()?; // [num_v_heads]

        // Initialize recurrent state if needed: [num_v_heads, head_k_dim, head_v_dim] in f32
        if self.recurrent_state.is_none() {
            self.recurrent_state = Some(Tensor::zeros(
                (self.num_v_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                device,
            )?);
        }
        let state = self.recurrent_state.as_ref().unwrap();

        // Delta rule: state = state * decay + outer(k, beta * (v - state^T @ k))
        // decay: [num_v_heads] → [num_v_heads, 1, 1]
        let decay_3d = decay.reshape((self.num_v_heads, 1, 1))?;
        let state_decayed = state.broadcast_mul(&decay_3d)?;

        // k: [num_k_heads, head_k_dim] → expand to [num_v_heads, head_k_dim]
        let k_f32 = k.to_dtype(DType::F32)?;
        let k_expanded = if kv_ratio > 1 {
            k_f32
                .unsqueeze(1)?
                .expand((self.num_k_heads, kv_ratio, self.head_k_dim))?
                .reshape((self.num_v_heads, self.head_k_dim))?
        } else {
            k_f32
        };
        let k_col = k_expanded.unsqueeze(D::Minus1)?; // [num_v_heads, head_k_dim, 1]

        // Delta correction: v' = beta * (v - state_decayed^T @ k)
        // state_decayed^T @ k: [num_v_heads, head_v_dim, head_k_dim] @ [num_v_heads, head_k_dim, 1]
        let state_k = state_decayed
            .transpose(1, 2)?
            .matmul(&k_col)?
            .squeeze(D::Minus1)?; // [num_v_heads, head_v_dim]
        let v_f32 = v.to_dtype(DType::F32)?;
        let v_error = (v_f32 - state_k)?; // [num_v_heads, head_v_dim]

        // beta: [num_v_heads] → [num_v_heads, 1]
        let beta_2d = beta.reshape((self.num_v_heads, 1))?;
        let v_prime = v_error.broadcast_mul(&beta_2d)?; // [num_v_heads, head_v_dim]

        // state += outer(k, v_prime)
        let v_row = v_prime.unsqueeze(1)?; // [num_v_heads, 1, head_v_dim]
        let outer = k_col.matmul(&v_row)?;
        let state = (state_decayed + outer)?;

        // output = state^T @ (q * scale): scale = 1/sqrt(head_k_dim)
        let scale = (self.head_k_dim as f64).powf(-0.5);
        let q_f32 = (q.to_dtype(DType::F32)? * scale)?;
        let q_expanded = if kv_ratio > 1 {
            q_f32
                .unsqueeze(1)?
                .expand((self.num_k_heads, kv_ratio, self.head_k_dim))?
                .reshape((self.num_v_heads, self.head_k_dim))?
        } else {
            q_f32
        };
        let q_col = q_expanded.unsqueeze(D::Minus1)?; // [num_v_heads, head_k_dim, 1]
        let out = state.transpose(1, 2)?.matmul(&q_col)?; // [num_v_heads, head_v_dim, 1]
        let out = out.squeeze(D::Minus1)?; // [num_v_heads, head_v_dim]
        let out = out.to_dtype(v.dtype())?;
        let out = out.reshape((self.value_dim,))?;

        self.recurrent_state = Some(state);
        Ok(out)
    }

    /// Causal conv1d for a single token (decode step).
    fn conv1d_decode(&mut self, x: &Tensor) -> Result<Tensor> {
        // x: [conv_dim]
        let device = x.device();
        let dtype = x.dtype();

        if self.conv_state.is_none() {
            self.conv_state = Some(Tensor::zeros(
                (self.conv_dim, self.conv_kernel - 1),
                dtype,
                device,
            )?);
        }
        let state = self.conv_state.as_ref().unwrap();

        // state: [conv_dim, kernel-1] (the kernel-1 most recent inputs).
        // For the dot product we want [conv_dim, kernel]: [old state | current_input].
        let x_col = x.unsqueeze(D::Minus1)?; // [conv_dim, 1]
        let full_window = Tensor::cat(&[state, &x_col], 1)?; // [conv_dim, kernel]
        let out = (full_window * &self.conv_weight)?.sum(D::Minus1)?; // [conv_dim]

        // Shift state left and append new input
        let new_state = if self.conv_kernel > 2 {
            let kept = state.narrow(1, 1, self.conv_kernel - 2)?;
            Tensor::cat(&[kept, x_col], 1)?
        } else {
            x_col
        };
        self.conv_state = Some(new_state);
        Ok(out)
    }

    /// Causal conv1d for a full sequence (prefill).
    fn conv1d_prefill(&mut self, x: &Tensor) -> Result<Tensor> {
        // x: [1, L, conv_dim]
        let (b, seq_len, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        // Transpose to [1, conv_dim, L] for conv
        let x_t = x.transpose(1, 2)?; // [1, conv_dim, L]

        // Left-pad with zeros (or existing conv_state)
        let pad_len = self.conv_kernel - 1;
        let prefix = if let Some(ref state) = self.conv_state {
            state.unsqueeze(0)?
        } else {
            Tensor::zeros((b, self.conv_dim, pad_len), dtype, device)?
        };
        let padded = Tensor::cat(&[prefix, x_t.clone()], 2)?; // [1, conv_dim, pad+L]

        // Manual conv1d: slide window of size kernel over padded
        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let window = padded.narrow(2, t, self.conv_kernel)?; // [1, conv_dim, kernel]
            let out = (window.squeeze(0)? * &self.conv_weight)?.sum(D::Minus1)?; // [conv_dim]
            outputs.push(out);
        }
        let result = Tensor::stack(&outputs, 0)?.unsqueeze(0)?; // [1, L, conv_dim]

        // Save last kernel-1 inputs as conv_state
        let x_t_2d = x_t.squeeze(0)?; // [conv_dim, L]
        if seq_len >= pad_len {
            self.conv_state = Some(x_t_2d.narrow(1, seq_len - pad_len, pad_len)?);
        } else {
            let old = if let Some(ref state) = self.conv_state {
                state.narrow(1, seq_len, pad_len - seq_len)?
            } else {
                Tensor::zeros((self.conv_dim, pad_len - seq_len), dtype, device)?
            };
            self.conv_state = Some(Tensor::cat(&[old, x_t_2d], 1)?);
        }

        Ok(result)
    }
}

fn l2_normalize_last_dim(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let norm = x_f32.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let norm = (norm + 1e-12)?;
    x_f32.broadcast_div(&norm)?.to_dtype(x.dtype())
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    // softplus(x) = log(1 + exp(x))
    let exp_x = x.exp()?;
    let one_plus_exp = (exp_x + 1.0)?;
    one_plus_exp.log()
}

// ── Gated Attention ────────────────────────────────────────────────────

struct Qwen3_5Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    rope: PartialRotaryEmbedding,
    kv_cache: Option<(Tensor, Tensor)>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    softmax_scale: f64,
    attn_output_gate: bool,
}

impl Qwen3_5Attention {
    fn new(cfg: &Qwen3_5Config, rope: PartialRotaryEmbedding, vb: VarBuilder) -> Result<Self> {
        let q_proj_dim = if cfg.attn_output_gate {
            cfg.num_attention_heads * cfg.head_dim * 2 // 2x for gate
        } else {
            cfg.num_attention_heads * cfg.head_dim
        };
        let kv_proj_dim = cfg.num_key_value_heads * cfg.head_dim;

        let q_proj = Linear::load(vb.pp("q_proj"), cfg.hidden_size, q_proj_dim, false)?;
        let k_proj = Linear::load(vb.pp("k_proj"), cfg.hidden_size, kv_proj_dim, false)?;
        let v_proj = Linear::load(vb.pp("v_proj"), cfg.hidden_size, kv_proj_dim, false)?;
        let o_proj = Linear::load(
            vb.pp("o_proj"),
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            false,
        )?;

        // Qwen3.5 uses residual RMSNorm: output = norm(x) * (1 + weight)
        let q_norm_weight = (vb.pp("q_norm").get(cfg.head_dim, "weight")? + 1.0)?;
        let q_norm = RmsNorm::from_weight(q_norm_weight.clone(), cfg.rms_norm_eps);

        let k_norm_weight = (vb.pp("k_norm").get(cfg.head_dim, "weight")? + 1.0)?;
        let k_norm = RmsNorm::from_weight(k_norm_weight.clone(), cfg.rms_norm_eps);



        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            q_norm_weight,
            k_norm_weight,
            rope,
            kv_cache: None,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            softmax_scale: 1.0 / (cfg.head_dim as f64).sqrt(),
            attn_output_gate: cfg.attn_output_gate,
        })
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }

    /// Flash-attn-v3 varlen forward for GPU prefill.
    fn forward(&mut self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
        let total_tokens = x.dim(0)?;

        // Project
        let q_raw = x.apply(&self.q_proj)?;
        let k = x.apply(&self.k_proj)?;
        let v = x.apply(&self.v_proj)?;

        // Split Q and gate
        let (q, gate) = if self.attn_output_gate {
            let q_and_gate = q_raw.reshape((total_tokens, self.num_heads, self.head_dim * 2))?;
            let q = q_and_gate.narrow(D::Minus1, 0, self.head_dim)?;
            let gate = q_and_gate
                .narrow(D::Minus1, self.head_dim, self.head_dim)?
                .reshape((total_tokens, self.num_heads * self.head_dim))?;
            (q, Some(gate))
        } else {
            let q = q_raw.reshape((total_tokens, self.num_heads, self.head_dim))?;
            (q, None)
        };

        let k = k.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;

        // Per-head QK normalization
        let q = fast_rms_norm(
            &q.reshape((total_tokens * self.num_heads, self.head_dim))?,
            &self.q_norm,
            &self.q_norm_weight,
            self.rms_norm_eps,
        )?
        .reshape((total_tokens, self.num_heads, self.head_dim))?;
        let k = fast_rms_norm(
            &k.reshape((total_tokens * self.num_kv_heads, self.head_dim))?,
            &self.k_norm,
            &self.k_norm_weight,
            self.rms_norm_eps,
        )?
        .reshape((total_tokens, self.num_kv_heads, self.head_dim))?;

        // Partial RoPE
        let (q, k) = self.rope.apply_varlen(&q, &k, ctx.position_ids)?;

        // Unified varlen attention (handles both plain and paged paths)
        let softmax_scale = self.softmax_scale as f32;
        let (cu_seqlens_k, max_seqlen_k) = match ctx.paged_kv {
            Some(kv) => (kv.cu_seqlens_k, kv.max_seqlen_k),
            None => (ctx.cu_seqlens_q, ctx.max_seqlen_q),
        };
        let attn_output = varlen_attention(
            &q,
            &k,
            &v,
            ctx.cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            ctx.paged_kv,
        )?;
        let attn_output = attn_output.reshape((total_tokens, self.num_heads * self.head_dim))?;
        let gated = if let Some(gate) = gate {
            (attn_output * crate::nn_ops::ops::sigmoid(&gate)?)?
        } else {
            attn_output
        };
        gated.apply(&self.o_proj)
    }
}

// ── Dense MLP ───────────────────────────────────────────────────────────

struct Qwen3_5Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3_5Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::load(vb.pp("gate_proj"), hidden_size, intermediate_size, false)?,
            up_proj: Linear::load(vb.pp("up_proj"), hidden_size, intermediate_size, false)?,
            down_proj: Linear::load(vb.pp("down_proj"), intermediate_size, hidden_size, false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        crate::models::common::fast_silu_mul(&gate, &up)?.apply(&self.down_proj)
    }
}

// ── Sparse MoE Block ────────────────────────────────────────────────────

struct Qwen3_5SparseMoeBlock {
    gate: CandleLinear, // [num_experts, hidden_size]
    // Fused expert weights: gate_up [E, 2*inter, hidden], down [E, hidden, inter]
    experts_gate_up: Tensor,
    experts_down: Tensor,
    moe_intermediate_size: usize,
    shared_expert: Option<Qwen3_5Mlp>,
    shared_expert_gate: Option<CandleLinear>, // [1, hidden_size]
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl Qwen3_5SparseMoeBlock {
    fn new(cfg: &Qwen3_5Config, vb: VarBuilder) -> Result<Self> {
        let num_experts = cfg.num_experts.unwrap();
        let num_experts_per_tok = cfg.num_experts_per_tok.unwrap();
        let moe_intermediate_size = cfg.moe_intermediate_size.unwrap();

        let gate = {
            let gvb = vb.pp("gate");
            let w = gvb.get((num_experts, cfg.hidden_size), "weight")?;
            CandleLinear::new(w, None)
        };

        // Load fused expert weights: [num_experts, 2*inter, hidden] and [num_experts, hidden, inter]
        let vb_experts = vb.pp("experts");
        let experts_gate_up = vb_experts.get(
            (num_experts, 2 * moe_intermediate_size, cfg.hidden_size),
            "gate_up_proj",
        )?;
        let experts_down = vb_experts.get(
            (num_experts, cfg.hidden_size, moe_intermediate_size),
            "down_proj",
        )?;

        let shared_expert = if let Some(shared_size) = cfg.shared_expert_intermediate_size {
            if shared_size > 0 {
                Some(Qwen3_5Mlp::new(
                    cfg.hidden_size,
                    shared_size,
                    vb.pp("shared_expert"),
                )?)
            } else {
                None
            }
        } else {
            None
        };

        let shared_expert_gate = if shared_expert.is_some() {
            Some({
                let gvb = vb.pp("shared_expert_gate");
                let w = gvb.get((1, cfg.hidden_size), "weight")?;
                CandleLinear::new(w, None)
            })
        } else {
            None
        };

        Ok(Self {
            gate,
            experts_gate_up,
            experts_down,
            moe_intermediate_size,
            shared_expert,
            shared_expert_gate,
            num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ndim = xs.dims().len();
        if ndim == 2 {
            return self.forward_2d(xs);
        }
        let (b, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let result = self.forward_2d(&xs_flat)?;
        result.reshape((b, seq_len, hidden_dim))
    }

    fn expert_forward(&self, expert_idx: usize, x: &Tensor) -> Result<Tensor> {
        let gate_up_w = self.experts_gate_up.get(expert_idx)?; // [2*inter, hidden]
        let down_w = self.experts_down.get(expert_idx)?; // [hidden, inter]
        let inter = self.moe_intermediate_size;

        #[cfg(feature = "onednn")]
        if x.device().is_cpu() && x.dtype() == DType::F32 {
            return self.expert_forward_onednn(x, &gate_up_w, &down_w, inter);
        }

        let gate_up = x.matmul(&gate_up_w.t()?)?;
        let gate = gate_up.narrow(D::Minus1, 0, inter)?;
        let up = gate_up.narrow(D::Minus1, inter, inter)?;
        let act = crate::nn_ops::Activation::Silu.forward(&gate)?;
        let hidden = (act * up)?;
        hidden.matmul(&down_w.t()?)
    }

    #[cfg(feature = "onednn")]
    fn expert_forward_onednn(
        &self, x: &Tensor, gate_up_w: &Tensor, down_w: &Tensor, inter: usize,
    ) -> Result<Tensor> {
        crate::ops::onednn::init();
        let x_cont = x.contiguous()?;
        let x_slice = crate::ops::cpu::tensor_as_f32_slice(&x_cont)?;
        let gate_up_cont = gate_up_w.contiguous()?;
        let gate_up_slice = crate::ops::cpu::tensor_as_f32_slice(&gate_up_cont)?;
        let down_cont = down_w.contiguous()?;
        let down_slice = crate::ops::cpu::tensor_as_f32_slice(&down_cont)?;

        let x_dims = x_cont.dims();
        let m = if x_dims.len() == 1 { 1 } else { x_dims[0] };
        let k_in = *x_dims.last().unwrap();
        let n_gate_up = 2 * inter;
        let hidden_out = down_w.dim(0)?;

        let mut gate_up_buf = vec![0.0f32; m * n_gate_up];
        unsafe {
            crate::ops::onednn::ffi::onednn_f32_linear(
                x_slice.as_ptr() as *const _,
                gate_up_slice.as_ptr() as *const _,
                gate_up_buf.as_mut_ptr() as *mut _,
                m as i64, k_in as i64, n_gate_up as i64,
            );
        }

        // SiLU(gate) * up
        let mut hidden_buf = vec![0.0f32; m * inter];
        for row in 0..m {
            let base = row * n_gate_up;
            let dst = row * inter;
            for i in 0..inter {
                let gate_val = gate_up_buf[base + i];
                let silu = gate_val / (1.0 + (-gate_val).exp());
                hidden_buf[dst + i] = silu * gate_up_buf[base + inter + i];
            }
        }

        // hidden @ down_w^T: [m, inter] @ [hidden_out, inter]^T → [m, hidden_out]
        let mut out_buf = vec![0.0f32; m * hidden_out];
        unsafe {
            crate::ops::onednn::ffi::onednn_f32_linear(
                hidden_buf.as_ptr() as *const _,
                down_slice.as_ptr() as *const _,
                out_buf.as_mut_ptr() as *mut _,
                m as i64, inter as i64, hidden_out as i64,
            );
        }

        let out_shape: Vec<usize> = if x_dims.len() == 1 {
            vec![hidden_out]
        } else {
            let mut s = x_dims[..x_dims.len() - 1].to_vec();
            s.push(hidden_out);
            s
        };
        Tensor::from_vec(out_buf, out_shape.as_slice(), x.device())
    }

    fn forward_2d(&self, xs: &Tensor) -> Result<Tensor> {
        let (_n_tokens, hidden_dim) = xs.dims2()?;

        // Router: softmax → topk → gather weights
        let router_logits = xs.apply(&self.gate)?;
        let routing_weights = crate::nn_ops::ops::softmax_last_dim(&router_logits)?;

        let experts_per_tok = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let mut topk_weights = routing_weights
            .gather(&experts_per_tok, D::Minus1)?
            .to_dtype(DType::F32)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        // Sequential expert dispatch (CPU path)
        let topk_weights_vec: Vec<Vec<f32>> = topk_weights.to_vec2()?;
        let experts_per_tok_vec: Vec<Vec<u32>> = experts_per_tok.to_vec2()?;

        let n_tokens = topk_weights_vec.len();
        let mut routed_out = Tensor::zeros((n_tokens, hidden_dim), xs.dtype(), xs.device())?;

        for t in 0..n_tokens {
            let x_t = xs.get(t)?.unsqueeze(0)?; // [1, hidden]
            let mut acc = Tensor::zeros((1, hidden_dim), DType::F32, xs.device())?;
            for k in 0..self.num_experts_per_tok {
                let expert_idx = experts_per_tok_vec[t][k] as usize;
                let weight = topk_weights_vec[t][k];
                let expert_out = self
                    .expert_forward(expert_idx, &x_t)?
                    .to_dtype(DType::F32)?;
                acc = (acc + expert_out * weight as f64)?;
            }
            let acc = acc.to_dtype(xs.dtype())?;
            routed_out = routed_out.slice_assign(&[t..t + 1, 0..hidden_dim], &acc)?;
        }

        // Shared expert
        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(xs)?;
            let shared_out = if let Some(ref gate) = self.shared_expert_gate {
                let gate_val = crate::nn_ops::ops::sigmoid(&xs.apply(gate)?)?; // [n, 1]
                shared_out.broadcast_mul(&gate_val)?
            } else {
                shared_out
            };
            routed_out = (routed_out + shared_out)?;
        }

        Ok(routed_out)
    }
}

enum MlpVariant {
    Dense(Qwen3_5Mlp),
    Sparse(Qwen3_5SparseMoeBlock),
}

impl MlpVariant {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            MlpVariant::Dense(mlp) => mlp.forward(x),
            MlpVariant::Sparse(moe) => moe.forward(x),
        }
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────

enum TokenMixer {
    LinearAttention(Qwen3_5GatedDeltaNet),
    FullAttention(Qwen3_5Attention),
}

struct Qwen3_5DecoderLayer {
    input_layernorm: RmsNorm,
    token_mixer: TokenMixer,
    post_attention_layernorm: RmsNorm,
    mlp: MlpVariant,
    input_ln_weight: Tensor,
    post_attn_ln_weight: Tensor,
    rms_norm_eps: f64,
}

/// Free function to run DeltaNet on packed varlen input (avoids borrow conflicts).
fn deltanet_varlen(
    gdn: &mut Qwen3_5GatedDeltaNet,
    packed: &Tensor,
    seq_lens: &[usize],
) -> Result<Tensor> {
    let mut outputs = Vec::new();
    let mut offset = 0usize;
    for &len in seq_lens {
        let seq = packed.narrow(0, offset, len)?.unsqueeze(0)?; // [1, L, D]
        let out = gdn.forward(&seq, 0)?; // [1, L, D]
        outputs.push(out.squeeze(0)?); // [L, D]
        offset += len;
    }
    Tensor::cat(&outputs, 0) // [total_tokens, D]
}

/// Pooled varlen DeltaNet: process each sequence independently, scatter final state to pool.
/// Used during prefill to initialize per-request state in the pool.
fn deltanet_varlen_pooled(
    gdn: &mut Qwen3_5GatedDeltaNet,
    packed: &Tensor,
    seq_lens: &[usize],
    pool: &mut crate::deltanet_pool::DeltaNetPool,
    slot_ids: &[u32],
    dn_layer_idx: usize,
) -> Result<Tensor> {
    let mut outputs = Vec::new();
    let mut offset = 0usize;
    for (i, &len) in seq_lens.iter().enumerate() {
        gdn.clear_state();
        let seq = packed.narrow(0, offset, len)?.unsqueeze(0)?; // [1, L, D]
        let out = gdn.forward(&seq, 0)?; // [1, L, D]
        outputs.push(out.squeeze(0)?); // [L, D]

        // Scatter final state to pool at this request's slot
        if let Some(ref state) = gdn.recurrent_state {
            pool.recurrent_states[dn_layer_idx].slice_set(
                &state.unsqueeze(0)?,
                0,
                slot_ids[i] as usize,
            )?;
        }
        if let Some(ref state) = gdn.conv_state {
            pool.conv_states[dn_layer_idx].slice_set(
                &state.unsqueeze(0)?,
                0,
                slot_ids[i] as usize,
            )?;
        }
        offset += len;
    }
    gdn.clear_state(); // State is now in the pool
    Tensor::cat(&outputs, 0) // [total_tokens, D]
}

impl Qwen3_5DecoderLayer {
    fn new(
        cfg: &Qwen3_5Config,
        layer_idx: usize,
        rope: PartialRotaryEmbedding,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Qwen3.5 uses residual RMSNorm: output = norm(x) * (1 + weight)
        let input_ln_weight = (vb.pp("input_layernorm").get(cfg.hidden_size, "weight")? + 1.0)?;
        let input_layernorm = RmsNorm::from_weight(input_ln_weight.clone(), cfg.rms_norm_eps);
        let post_attn_ln_weight = (vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")?
            + 1.0)?;
        let post_attention_layernorm =
            RmsNorm::from_weight(post_attn_ln_weight.clone(), cfg.rms_norm_eps);

        let token_mixer = match cfg.layer_type(layer_idx) {
            LayerType::LinearAttention => {
                TokenMixer::LinearAttention(Qwen3_5GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?)
            }
            LayerType::FullAttention => {
                TokenMixer::FullAttention(Qwen3_5Attention::new(cfg, rope, vb.pp("self_attn"))?)
            }
        };

        let mlp = if cfg.is_moe() {
            MlpVariant::Sparse(Qwen3_5SparseMoeBlock::new(cfg, vb.pp("mlp"))?)
        } else {
            MlpVariant::Dense(Qwen3_5Mlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
            )?)
        };

        Ok(Self {
            input_layernorm,
            token_mixer,
            post_attention_layernorm,
            mlp,
            input_ln_weight,
            post_attn_ln_weight,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    /// Flash-attn varlen forward for GPU. DeltaNet layers fall through to standard forward.
    fn forward(
        &mut self,
        x: &Tensor,
        ctx: &LayerAttnContext,
        seq_lens: &[usize],
    ) -> Result<Tensor> {
        let h = fast_rms_norm(
            x,
            &self.input_layernorm,
            &self.input_ln_weight,
            self.rms_norm_eps,
        )?;
        let h = match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.forward(&h, ctx)?,
            TokenMixer::LinearAttention(gdn) => deltanet_varlen(gdn, &h, seq_lens)?,
        };
        let x = (x + h)?;
        let h2 = fast_rms_norm(
            &x,
            &self.post_attention_layernorm,
            &self.post_attn_ln_weight,
            self.rms_norm_eps,
        )?;
        let h2 = self.mlp.forward(&h2)?;
        (x + h2).map(|t| t)
    }

    /// Varlen prefill for DeltaNet layers using pool — scatters state per-sequence.
    fn forward_with_paged_prefix_pooled(
        &mut self,
        x: &Tensor,
        _cu_seqlens_q: &Tensor,
        _cu_seqlens_k: &Tensor,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        _position_ids: &Tensor,
        seq_lens: &[usize],
        pool: &mut crate::deltanet_pool::DeltaNetPool,
        slot_ids: &[u32],
        dn_layer_idx: usize,
    ) -> Result<Tensor> {
        let h = fast_rms_norm(
            x,
            &self.input_layernorm,
            &self.input_ln_weight,
            self.rms_norm_eps,
        )?;
        let h = match &mut self.token_mixer {
            TokenMixer::LinearAttention(gdn) => {
                deltanet_varlen_pooled(gdn, &h, seq_lens, pool, slot_ids, dn_layer_idx)?
            }
            TokenMixer::FullAttention(_) => {
                candle_core::bail!("forward_with_paged_prefix_pooled called on FullAttention layer")
            }
        };
        let x = (x + h)?;
        let h2 = fast_rms_norm(
            &x,
            &self.post_attention_layernorm,
            &self.post_attn_ln_weight,
            self.rms_norm_eps,
        )?;
        let h2 = self.mlp.forward(&h2)?;
        (x + h2).map(|t| t)
    }

    fn clear_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::LinearAttention(gdn) => gdn.clear_state(),
            TokenMixer::FullAttention(attn) => attn.clear_cache(),
        }
    }
}

// ── Model ───────────────────────────────────────────────────────────────

struct Qwen3_5Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3_5DecoderLayer>,
    norm: RmsNorm,
    norm_weight: Tensor,
    rms_norm_eps: f64,
}

impl Qwen3_5Model {
    fn new(cfg: &Qwen3_5Config, vb: VarBuilder) -> Result<Self> {
        // VL models use "model.language_model" prefix, text-only use "model"
        let vb_m = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model.language_model")
        } else {
            vb.pp("model")
        };
        let embed_tokens = {
            let emb_vb = vb_m.pp("embed_tokens");
            let weight = emb_vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            Embedding::new(weight, cfg.hidden_size)
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for idx in 0..cfg.num_hidden_layers {
            let rope = PartialRotaryEmbedding::new(cfg, vb.dtype(), vb.device())?;
            layers.push(Qwen3_5DecoderLayer::new(cfg, idx, rope, vb_l.pp(idx))?);
        }

        // Qwen3.5 uses residual RMSNorm: output = norm(x) * (1 + weight)
        let norm_weight = (vb_m.pp("norm").get(cfg.hidden_size, "weight")? + 1.0)?;
        let norm = RmsNorm::from_weight(norm_weight.clone(), cfg.rms_norm_eps);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            norm_weight,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let mut h = self.embed_tokens.forward(packed_input)?;
        let seq_lens = ctx.seq_lens;
        if let Some(paged) = ctx.paged_kv {
            let mut attn_layer_idx = 0usize;
            let mut dn_layer_idx = 0usize;
            for layer in self.layers.iter_mut() {
                match &layer.token_mixer {
                    TokenMixer::FullAttention(_) => {
                        let layer_kv = paged.layer(attn_layer_idx);
                        let layer_ctx = LayerAttnContext {
                            cu_seqlens_q: ctx.cu_seqlens_q,
                            max_seqlen_q: ctx.max_seqlen_q,
                            position_ids: ctx.position_ids,
                            paged_kv: Some(&layer_kv),
                        };
                        h = layer.forward(&h, &layer_ctx, seq_lens)?;
                        attn_layer_idx += 1;
                    }
                    TokenMixer::LinearAttention(_) => {
                        if let (Some(pool), Some(slots)) =
                            (ctx.deltanet_pool.as_deref_mut(), ctx.deltanet_slots)
                        {
                            h = layer.forward_with_paged_prefix_pooled(
                                &h,
                                ctx.cu_seqlens_q,
                                &paged.cu_seqlens_k,
                                ctx.max_seqlen_q,
                                paged.max_seqlen_k,
                                ctx.position_ids,
                                seq_lens,
                                pool,
                                slots,
                                dn_layer_idx,
                            )?;
                        } else {
                            let layer_ctx = LayerAttnContext {
                                cu_seqlens_q: ctx.cu_seqlens_q,
                                max_seqlen_q: ctx.max_seqlen_q,
                                position_ids: ctx.position_ids,
                                paged_kv: None,
                            };
                            h = layer.forward(&h, &layer_ctx, seq_lens)?;
                        }
                        dn_layer_idx += 1;
                    }
                }
            }
        } else {
            let layer_ctx = LayerAttnContext {
                cu_seqlens_q: ctx.cu_seqlens_q,
                max_seqlen_q: ctx.max_seqlen_q,
                position_ids: ctx.position_ids,
                paged_kv: None,
            };
            for layer in self.layers.iter_mut() {
                h = layer.forward(&h, &layer_ctx, seq_lens)?;
            }
        }
        fast_rms_norm(&h, &self.norm, &self.norm_weight, self.rms_norm_eps)
    }

    fn clear_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }
}

// ── ForCausalLM ─────────────────────────────────────────────────────────

pub struct Qwen3_5ForCausalLM {
    model: Qwen3_5Model,
    lm_head: Linear,
}

impl Qwen3_5ForCausalLM {
    pub fn new(cfg: &Qwen3_5Config, vb: VarBuilder) -> Result<Self> {
        let model = Qwen3_5Model::new(cfg, vb.clone())?;
        // VL models use "model.language_model" prefix
        let model_prefix = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            "model.language_model"
        } else {
            "model"
        };
        let lm_head = if cfg.tie_word_embeddings {
            let w = vb
                .pp(model_prefix)
                .pp("embed_tokens")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            Linear::from_weight(w, None)?
        } else {
            Linear::load(vb.pp("lm_head"), cfg.hidden_size, cfg.vocab_size, false)?
        };
        Ok(Self { model, lm_head })
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_cache();
    }

    pub fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let hidden = self.model.forward(packed_input, ctx)?;
        last_token_select(&hidden, ctx.seq_lens)?
            .unsqueeze(1)?
            .apply(&self.lm_head)
    }
}

impl crate::models::ModelForward for Qwen3_5ForCausalLM {
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> candle_core::Result<Tensor> {
        self.forward(packed_input, ctx)
    }

    fn forward_hidden_states(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> candle_core::Result<Tensor> {
        self.model.forward(packed_input, ctx)
    }

    fn compute_logits(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        hidden.apply(&self.lm_head)
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache();
    }
}
