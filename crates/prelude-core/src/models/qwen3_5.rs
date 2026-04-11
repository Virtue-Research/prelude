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


use crate::tensor::{DType, Device, Module, Result, Tensor, D};
use crate::models::commons::embedding::Embedding;
use crate::models::commons::linear::DenseLinear;
use crate::loading::var_builder::VarBuilder;

use crate::models::commons::{
    last_token_select, BatchAttnContext,
    BatchState, LayerAttnContext, Linear, RmsNorm,
};
use crate::ops::{MaskType, VarlenParams, PagedParams};
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
            intermediate_size: resolve_or_warn!(r.intermediate_size, 6144, "intermediate_size", MODEL),
            num_hidden_layers: resolve_or_warn!(r.num_hidden_layers, 24, "num_hidden_layers", MODEL),
            num_attention_heads: resolve_or_warn!(r.num_attention_heads, 16, "num_attention_heads", MODEL),
            num_key_value_heads: resolve_or_warn!(r.num_key_value_heads, 2, "num_key_value_heads", MODEL),
            head_dim: resolve_or_warn!(r.head_dim, 256, "head_dim", MODEL),
            max_position_embeddings: resolve_or_warn!(r.max_position_embeddings, 262144, "max_position_embeddings", MODEL),
            rms_norm_eps: resolve_or_warn!(r.rms_norm_eps, 1e-6, "rms_norm_eps", MODEL),
            rope_theta: resolve_or_warn!(rope_theta, 10_000_000.0, "rope_theta", MODEL),
            partial_rotary_factor: resolve_or_warn!(partial_rotary_factor, 0.25, "partial_rotary_factor", MODEL),
            full_attention_interval: resolve_or_warn!(r.full_attention_interval, 4, "full_attention_interval", MODEL),
            attn_output_gate: resolve_or_warn!(r.attn_output_gate, true, "attn_output_gate", MODEL),
            linear_num_key_heads: resolve_or_warn!(r.linear_num_key_heads, 16, "linear_num_key_heads", MODEL),
            linear_num_value_heads: resolve_or_warn!(r.linear_num_value_heads, 16, "linear_num_value_heads", MODEL),
            linear_key_head_dim: resolve_or_warn!(r.linear_key_head_dim, 128, "linear_key_head_dim", MODEL),
            linear_value_head_dim: resolve_or_warn!(r.linear_value_head_dim, 128, "linear_value_head_dim", MODEL),
            linear_conv_kernel_dim: resolve_or_warn!(r.linear_conv_kernel_dim, 4, "linear_conv_kernel_dim", MODEL),
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

pub(super) struct PartialRotaryEmbedding {
    pub(super) cos: Tensor,
    pub(super) sin: Tensor,
    pub(super) rotary_dim: usize,
}

impl PartialRotaryEmbedding {
    pub(super) fn new(cfg: &Qwen3_5Config, dtype: DType, device: &Device) -> Result<Self> {
        let rotary_dim = cfg.rotary_dim();
        let half = rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1.0 / cfg.rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        // Build the [L, D/2] frequency table via broadcast_mul instead of a K=1
        // matmul (CUTLASS/DeepGEMM only support TN layout; a literal outer-product
        // matmul falls through to an unsupported NN kernel).
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = positions.broadcast_mul(&inv_freq)?;
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

pub(super) fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let part1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let part2 = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;
    Tensor::cat(&[&part1, &part2], D::Minus1)
}

// ── RMSNormGated ────────────────────────────────────────────────────────

pub(super) struct RmsNormGated {
    pub(super) weight: Tensor,
    pub(super) eps: f64,
    pub(super) num_heads: usize,
    pub(super) head_dim: usize,
}

impl RmsNormGated {
    pub(super) fn new(head_dim: usize, num_heads: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        // Qwen3.5-35B-A3B stores this weight as F32 in the checkpoint; compute
        // it in F32 to avoid precision loss in the DeltaNet output scaling.
        let weight = vb.get_with_hints_dtype(
            head_dim,
            "weight",
            Default::default(),
            DType::F32,
        )?;
        Ok(Self {
            weight,
            eps,
            num_heads,
            head_dim,
        })
    }

    /// Apply per-head RMS normalization then gate with SiLU(z).
    /// x and z: [..., num_heads * head_dim], weight: [head_dim] (broadcast over heads).
    fn forward(&self, x: &Tensor, z: &Tensor, ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        let orig_shape = x.shape().clone();
        let leading: Vec<usize> = orig_shape.dims()[..orig_shape.dims().len() - 1].to_vec();
        let mut new_shape = leading.clone();
        new_shape.push(self.num_heads);
        new_shape.push(self.head_dim);

        // Reshape to [..., num_heads, head_dim] for per-head norm
        let x = x.reshape(new_shape.as_slice())?;
        let z = z.reshape(new_shape.as_slice())?;

        // Match HF `Qwen3_5MoeRMSNormGated`: normalize in F32, apply weight in
        // F32, compute SiLU(gate) in F32, do the final gate multiply in F32,
        // and only cast the result back at the very end.
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let normed = normed.broadcast_mul(&self.weight)?; // F32 × F32
        let gate_f32 = z.to_dtype(DType::F32)?;
        let silu_gate = ops.silu(&gate_f32)?;
        let result = normed
            .broadcast_mul(&silu_gate)?
            .to_dtype(x.dtype())?;

        // Reshape back to [..., num_heads * head_dim]
        result.reshape(orig_shape)
    }
}

// ── Gated DeltaNet (Linear Attention) ────────────────────────────────────

pub(super) struct Qwen3_5GatedDeltaNet {
    // Qwen3.5 uses split projections (not fused like Qwen3-Next)
    pub(super) in_proj_qkv: Linear, // hidden → key_dim*2 + value_dim
    pub(super) in_proj_z: Linear,   // hidden → value_dim
    pub(super) in_proj_b: Linear,   // hidden → num_v_heads
    pub(super) in_proj_a: Linear,   // hidden → num_v_heads
    pub(super) conv_weight: Tensor,     // [conv_dim, kernel_size] reshaped for dot product
    pub(super) dt_bias: Tensor,         // [num_v_heads]
    pub(super) a_log: Tensor,           // [num_v_heads]
    pub(super) norm: RmsNormGated,
    pub(super) out_proj: Linear,
    // State
    pub(super) conv_state: Option<Tensor>,      // [conv_dim, kernel-1]
    pub(super) recurrent_state: Option<Tensor>, // [num_v_heads, v_dim, k_dim] in f32
    // Config
    pub(super) num_k_heads: usize,
    pub(super) num_v_heads: usize,
    pub(super) head_k_dim: usize,
    pub(super) head_v_dim: usize,
    pub(super) key_dim: usize,
    pub(super) value_dim: usize,
    pub(super) conv_dim: usize,
    pub(super) conv_kernel: usize,
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
        // `A_log` is stored as F32 in Qwen3.5-35B-A3B checkpoints and is always
        // consumed in F32; load it in native precision to avoid the BF16 truncation.
        let a_log = vb.get_with_hints_dtype(
            cfg.linear_num_value_heads,
            "A_log",
            Default::default(),
            DType::F32,
        )?;

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
    fn forward(&mut self, x: &Tensor, _offset: usize, ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        assert_eq!(b, 1, "Qwen3.5 DeltaNet only supports batch_size=1");
        let bst = BatchState::no_lora();

        // Project with split projections
        let qkv = self.in_proj_qkv.forward(x, &bst, ops)?; // [1, L, key_dim*2 + value_dim]
        let z = self.in_proj_z.forward(x, &bst, ops)?; // [1, L, value_dim]
        let b_param = self.in_proj_b.forward(x, &bst, ops)?; // [1, L, num_v_heads]
        let a_param = self.in_proj_a.forward(x, &bst, ops)?; // [1, L, num_v_heads]

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
        let qkv_conv = ops.silu(&qkv_conv)?;

        // Split into q, k, v
        let q = qkv_conv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = qkv_conv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = qkv_conv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        // Squeeze batch dim: [1, L, dim] -> [L, dim]
        let q = q.get(0)?;
        let k = k.get(0)?;
        let v = v.get(0)?;
        let b_param = b_param.get(0)?;
        let a_param = a_param.get(0)?;

        // Batched delta-rule prefill. For seq_len==1 we still go through this
        // path; the single-step "loop" is cheap and keeps the hot code in one
        // place. The heavy lift (L2 norm, GQA expand, gating/beta, cast-to-F32)
        // is done once over the whole sequence, then the recurrence runs with
        // three real batched matmuls per step instead of broadcast_mul+sum.
        let device = x.device();
        let output = self.delta_rule_prefill(
            &q, &k, &v, &b_param, &a_param, device, ops,
        )?
        .unsqueeze(0)?;

        // Reshape z for gated norm: [1, L, value_dim]
        let z = z.contiguous()?;

        // Gated RMSNorm + output projection
        let normed = self.norm.forward(&output, &z, ops)?;
        self.out_proj.forward(&normed, &bst, ops)
    }

    /// Batched delta-rule prefill for a full sequence.
    ///
    /// Inputs are all 2D (seq_len as dim 0), output is [T, value_dim].
    /// Does the L2 norm / GQA expand / gating / beta in a single batched pass
    /// over all tokens, then runs the recurrence loop with three batched
    /// matmuls per step:
    ///
    ///   state_decayed @ k_col        → delta correction
    ///   v_prime @ k_row^T            → outer update
    ///   state @ q_col                → output
    ///
    /// Replaces an older per-step `broadcast_mul + sum(2)` implementation
    /// that launched ~15 kernels per token and allocated a fresh 2 MB
    /// intermediate each time — catastrophic at 1 K-token prefill × 30
    /// DeltaNet layers.
    fn delta_rule_prefill(
        &mut self,
        q_in: &Tensor, // [T, key_dim]
        k_in: &Tensor, // [T, key_dim]
        v_in: &Tensor, // [T, value_dim]
        b_in: &Tensor, // [T, num_v_heads]
        a_in: &Tensor, // [T, num_v_heads]
        device: &Device,
        ops: &dyn crate::ops::Ops,
    ) -> Result<Tensor> {
        let t = q_in.dim(0)?;
        let hv = self.num_v_heads;
        let hk = self.num_k_heads;
        let kdim = self.head_k_dim;
        let vdim = self.head_v_dim;
        let out_dtype = v_in.dtype();

        // Fast path: FlashInfer SM90 `gdn_prefill` kernel — the one
        // Qwen3.5 / Qwen3-next / FLA's `chunk_gated_delta_rule` were
        // designed for. Scalar-per-head linear-space decay, no chunk
        // cumsum / no safe_gate clamp — bit-exact (within BF16) to the
        // HF reference.
        if device.is_cuda() && kdim == vdim && kdim == 128 {
            if let Some(out) = self.delta_rule_prefill_gdn(
                q_in, k_in, v_in, b_in, a_in, device, ops,
            )? {
                return Ok(out.to_dtype(out_dtype)?);
            }
        }

        // ── Fallback: batched-matmul recurrence on whatever device we have ──

        let kv_ratio = hv / hk;

        // q, k: [T, key_dim] → [T, HK, K] → L2 norm → F32 → (GQA expand) → [T, HV, K]
        let q = q_in.reshape((t, hk, kdim))?;
        let q = l2_normalize_last_dim(&q)?;
        let k = k_in.reshape((t, hk, kdim))?;
        let k = l2_normalize_last_dim(&k)?;

        let expand_heads = |x: &Tensor| -> Result<Tensor> {
            if kv_ratio > 1 {
                x.to_dtype(DType::F32)?
                    .unsqueeze(2)?
                    .expand((t, hk, kv_ratio, kdim))?
                    .reshape((t, hv, kdim))?
                    .contiguous()
            } else {
                x.to_dtype(DType::F32)?.contiguous()
            }
        };
        let k_all = expand_heads(&k)?; // [T, HV, K] f32
        let scale = (kdim as f64).powf(-0.5);
        let q_all = (expand_heads(&q)? * scale)?; // [T, HV, K] f32

        // v: [T, value_dim] → [T, HV, V] f32
        let v_all = v_in
            .reshape((t, hv, vdim))?
            .to_dtype(DType::F32)?
            .contiguous()?;

        // Gating: decay[t, hv] = exp(-exp(a_log[hv]) * softplus(a[t, hv] + dt_bias[hv]))
        // All shapes are [HV] scalar-per-head except `a_in` which is [T, HV].
        let dt_bias_f32 = self.dt_bias.to_dtype(DType::F32)?;
        let a_log_f32 = self.a_log.to_dtype(DType::F32)?;
        let neg_a_exp = a_log_f32.exp()?.neg()?; // [HV]
        let a_f32 = a_in.to_dtype(DType::F32)?; // [T, HV]
        let a_plus_dt = a_f32.broadcast_add(&dt_bias_f32)?; // [T, HV]
        let softplus_val = softplus(&a_plus_dt)?;
        let g = softplus_val.broadcast_mul(&neg_a_exp)?; // [T, HV]
        let decay_all = g.exp()?; // [T, HV]

        // beta = sigmoid(b) — [T, HV] f32
        let beta_all = ops.sigmoid(&b_in.to_dtype(DType::F32)?)?;

        // ── State init ─────────────────────────────────────────────────────
        if self.recurrent_state.is_none() {
            self.recurrent_state = Some(Tensor::zeros((hv, vdim, kdim), DType::F32, device)?);
        }
        let mut state = self.recurrent_state.take().unwrap();

        // ── Sequential recurrence loop ────────────────────────────────────
        //
        // The per-step math is the classic gated delta rule:
        //   state   *= decay
        //   state_k  = state @ k              (a K=1 batched matmul per head)
        //   v'       = beta * (v - state_k)
        //   state   += outer(v', k)
        //   out      = state @ q              (also K=1 batched matmul)
        //
        // We intentionally use `broadcast_mul + sum` instead of `Tensor::matmul`
        // for the two K=1 GEMMs: candle's batched matmul on CUDA routes through
        // the cutlass-gemm wrapper, which only implements the TN transpose
        // combo. A batched matmul with shapes [HV, V, K] @ [HV, K, 1] lands in
        // the NN layout and the wrapper returns `unsupported transpose combo`.
        // The broadcast-and-reduce form stays elementwise and works on both
        // CPU and CUDA.
        let mut outputs = Vec::with_capacity(t);
        for i in 0..t {
            let k_row = k_all.get(i)?.contiguous()?; // [HV, K]
            let v_row = v_all.get(i)?; // [HV, V]
            let q_row = q_all.get(i)?.contiguous()?; // [HV, K]
            let decay = decay_all.get(i)?; // [HV]
            let beta = beta_all.get(i)?; // [HV]

            // state *= decay.unsqueeze(-1).unsqueeze(-1)
            let decay_3d = decay.reshape((hv, 1, 1))?;
            let state_decayed = state.broadcast_mul(&decay_3d)?;

            // state_k[b, v] = Σ_k state_decayed[b, v, k] * k_row[b, k]
            let k_row_1xk = k_row.reshape((hv, 1, kdim))?; // broadcast over V
            let state_k = state_decayed
                .broadcast_mul(&k_row_1xk)?
                .sum(D::Minus1)?; // [HV, V]

            // v_error = v_row - state_k; v_prime = beta * v_error
            let v_err = (v_row - state_k)?;
            let beta_col = beta.unsqueeze(D::Minus1)?; // [HV, 1]
            let v_prime = v_err.broadcast_mul(&beta_col)?; // [HV, V]

            // outer[b, v, k] = v_prime[b, v] * k_row[b, k]  — pure broadcast
            let v_col = v_prime.reshape((hv, vdim, 1))?;
            let outer = v_col.broadcast_mul(&k_row_1xk)?; // [HV, V, K]

            state = (state_decayed + outer)?;

            // out[b, v] = Σ_k state[b, v, k] * q_row[b, k]
            let q_row_1xk = q_row.reshape((hv, 1, kdim))?;
            let out = state
                .broadcast_mul(&q_row_1xk)?
                .sum(D::Minus1)?; // [HV, V]
            outputs.push(out);
        }

        self.recurrent_state = Some(state);

        // Stack outputs: [T, HV, V] → [T, value_dim] → caller dtype
        let out_stacked = Tensor::stack(&outputs, 0)?; // [T, HV, V]
        out_stacked.reshape((t, self.value_dim))?.to_dtype(out_dtype)
    }

    /// FlashInfer-backed GDN prefill fast path for Qwen3.5 DeltaNet.
    ///
    /// This matches HF transformers' `chunk_gated_delta_rule` semantics
    /// bit-for-bit (modulo BF16 rounding): scalar-per-head **linear-space**
    /// decay `alpha = exp(-exp(A_log) * softplus(a + dt_bias))`, no
    /// `RCP_LN2` rescaling, no `safe_gate` clamp, no per-element broadcast.
    /// The kernel's internal delta-rule math is the same one Qwen3.5 was
    /// trained against.
    ///
    /// Returns `Ok(None)` when the backend can't serve the call (non-SM90,
    /// shape mismatch, kernel not compiled); callers fall back.
    fn delta_rule_prefill_gdn(
        &mut self,
        q_in: &Tensor,
        k_in: &Tensor,
        v_in: &Tensor,
        b_in: &Tensor,
        a_in: &Tensor,
        device: &Device,
        ops: &dyn crate::ops::Ops,
    ) -> Result<Option<Tensor>> {
        let t = q_in.dim(0)?;
        let hk = self.num_k_heads;
        let hv = self.num_v_heads;
        let kdim = self.head_k_dim;
        let vdim = self.head_v_dim;
        // FlashInfer uses `num_sab_heads = max(num_q, num_v)` for the
        // state / gate / output head axis. Qwen3.5 is GVA (num_v > num_k),
        // so alpha/beta/output live on the V-head axis.
        debug_assert!(hv >= hk, "Qwen3.5 DeltaNet is GVA: num_v >= num_k");

        // ── Q, K: L2-normalize in BF16 on the Q/K physical head count ────
        // (GVA: hq == hk == num_k_heads = 16 for A3B)
        let q_bf16 = l2_normalize_last_dim(&q_in.reshape((t, hk, kdim))?)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let k_bf16 = l2_normalize_last_dim(&k_in.reshape((t, hk, kdim))?)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let v_bf16 = v_in
            .reshape((t, hv, vdim))?
            .to_dtype(DType::BF16)?
            .contiguous()?;

        // ── Linear-space decay alpha = exp(g_scalar), shape [T, HV] ──────
        // g_scalar = -exp(A_log) * softplus(a + dt_bias). HF's reference
        // computes this in F32 — we do the same.
        let dt_bias_f32 = self.dt_bias.to_dtype(DType::F32)?;
        let a_log_f32 = self.a_log.to_dtype(DType::F32)?;
        let neg_exp_a = a_log_f32.exp()?.neg()?; // [HV]
        let a_f32 = a_in.to_dtype(DType::F32)?;
        let a_plus_dt = a_f32.broadcast_add(&dt_bias_f32)?;
        let g_soft = softplus(&a_plus_dt)?;
        let g_scalar = g_soft.broadcast_mul(&neg_exp_a)?; // [T, HV]
        let alpha = g_scalar.exp()?.contiguous()?; // [T, HV] f32 linear decay

        // ── Beta: sigmoid in F32 over [T, HV] ────────────────────────────
        let beta = ops
            .sigmoid(&b_in.to_dtype(DType::F32)?)?
            .contiguous()?; // [T, HV] f32

        // ── cu_seqlens = [0, T] for a single packed sequence (I64) ───────
        // Note: flashinfer expects I64, unlike cuLA's I32.
        let cu_seqlens = Tensor::from_vec(vec![0i64, t as i64], (2,), device)?;

        // ── Initial state: [1, HV, D, D] f32 or None ─────────────────────
        // Our own recurrent state is stored [HV, V, K]; the kernel wants a
        // leading num_seqs=1 dim. Pull it out of `self` so the fallback
        // can restore it on the backend-declined branch.
        let initial_state = self.recurrent_state.take();
        let initial_state_4d = match initial_state.as_ref() {
            Some(s) => Some(s.unsqueeze(0)?.contiguous()?),
            None => None,
        };

        let scale = (kdim as f32).powf(-0.5);

        // ── Launch ───────────────────────────────────────────────────────
        let Some(result) = ops.gdn_prefill_varlen(
            &q_bf16, &k_bf16, &v_bf16,
            &alpha, &beta, &cu_seqlens,
            initial_state_4d.as_ref(), scale,
        ) else {
            // Backend declined — restore recurrent_state so the composed
            // fallback picks up where we left off.
            self.recurrent_state = initial_state;
            return Ok(None);
        };
        let (out, final_state) = result?;

        // Save the updated recurrent state: [1, HV, D, D] → [HV, D, D].
        // Note: kernel returns `[num_seqs, num_sab_heads, head_dim, head_dim]`
        // which for Qwen3.5 is `[1, HV, D, D]` with `D == head_k_dim == head_v_dim`.
        self.recurrent_state = Some(final_state.squeeze(0)?.contiguous()?);

        // out is [T, HV, D] BF16 (num_sab_heads == hv for GVA). Fold HV*D
        // back into value_dim.
        Ok(Some(out.reshape((t, self.value_dim))?))
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

pub(super) fn l2_normalize_last_dim(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let norm = x_f32.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let norm = (norm + 1e-12)?;
    x_f32.broadcast_div(&norm)?.to_dtype(x.dtype())
}

pub(super) fn softplus(x: &Tensor) -> Result<Tensor> {
    // softplus(x) = log(1 + exp(x))
    let exp_x = x.exp()?;
    let one_plus_exp = (exp_x + 1.0)?;
    one_plus_exp.log()
}

// ── Gated Attention ────────────────────────────────────────────────────

pub(super) struct Qwen3_5Attention {
    pub(super) q_proj: Linear,
    pub(super) k_proj: Linear,
    pub(super) v_proj: Linear,
    pub(super) o_proj: Linear,
    pub(super) q_norm: RmsNorm,
    pub(super) k_norm: RmsNorm,
    pub(super) q_norm_weight: Tensor,
    pub(super) k_norm_weight: Tensor,
    pub(super) rope: PartialRotaryEmbedding,
    pub(super) kv_cache: Option<(Tensor, Tensor)>,
    pub(super) k_cache: Vec<Tensor>,
    pub(super) v_cache: Vec<Tensor>,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rms_norm_eps: f64,
    pub(super) softmax_scale: f64,
    pub(super) attn_output_gate: bool,
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

        // Qwen3.5 uses residual RMSNorm: output = norm(x) * (1 + weight). HF
        // casts weight to F32 before the `1 +` and the final multiply, so small
        // weight values aren't crushed by BF16 rounding near 1.0. Match that by
        // materialising `(1 + weight)` in F32 and letting `ops.rms_norm` fall
        // through to the composed F32 path when it sees an F32 weight.
        let q_norm_weight = (vb
            .pp("q_norm")
            .get(cfg.head_dim, "weight")?
            .to_dtype(DType::F32)?
            + 1.0)?;
        let q_norm = RmsNorm::from_weight(q_norm_weight.clone(), cfg.rms_norm_eps);

        let k_norm_weight = (vb
            .pp("k_norm")
            .get(cfg.head_dim, "weight")?
            .to_dtype(DType::F32)?
            + 1.0)?;
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
            k_cache: Vec::new(),
            v_cache: Vec::new(),
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
        self.k_cache.clear();
        self.v_cache.clear();
    }

    /// Flash-attn-v3 varlen forward for GPU prefill.
    fn forward(&mut self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
        let total_tokens = x.dim(0)?;
        let bs = BatchState::no_lora();

        // Project
        let q_raw = self.q_proj.forward(x, &bs, ctx.ops)?;
        let k = self.k_proj.forward(x, &bs, ctx.ops)?;
        let v = self.v_proj.forward(x, &bs, ctx.ops)?;

        // Split Q and gate. `narrow` yields a strided view over the packed
        // `[Q | gate]` tensor; downstream fused kernels (rms_norm, RoPE) assume
        // contiguous input and can hit misaligned-address errors on strided data,
        // so materialize here.
        let (q, gate) = if self.attn_output_gate {
            let q_and_gate = q_raw.reshape((total_tokens, self.num_heads, self.head_dim * 2))?;
            let q = q_and_gate
                .narrow(D::Minus1, 0, self.head_dim)?
                .contiguous()?;
            let gate = q_and_gate
                .narrow(D::Minus1, self.head_dim, self.head_dim)?
                .contiguous()?
                .reshape((total_tokens, self.num_heads * self.head_dim))?;
            (q, Some(gate))
        } else {
            let q = q_raw.reshape((total_tokens, self.num_heads, self.head_dim))?;
            (q, None)
        };

        let k = k.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;

        // Per-head QK normalization
        let q = ctx.ops.rms_norm(
            &q.reshape((total_tokens * self.num_heads, self.head_dim))?,
            &self.q_norm_weight,
            self.rms_norm_eps as f32,
        )?
        .reshape((total_tokens, self.num_heads, self.head_dim))?;
        let k = ctx.ops.rms_norm(
            &k.reshape((total_tokens * self.num_kv_heads, self.head_dim))?,
            &self.k_norm_weight,
            self.rms_norm_eps as f32,
        )?
        .reshape((total_tokens, self.num_kv_heads, self.head_dim))?;

        // Partial RoPE
        let (q, k) = self.rope.apply_varlen(&q, &k, ctx.position_ids)?;
        // FA4 / FlashInfer require stride(-1) == 1. `apply_varlen` goes through
        // `Tensor::cat` which can leave a non-row-stride layout; force contig.
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Attention dispatch: paged KV cache or plain varlen
        let softmax_scale = self.softmax_scale as f32;
        let attn_output = if let Some(kv) = ctx.paged_kv {
            ctx.ops.reshape_and_cache(&k, &v, kv.key_cache, kv.value_cache, kv.slot_mapping)?;
            ctx.ops.paged_attention(&q, kv.key_cache, kv.value_cache, &PagedParams {
                block_tables: kv.block_tables,
                cu_seqlens_q: ctx.cu_seqlens_q, cu_seqlens_k: kv.cu_seqlens_k,
                max_seqlen_q: ctx.max_seqlen_q, max_seqlen_k: kv.max_seqlen_k,
                scale: softmax_scale, mask: MaskType::Causal, softcap: None,
            })?
        } else {
            ctx.ops.varlen_attention(&q, &k, &v, &VarlenParams {
                cu_seqlens_q: ctx.cu_seqlens_q, cu_seqlens_k: ctx.cu_seqlens_q,
                max_seqlen_q: ctx.max_seqlen_q, max_seqlen_k: ctx.max_seqlen_q,
                scale: softmax_scale, mask: MaskType::Causal, softcap: None,
            })?
        };
        let attn_output = attn_output.reshape((total_tokens, self.num_heads * self.head_dim))?;
        let gated = if let Some(gate) = gate {
            (attn_output * ctx.ops.sigmoid(&gate)?)?
        } else {
            attn_output
        };
        self.o_proj.forward(&gated, &bs, ctx.ops)
    }

    /// Cached forward for CPU decode: handles both prefill (L>1) and decode (L=1).
    /// KV cache accumulates across calls; call `clear_cache()` between requests.
    fn forward_with_cache(&mut self, ops: &dyn crate::ops::Ops, x: &Tensor, position_offset: usize) -> Result<Tensor> {
        let seq_len = x.dim(0)?;
        let bs = BatchState::no_lora();

        // 1. Project Q/K/V (separate projections, same as varlen forward)
        let q_raw = self.q_proj.forward(x, &bs, ops)?;
        let k = self.k_proj.forward(x, &bs, ops)?;
        let v = self.v_proj.forward(x, &bs, ops)?;

        // Split Q and gate
        let (q, gate) = if self.attn_output_gate {
            let q_and_gate = q_raw.reshape((seq_len, self.num_heads, self.head_dim * 2))?;
            let q = q_and_gate.narrow(D::Minus1, 0, self.head_dim)?;
            let gate = q_and_gate
                .narrow(D::Minus1, self.head_dim, self.head_dim)?
                .reshape((seq_len, self.num_heads * self.head_dim))?;
            (q.contiguous()?, Some(gate))
        } else {
            let q = q_raw.reshape((seq_len, self.num_heads, self.head_dim))?;
            (q, None)
        };

        let k = k.reshape((seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        // 2. Per-head QK normalization
        let q = ops.rms_norm(
            &q.reshape((seq_len * self.num_heads, self.head_dim))?,
            &self.q_norm_weight,
            self.rms_norm_eps as f32,
        )?
        .reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = ops.rms_norm(
            &k.reshape((seq_len * self.num_kv_heads, self.head_dim))?,
            &self.k_norm_weight,
            self.rms_norm_eps as f32,
        )?
        .reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        // 3. Partial RoPE
        let position_ids: Vec<u32> = (0..seq_len).map(|i| (position_offset + i) as u32).collect();
        let position_ids_t = Tensor::from_vec(position_ids, (seq_len,), x.device())?;
        let (q, k) = self.rope.apply_varlen(&q, &k, &position_ids_t)?;

        // 4. KV cache: append and get full
        self.k_cache.push(k);
        self.v_cache.push(v);
        let k_full = Tensor::cat(&self.k_cache, 0)?;
        let v_full = Tensor::cat(&self.v_cache, 0)?;
        let total_kv_len = k_full.dim(0)?;

        // 5. GQA repeat
        let kv_ratio = self.num_heads / self.num_kv_heads;
        let (k_full, v_full) = if kv_ratio > 1 {
            let k_expanded = k_full
                .unsqueeze(2)?
                .expand((total_kv_len, self.num_kv_heads, kv_ratio, self.head_dim))?
                .reshape((total_kv_len, self.num_heads, self.head_dim))?;
            let v_expanded = v_full
                .unsqueeze(2)?
                .expand((total_kv_len, self.num_kv_heads, kv_ratio, self.head_dim))?
                .reshape((total_kv_len, self.num_heads, self.head_dim))?;
            (k_expanded, v_expanded)
        } else {
            (k_full, v_full)
        };

        // 6. Attention via matmul (simple CPU path)
        let q = q.transpose(0, 1)?; // [H, L, D]
        let k_t = k_full.transpose(0, 1)?; // [H, kv_len, D]
        let v_t = v_full.transpose(0, 1)?; // [H, kv_len, D]
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn_weights = (q.matmul(&k_t.transpose(1, 2)?)? * scale)?;

        // Causal mask (only needed for prefill, decode has seq_len=1)
        let attn_weights = if seq_len > 1 {
            let offset = total_kv_len - seq_len;
            let mut mask_data = vec![0.0f32; seq_len * total_kv_len];
            for i in 0..seq_len {
                for j in (offset + i + 1)..total_kv_len {
                    mask_data[i * total_kv_len + j] = f32::NEG_INFINITY;
                }
            }
            let causal_mask =
                Tensor::from_vec(mask_data, (1, seq_len, total_kv_len), x.device())?;
            attn_weights
                .to_dtype(DType::F32)?
                .broadcast_add(&causal_mask)?
        } else {
            attn_weights.to_dtype(DType::F32)?
        };

        let last_dim = attn_weights.rank() - 1;
        let attn_weights = ops.softmax(&attn_weights, last_dim)?;
        let attn_weights = attn_weights.to_dtype(v_t.dtype())?;
        let attn_out = attn_weights.matmul(&v_t)?; // [H, L, D]
        let attn_out = attn_out
            .transpose(0, 1)? // [L, H, D]
            .reshape((seq_len, self.num_heads * self.head_dim))?;

        // 7. Gate + O projection
        let gated = if let Some(gate) = gate {
            (attn_out * ops.sigmoid(&gate)?)?
        } else {
            attn_out
        };
        self.o_proj.forward(&gated, &bs, ops)
    }
}

// ── Dense MLP ───────────────────────────────────────────────────────────

pub(super) struct Qwen3_5Mlp {
    pub(super) gate_proj: Linear,
    pub(super) up_proj: Linear,
    pub(super) down_proj: Linear,
}

impl Qwen3_5Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::load(vb.pp("gate_proj"), hidden_size, intermediate_size, false)?,
            up_proj: Linear::load(vb.pp("up_proj"), hidden_size, intermediate_size, false)?,
            down_proj: Linear::load(vb.pp("down_proj"), intermediate_size, hidden_size, false)?,
        })
    }

    fn forward(&self, ops: &dyn crate::ops::Ops, x: &Tensor) -> Result<Tensor> {
        let bs = BatchState::no_lora();
        let gate = self.gate_proj.forward(x, &bs, ops)?;
        let up = self.up_proj.forward(x, &bs, ops)?;
        self.down_proj.forward(&ops.silu_mul(&gate, &up)?, &bs, ops)
    }
}

// ── Sparse MoE Block ────────────────────────────────────────────────────

struct Qwen3_5SparseMoeBlock {
    gate: DenseLinear, // [num_experts, hidden_size]
    // Fused expert weights: gate_up [E, 2*inter, hidden], down [E, hidden, inter]
    experts_gate_up: Tensor,
    experts_down: Tensor,
    moe_intermediate_size: usize,
    shared_expert: Option<Qwen3_5Mlp>,
    /// Shared-expert gating weight `[hidden_size]` (raw, not wrapped as Linear).
    /// We apply it as `(x * w).sum(-1, keepdim)` because a degenerate `out_dim=1`
    /// matmul lands on an NN GEMM layout that CUTLASS/DeepGEMM don't support.
    shared_expert_gate_weight: Option<Tensor>,
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
            DenseLinear::new(w, None)
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

        let shared_expert_gate_weight = if shared_expert.is_some() {
            // Stored as `[1, hidden]` in the checkpoint; flatten to `[hidden]`.
            let gvb = vb.pp("shared_expert_gate");
            Some(gvb.get((1, cfg.hidden_size), "weight")?.squeeze(0)?)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts_gate_up,
            experts_down,
            moe_intermediate_size,
            shared_expert,
            shared_expert_gate_weight,
            num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, ops: &dyn crate::ops::Ops, xs: &Tensor) -> Result<Tensor> {
        let ndim = xs.dims().len();
        if ndim == 2 {
            return self.forward_2d(ops, xs);
        }
        let (b, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let result = self.forward_2d(ops, &xs_flat)?;
        result.reshape((b, seq_len, hidden_dim))
    }

    fn expert_forward(&self, expert_idx: usize, x: &Tensor, ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        let gate_up_w = self.experts_gate_up.get(expert_idx)?; // [2*inter, hidden]
        let down_w = self.experts_down.get(expert_idx)?; // [hidden, inter]
        let inter = self.moe_intermediate_size;

        if x.device().is_cpu() {
            return self.expert_forward_matmul(x, &gate_up_w, &down_w, inter, ops);
        }

        let gate_up = x.matmul(&gate_up_w.t()?)?;
        let gate = gate_up.narrow(D::Minus1, 0, inter)?;
        let up = gate_up.narrow(D::Minus1, inter, inter)?;
        let act = ops.silu(&gate)?;
        let hidden = (act * up)?;
        hidden.matmul(&down_w.t()?)
    }

    fn expert_forward_matmul(
        &self, x: &Tensor, gate_up_w: &Tensor, down_w: &Tensor, inter: usize,
        ops: &dyn crate::ops::Ops,
    ) -> Result<Tensor> {
        // gate_up GEMM: x @ gate_up_w^T
        let gate_up = x.matmul(&gate_up_w.t()?)?;
        let gate = gate_up.narrow(gate_up.dims().len() - 1, 0, inter)?;
        let up = gate_up.narrow(gate_up.dims().len() - 1, inter, inter)?;
        // SiLU(gate) * up
        let silu_gate = ops.silu(&gate)?;
        let hidden = (&silu_gate * &up)?;
        // down GEMM: hidden @ down_w^T
        hidden.matmul(&down_w.t()?)
    }

    fn forward_2d(&self, ops: &dyn crate::ops::Ops, xs: &Tensor) -> Result<Tensor> {
        let (n_tokens, hidden_dim) = xs.dims2()?;

        // Router: softmax in F32 (to match reference `softmax(..., dtype=float32)`)
        // over 256 experts — BF16 softmax loses too much precision at this width.
        let router_logits = xs.apply(&self.gate)?;
        let router_logits_f32 = router_logits.to_dtype(DType::F32)?;
        let last_dim = router_logits_f32.rank() - 1;
        let routing_weights = ops.softmax(&router_logits_f32, last_dim)?;

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

        let mut routed_out = if xs.device().is_cuda() {
            self.forward_fused(ops, xs, &topk_weights, &experts_per_tok, n_tokens, hidden_dim)?
        } else {
            self.forward_sequential_cpu(ops, xs, &topk_weights, &experts_per_tok, hidden_dim)?
        };

        // Shared expert
        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(ops, xs)?;
            let shared_out = if let Some(ref gate_w) = self.shared_expert_gate_weight {
                // logit[n] = Σ_h xs[n, h] * gate_w[h]
                let logits = xs.broadcast_mul(&gate_w.unsqueeze(0)?)?.sum_keepdim(D::Minus1)?;
                let gate_val = ops.sigmoid(&logits)?; // [n, 1]
                shared_out.broadcast_mul(&gate_val)?
            } else {
                shared_out
            };
            routed_out = (routed_out + shared_out)?;
        }

        Ok(routed_out)
    }

    /// Fused MoE dispatch (CUDA fast path).
    ///
    /// One `grouped_gemm` over the stacked `[E, 2*inter, hidden]` gate_up
    /// weight handles all experts in a single kernel, then `silu_mul_concat`
    /// fuses the activation, then `fused_moe_gemm` applies the down projection
    /// with topk-weighted accumulation. This replaces a sequential
    /// `for t { for k in 0..topk { two tiny matmuls } }` loop that did two
    /// D2H syncs per layer via `to_vec2()` and launched ~`n_tokens * topk * 2`
    /// tiny GEMMs — catastrophic for a 40-layer 256-expert top-8 model.
    fn forward_fused(
        &self,
        ops: &dyn crate::ops::Ops,
        xs: &Tensor,
        topk_weights: &Tensor,
        experts_per_tok: &Tensor,
        n_tokens: usize,
        hidden_dim: usize,
    ) -> Result<Tensor> {
        let (sorted_expert_ids, sorted_token_ids) =
            sort_expert_assignments(experts_per_tok, xs.device())?;

        let is_prefill = n_tokens > 1;
        // `num_tokens_per_expert` is unused by the current CUDA grouped_gemm
        // kernel (it derives offsets internally), but keep the arg for trait
        // compatibility. Reuse `sorted_expert_ids` as the sentinel to avoid an
        // extra D2H sync via count_tokens_per_expert.
        let counts_sentinel = &sorted_expert_ids;

        // Single grouped GEMM against fused [E, 2*inter, hidden] → [N*topk, 2*inter]
        let gate_up = ops.grouped_gemm(
            xs,
            &self.experts_gate_up,
            &sorted_token_ids,
            &sorted_expert_ids,
            counts_sentinel,
        )?;

        // silu(gate) * up — prefer fused concat path that avoids a narrow+copy
        let inter = self.moe_intermediate_size;
        let down_input = match ops.silu_mul_concat(&gate_up) {
            Some(r) => r?,
            None => {
                let gate = gate_up.narrow(D::Minus1, 0, inter)?.contiguous()?;
                let up = gate_up.narrow(D::Minus1, inter, inter)?.contiguous()?;
                ops.silu_mul(&gate, &up)?
            }
        };

        // Down projection with fused topk weighted accumulation.
        let ys = match ops.fused_moe_gemm(
            &down_input,
            &self.experts_down,
            topk_weights,
            &sorted_token_ids,
            &sorted_expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        ) {
            Some(r) => r?,
            None => {
                // Fallback: unweighted grouped_gemm + manual weighted sum.
                let raw = ops.grouped_gemm(
                    &down_input,
                    &self.experts_down,
                    &sorted_token_ids,
                    &sorted_expert_ids,
                    counts_sentinel,
                )?;
                let raw = raw.reshape((n_tokens, self.num_experts_per_tok, hidden_dim))?;
                let w = topk_weights.unsqueeze(D::Minus1)?;
                return (raw * w)?.sum(D::Minus2);
            }
        };

        ys.reshape((n_tokens, self.num_experts_per_tok, hidden_dim))?
            .sum(D::Minus2)
    }

    /// CPU fallback: sequential per-token expert dispatch. Kept for
    /// correctness when the CUDA fused path is unavailable.
    fn forward_sequential_cpu(
        &self,
        ops: &dyn crate::ops::Ops,
        xs: &Tensor,
        topk_weights: &Tensor,
        experts_per_tok: &Tensor,
        hidden_dim: usize,
    ) -> Result<Tensor> {
        let topk_weights_vec: Vec<Vec<f32>> = topk_weights.to_vec2()?;
        let experts_per_tok_vec: Vec<Vec<u32>> = experts_per_tok.to_vec2()?;

        let n_tokens = topk_weights_vec.len();
        let mut routed_out = Tensor::zeros((n_tokens, hidden_dim), xs.dtype(), xs.device())?;

        for t in 0..n_tokens {
            let x_t = xs.get(t)?.unsqueeze(0)?;
            let mut acc = Tensor::zeros((1, hidden_dim), DType::F32, xs.device())?;
            for k in 0..self.num_experts_per_tok {
                let expert_idx = experts_per_tok_vec[t][k] as usize;
                let weight = topk_weights_vec[t][k];
                let expert_out = self
                    .expert_forward(expert_idx, &x_t, ops)?
                    .to_dtype(DType::F32)?;
                acc = (acc + (expert_out * weight as f64)?)?;
            }
            let acc = acc.to_dtype(xs.dtype())?;
            routed_out = routed_out.slice_assign(&[t..t + 1, 0..hidden_dim], &acc)?;
        }
        Ok(routed_out)
    }
}

/// Sort topk expert assignments so that all tokens routed to the same expert
/// are contiguous in the output. Returns `(sorted_expert_ids, sorted_token_ids)`
/// both as flat `[num_tokens * topk]` u32 tensors. The `sorted_token_ids[i]`
/// gives the original flat assignment index (token * topk + k) for the i-th
/// entry in the sorted expert stream, which is what `grouped_gemm` expects.
fn sort_expert_assignments(
    experts_per_tok: &Tensor,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let flat = experts_per_tok.flatten_all()?;
    let n = flat.elem_count();

    if n <= 1024 && device.is_cuda() {
        let flat_2d = flat.reshape((1, n))?;
        let (sorted_vals, sorted_idx) = flat_2d.sort_last_dim(true)?;
        return Ok((sorted_vals.flatten_all()?, sorted_idx.flatten_all()?));
    }

    let flat_vec = flat.to_vec1::<u32>()?;
    let mut indices: Vec<u32> = (0..n as u32).collect();
    indices.sort_by_key(|&i| flat_vec[i as usize]);
    let sorted_expert_ids: Vec<u32> = indices.iter().map(|&i| flat_vec[i as usize]).collect();
    Ok((
        Tensor::from_vec(sorted_expert_ids, (n,), device)?,
        Tensor::from_vec(indices, (n,), device)?,
    ))
}

pub(super) enum MlpVariant {
    Dense(Qwen3_5Mlp),
    Sparse(Qwen3_5SparseMoeBlock),
}

impl MlpVariant {
    fn forward(&self, ops: &dyn crate::ops::Ops, x: &Tensor) -> Result<Tensor> {
        match self {
            MlpVariant::Dense(mlp) => mlp.forward(ops, x),
            MlpVariant::Sparse(moe) => moe.forward(ops, x),
        }
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────

pub(super) enum TokenMixer {
    LinearAttention(Qwen3_5GatedDeltaNet),
    FullAttention(Qwen3_5Attention),
}

pub(super) struct Qwen3_5DecoderLayer {
    pub(super) token_mixer: TokenMixer,
    pub(super) mlp: MlpVariant,
    ln1_weight: Tensor,
    ln2_weight: Tensor,
    rms_norm_eps: f32,
}

/// Free function to run DeltaNet on packed varlen input WITHOUT the pool
/// (single-batch prefill-only path, e.g. `prefill_pipeline` on servers that
/// have no DeltaNet pool wired up). State is cleared per sequence so each
/// batch starts fresh and requests can't leak state into each other through
/// `gdn`'s transient fields. Decode never lands here — the GPU decode runners
/// always build `BatchAttnContext` with `deltanet_slots` populated, which
/// takes `deltanet_varlen_pooled` instead.
fn deltanet_varlen(
    gdn: &mut Qwen3_5GatedDeltaNet,
    packed: &Tensor,
    seq_lens: &[usize],
    ops: &dyn crate::ops::Ops,
) -> Result<Tensor> {
    let mut outputs = Vec::new();
    let mut offset = 0usize;
    for &len in seq_lens {
        gdn.clear_state();
        let seq = packed.narrow(0, offset, len)?.unsqueeze(0)?; // [1, L, D]
        let out = gdn.forward(&seq, 0, ops)?; // [1, L, D]
        outputs.push(out.squeeze(0)?); // [L, D]
        offset += len;
    }
    gdn.clear_state();
    Tensor::cat(&outputs, 0) // [total_tokens, D]
}

/// Pooled varlen DeltaNet: for each sequence in the batch,
///   1. load this request's recurrent + conv state from the pool slot,
///   2. run the per-token delta rule / conv1d forward,
///   3. write the updated state back to the same slot.
///
/// Works for both prefill (len > 1, starting from zero because
/// `DeltaNetPool::allocate` zeros the slot) and decode (len == 1, reading the
/// state left over from the previous decode step). `gdn`'s transient
/// `recurrent_state` / `conv_state` fields are used as per-call scratch — we
/// load into them before `gdn.forward` and clear them after we scatter back,
/// so concurrent requests sharing the same layer struct don't leak state.
fn deltanet_varlen_pooled(
    gdn: &mut Qwen3_5GatedDeltaNet,
    packed: &Tensor,
    seq_lens: &[usize],
    pool: &mut crate::deltanet_pool::DeltaNetPool,
    slot_ids: &[u32],
    dn_layer_idx: usize,
    ops: &dyn crate::ops::Ops,
) -> Result<Tensor> {
    // Fast path: when every request in the batch is a single decode token
    // AND the fused cuLA `kda_decode` kernel is available for this model's
    // shape, let one CUDA call handle the delta rule step for the whole
    // batch. The kernel reads each request's recurrent state from the pool
    // via slot indices and updates it in place.
    //
    // We must decide eligibility BEFORE running conv1d, because the fused
    // path and the sequential fallback BOTH mutate the pool's conv_state.
    // If we ran conv1d in the fused path and then fell through, the
    // conv_state would be advanced twice (producing "Paris Paris Paris …"
    // on repeat as the state drifts).
    let all_decode = !seq_lens.is_empty() && seq_lens.iter().all(|&l| l == 1);
    // The cuLA kda_decode launcher is specialized on (H, HV, K, V), so we
    // enumerate the AOT-compiled variants here. Keep this in sync with
    // `crates/prelude-cuda/cula/scripts/compile_kernels.py`.
    let fused_supported = matches!(
        (
            gdn.num_k_heads,
            gdn.num_v_heads,
            gdn.head_k_dim,
            gdn.head_v_dim,
        ),
        (32, 32, 128, 128)   // MHA H=32
        | (64, 64, 128, 128) // MHA H=64
        | (16, 32, 128, 128) // GQA H=16/HV=32 (Qwen3.5-35B-A3B)
    );
    let fused_eligible = all_decode && packed.device().is_cuda() && fused_supported;
    if fused_eligible {
        if let Some(out) = deltanet_decode_batched_fused(
            gdn, packed, pool, slot_ids, dn_layer_idx, ops,
        )? {
            return Ok(out);
        }
    }

    let mut outputs = Vec::new();
    let mut offset = 0usize;
    for (i, &len) in seq_lens.iter().enumerate() {
        let slot = slot_ids[i] as usize;

        // Load this request's state from the pool into the layer scratch.
        // Force contiguous so downstream ops (matmul, slice_set on write-back)
        // see a dense layout. `get(slot)` strips the leading max_slots dim.
        gdn.recurrent_state = Some(
            pool.recurrent_states[dn_layer_idx]
                .get(slot)?
                .contiguous()?,
        );
        gdn.conv_state = Some(
            pool.conv_states[dn_layer_idx]
                .get(slot)?
                .contiguous()?,
        );

        let seq = packed.narrow(0, offset, len)?.unsqueeze(0)?; // [1, L, D]
        let out = gdn.forward(&seq, 0, ops)?; // [1, L, D]
        outputs.push(out.squeeze(0)?); // [L, D]

        // Scatter updated state back to the same slot. `slice_set` needs the
        // src contiguous, and intermediate broadcasts in delta_rule_step can
        // leave a non-contiguous layout, so materialize before write-back.
        if let Some(ref state) = gdn.recurrent_state {
            let row = state.contiguous()?.unsqueeze(0)?.contiguous()?;
            pool.recurrent_states[dn_layer_idx].slice_set(&row, 0, slot)?;
        }
        if let Some(ref state) = gdn.conv_state {
            let row = state.contiguous()?.unsqueeze(0)?.contiguous()?;
            pool.conv_states[dn_layer_idx].slice_set(&row, 0, slot)?;
        }
        offset += len;
    }
    gdn.clear_state(); // scratch is only valid within this call
    Tensor::cat(&outputs, 0) // [total_tokens, D]
}

/// Batched decode-only fast path: one fused cuLA kernel call per layer for
/// the whole decode batch. Returns `Ok(None)` if the kernel can't be used
/// (wrong GPU arch, shape mismatch, etc.) so the caller falls back to the
/// sequential loop. Conv1d is still done sequentially per request — cuLA
/// doesn't ship a batched conv1d_decode, and it's cheap enough not to be
/// the bottleneck.
fn deltanet_decode_batched_fused(
    gdn: &mut Qwen3_5GatedDeltaNet,
    packed: &Tensor,
    pool: &mut crate::deltanet_pool::DeltaNetPool,
    slot_ids: &[u32],
    dn_layer_idx: usize,
    ops: &dyn crate::ops::Ops,
) -> Result<Option<Tensor>> {
    // Only hybrid GPU paths pass through here, but guard anyway.
    if !packed.device().is_cuda() {
        return Ok(None);
    }

    let n = slot_ids.len();
    if n == 0 {
        return Ok(None);
    }
    // `packed` is `[total_tokens, hidden]` and all seq_lens == 1, so
    // `total_tokens == n`. Fold the batch dim into the time dim for the
    // projections: `gdn.in_proj_*` expect the existing `[B, L, hidden]`
    // layout, so wrap `packed` as `[1, N, hidden]`.
    let bst = BatchState::no_lora();
    let x = packed.unsqueeze(0)?; // [1, N, hidden]

    let qkv = gdn.in_proj_qkv.forward(&x, &bst, ops)?;   // [1, N, key*2+val]
    let z   = gdn.in_proj_z  .forward(&x, &bst, ops)?;   // [1, N, value_dim]
    let b_raw_full = gdn.in_proj_b.forward(&x, &bst, ops)?; // [1, N, HV]
    let a_raw_full = gdn.in_proj_a.forward(&x, &bst, ops)?; // [1, N, HV]

    let q_cat = qkv.narrow(D::Minus1, 0, gdn.key_dim)?;
    let k_cat = qkv.narrow(D::Minus1, gdn.key_dim, gdn.key_dim)?;
    let v_cat = qkv.narrow(D::Minus1, gdn.key_dim * 2, gdn.value_dim)?;
    let qkv_for_conv = Tensor::cat(&[&q_cat, &k_cat, &v_cat], D::Minus1)?; // [1, N, conv_dim]
    let qkv_for_conv_2d = qkv_for_conv.squeeze(0)?; // [N, conv_dim]

    // ── Sequential conv1d_decode per request ───────────────────────
    // cuLA doesn't expose a batched variant, so for each request we load
    // its conv_state slice from the pool, run the one-token decode, then
    // scatter the updated conv_state back. This step is tiny (kernel size
    // = 4, conv_dim ≈ 8192) compared to the delta-rule step.
    let device = packed.device();
    let mut conv_outs: Vec<Tensor> = Vec::with_capacity(n);
    for (i, &slot) in slot_ids.iter().enumerate() {
        let slot_us = slot as usize;
        gdn.conv_state = Some(
            pool.conv_states[dn_layer_idx]
                .get(slot_us)?
                .contiguous()?,
        );
        let x_row = qkv_for_conv_2d.get(i)?.contiguous()?; // [conv_dim]
        let out = gdn.conv1d_decode(&x_row)?; // [conv_dim]
        conv_outs.push(out);
        if let Some(ref state) = gdn.conv_state {
            let row = state.contiguous()?.unsqueeze(0)?.contiguous()?;
            pool.conv_states[dn_layer_idx].slice_set(&row, 0, slot_us)?;
        }
    }
    gdn.conv_state = None;

    // Stack conv outputs: [N, conv_dim]
    let qkv_conv = Tensor::stack(&conv_outs, 0)?;
    let qkv_conv = ops.silu(&qkv_conv)?;

    // Split into q, k, v with head layout.
    let q_flat = qkv_conv.narrow(D::Minus1, 0, gdn.key_dim)?;
    let k_flat = qkv_conv.narrow(D::Minus1, gdn.key_dim, gdn.key_dim)?;
    let v_flat = qkv_conv.narrow(D::Minus1, gdn.key_dim * 2, gdn.value_dim)?;
    let q_nhk = q_flat.reshape((n, gdn.num_k_heads, gdn.head_k_dim))?.contiguous()?;
    let k_nhk = k_flat.reshape((n, gdn.num_k_heads, gdn.head_k_dim))?.contiguous()?;
    let v_nhv = v_flat.reshape((n, gdn.num_v_heads, gdn.head_v_dim))?.contiguous()?;

    // Scalar-per-head a/b drop the batch-of-1 dim.
    let a_2d = a_raw_full.squeeze(0)?.contiguous()?; // [N, HV]
    let b_2d = b_raw_full.squeeze(0)?.contiguous()?; // [N, HV]

    // Slot ids need to live on the device as an i32 tensor.
    let slot_ids_gpu = Tensor::from_vec(slot_ids.to_vec(), (n,), device)?; // U32 on GPU

    let decode_out = match ops.kda_decode_batched(
        &q_nhk,
        &k_nhk,
        &v_nhv,
        &a_2d,
        &b_2d,
        &gdn.a_log,
        &gdn.dt_bias,
        &pool.recurrent_states[dn_layer_idx],
        &slot_ids_gpu,
    ) {
        Some(Ok(o)) => o,    // [N, HV, V]
        Some(Err(e)) => return Err(e),
        None => return Ok(None),
    };

    // Reshape into [1, N, value_dim] for the gated norm + out projection,
    // matching what the sequential path hands to `gdn.norm.forward`.
    let out_nhv = decode_out.reshape((n, gdn.value_dim))?;
    let out_1nd = out_nhv.unsqueeze(0)?;

    let z_cont = z.contiguous()?; // [1, N, value_dim]
    let normed = gdn.norm.forward(&out_1nd, &z_cont, ops)?;
    let projected = gdn.out_proj.forward(&normed, &bst, ops)?; // [1, N, hidden]
    Ok(Some(projected.squeeze(0)?)) // [N, hidden]
}

impl Qwen3_5DecoderLayer {
    fn new(
        cfg: &Qwen3_5Config,
        layer_idx: usize,
        rope: PartialRotaryEmbedding,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Qwen3.5 uses residual RMSNorm: output = norm(x) * (1 + weight). See
        // `Qwen3_5Attention::new` for why this is computed in F32.
        let ln1_weight = (vb
            .pp("input_layernorm")
            .get(cfg.hidden_size, "weight")?
            .to_dtype(DType::F32)?
            + 1.0)?;
        let ln2_weight = (vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")?
            .to_dtype(DType::F32)?
            + 1.0)?;

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
            token_mixer, mlp, ln1_weight, ln2_weight,
            rms_norm_eps: cfg.rms_norm_eps as f32,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        ctx: &LayerAttnContext,
        seq_lens: &[usize],
    ) -> Result<Tensor> {
        let ops = ctx.ops;
        let h = ops.rms_norm(x, &self.ln1_weight, self.rms_norm_eps)?;
        let h = match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.forward(&h, ctx)?,
            TokenMixer::LinearAttention(gdn) => deltanet_varlen(gdn, &h, seq_lens, ops)?,
        };
        let (x_res, h2) = ops.add_rmsnorm(x, &h, &self.ln2_weight, self.rms_norm_eps)?;
        ops.add_or_fused(&x_res, &self.mlp.forward(ops, &h2)?)
    }

    fn forward_with_paged_prefix_pooled(
        &mut self,
        ops: &dyn crate::ops::Ops,
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
        let h = ops.rms_norm(x, &self.ln1_weight, self.rms_norm_eps)?;
        let h = match &mut self.token_mixer {
            TokenMixer::LinearAttention(gdn) => {
                deltanet_varlen_pooled(gdn, &h, seq_lens, pool, slot_ids, dn_layer_idx, ops)?
            }
            TokenMixer::FullAttention(_) => {
                crate::tensor::bail!("forward_with_paged_prefix_pooled called on FullAttention layer")
            }
        };
        let (x_res, h2) = ops.add_rmsnorm(x, &h, &self.ln2_weight, self.rms_norm_eps)?;
        ops.add_or_fused(&x_res, &self.mlp.forward(ops, &h2)?)
    }

    fn clear_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::LinearAttention(gdn) => gdn.clear_state(),
            TokenMixer::FullAttention(attn) => attn.clear_cache(),
        }
    }

    fn forward_with_cache(&mut self, ops: &dyn crate::ops::Ops, x: &Tensor, position_offset: usize) -> Result<Tensor> {
        let h = ops.rms_norm(x, &self.ln1_weight, self.rms_norm_eps)?;
        let h = match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.forward_with_cache(ops, &h, position_offset)?,
            TokenMixer::LinearAttention(gdn) => {
                let h3d = h.unsqueeze(0)?;
                gdn.forward(&h3d, 0, ops)?.squeeze(0)?
            }
        };
        let x = (x + h)?;
        let h2 = ops.rms_norm(&x, &self.ln2_weight, self.rms_norm_eps)?;
        let h2 = self.mlp.forward(ops, &h2)?;
        &x + h2
    }
}

// ── Model ───────────────────────────────────────────────────────────────

pub(super) struct Qwen3_5Model {
    pub(super) embed_tokens: Embedding,
    pub(super) layers: Vec<Qwen3_5DecoderLayer>,
    pub(super) norm: RmsNorm,
    pub(super) norm_weight: Tensor,
    pub(super) rms_norm_eps: f64,
}

impl Qwen3_5Model {
    pub(super) fn new(cfg: &Qwen3_5Config, vb: VarBuilder) -> Result<Self> {
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

        // Qwen3.5 uses residual RMSNorm: output = norm(x) * (1 + weight). See
        // `Qwen3_5Attention::new` for why this is computed in F32.
        let norm_weight = (vb_m
            .pp("norm")
            .get(cfg.hidden_size, "weight")?
            .to_dtype(DType::F32)?
            + 1.0)?;
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
                            ops: ctx.ops,
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
                                ctx.ops,
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
                                ops: ctx.ops,
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
                ops: ctx.ops,
                cu_seqlens_q: ctx.cu_seqlens_q,
                max_seqlen_q: ctx.max_seqlen_q,
                position_ids: ctx.position_ids,
                paged_kv: None,
            };
            for layer in self.layers.iter_mut() {
                h = layer.forward(&h, &layer_ctx, seq_lens)?;
            }
        }
        ctx.ops.rms_norm(&h, &self.norm_weight, self.rms_norm_eps as f32)
    }

    fn clear_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }

    pub(super) fn forward_with_cache(&mut self, input_ids: &Tensor, position_offset: usize) -> Result<Tensor> {
        let ops = crate::ops::select_ops(input_ids.device());
        let mut h = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            h = layer.forward_with_cache(ops, &h, position_offset)?;
        }
        ops.rms_norm(&h, &self.norm_weight, self.rms_norm_eps as f32)
    }
}

// ── ForCausalLM ─────────────────────────────────────────────────────────

pub struct Qwen3_5ForCausalLM {
    pub(super) model: Qwen3_5Model,
    pub(super) lm_head: Linear,
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
        self.lm_head.forward(
            &last_token_select(&hidden, ctx.seq_lens)?.unsqueeze(1)?,
            &BatchState::no_lora(), ctx.ops,
        )
    }

    /// Cached forward: returns logits `[L, vocab_size]` for all input tokens.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
        let hidden = self.model.forward_with_cache(input_ids, position_offset)?;
        self.lm_head.forward(&hidden, &BatchState::no_lora(), crate::ops::select_ops(hidden.device()))
    }
}

impl crate::models::LogitsSplitModel for Qwen3_5ForCausalLM {
    fn forward_hidden_states(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> crate::tensor::Result<Tensor> {
        self.model.forward(packed_input, ctx)
    }

    fn compute_logits(&self, hidden: &Tensor) -> crate::tensor::Result<Tensor> {
        self.lm_head.forward(hidden, &BatchState::no_lora(), crate::ops::select_ops(hidden.device()))
    }
}

impl crate::models::KvCacheModel for Qwen3_5ForCausalLM {
    fn forward_with_cache(
        &mut self,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> crate::tensor::Result<Tensor> {
        Qwen3_5ForCausalLM::forward_with_cache(self, input_ids, position_offset)
    }
}

impl crate::models::ModelForward for Qwen3_5ForCausalLM {
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
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

    fn as_kv_cache_model(&mut self) -> Option<&mut dyn crate::models::KvCacheModel> {
        Some(self)
    }
}

pub(crate) mod meta {
    use crate::loading::var_builder::VarBuilder;

    use super::{Qwen3_5Config, Qwen3_5ForCausalLM};
    use crate::cache::deltanet_pool::DeltaNetPoolConfig;
    use crate::engine::EngineError;
    use crate::engine::{CommonModelConfig, RuntimeCaps, TaskKind, WeightsBackend};
    use crate::models::registry::{
        candle_model_err, parse_json, ArchSpec, ParsedModelConfig,
    };

    const ARCHITECTURE_ALIASES: &[&str] = &["Qwen3_5", "Qwen35", "Qwen3_5Moe", "Qwen35Moe"];
    const MODEL_TYPE_ALIASES: &[&str] = &["qwen3_5_text", "qwen3_5", "qwen3_5_moe_text", "qwen3_5_moe"];
    const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate];

    fn deltanet_config_from(cfg: &Qwen3_5Config) -> DeltaNetPoolConfig {
        let num_deltanet_layers = (0..cfg.num_hidden_layers)
            .filter(|i| (i + 1) % cfg.full_attention_interval != 0)
            .count();
        DeltaNetPoolConfig {
            num_deltanet_layers,
            num_v_heads: cfg.linear_num_value_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            conv_dim: cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
                + cfg.linear_num_value_heads * cfg.linear_value_head_dim,
            conv_kernel: cfg.linear_conv_kernel_dim,
        }
    }

    pub(crate) struct Qwen3_5ArchSpec;

    pub(crate) static QWEN3_5_ARCH_SPEC: Qwen3_5ArchSpec = Qwen3_5ArchSpec;
    inventory::submit!(crate::models::registry::ArchSpecEntry::new(&QWEN3_5_ARCH_SPEC));

    impl ArchSpec for Qwen3_5ArchSpec {
        fn name(&self) -> &'static str {
            "qwen3_5"
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
            _task: TaskKind,
            _raw: &serde_json::Value,
            content: &str,
        ) -> Result<ParsedModelConfig, EngineError> {
            let cfg = parse_json::<Qwen3_5Config>(content, "Qwen3.5 config")?;
            let common = CommonModelConfig {
                vocab_size: cfg.vocab_size,
                num_hidden_layers: cfg.num_hidden_layers,
                max_position_embeddings: cfg.max_position_embeddings,
                num_attention_heads: cfg.num_attention_heads,
                num_key_value_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
            };
            let deltanet = Some(deltanet_config_from(&cfg));
            Ok(ParsedModelConfig {
                common,
                deltanet,
                arch_config: Box::new(cfg),
            })
        }

        fn build_model(
            &self,
            arch_config: &dyn std::any::Any,
            vb: VarBuilder<'_>,
        ) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
            let cfg = arch_config.downcast_ref::<Qwen3_5Config>().ok_or_else(|| {
                EngineError::Internal("unexpected arch config type for Qwen3.5".into())
            })?;
            Ok(Box::new(
                Qwen3_5ForCausalLM::new(cfg, vb).map_err(candle_model_err)?,
            ))
        }

        fn runtime_caps(
            &self,
            task: TaskKind,
            backend: WeightsBackend,
            device: &crate::tensor::Device,
        ) -> RuntimeCaps {
            let is_safetensors = backend == WeightsBackend::Safetensors;
            let _is_generate = task == TaskKind::Generate;

            RuntimeCaps {
                supports_kv_cache: false,
                supports_prefix_cache: false,
                supports_paged_attn: device.is_cuda() && is_safetensors,
                supports_varlen: device.is_cuda() && is_safetensors,
                supports_deltanet: true,
                supports_cuda_graph: false,
            }
        }

        fn gguf_aliases(&self) -> &'static [&'static str] {
            &["qwen35", "qwen35moe"]
        }

        fn load_gguf(
            &self,
            ct: crate::tensor::quantized::gguf_file::Content,
            reader: &mut std::fs::File,
            device: &crate::tensor::Device,
        ) -> Result<crate::models::registry::GgufLoadResult, EngineError> {
            let (model, cfg) = super::gguf::Qwen3_5GgufModel::from_gguf(ct, reader, device)
                .map_err(candle_model_err)?;
            let common = CommonModelConfig {
                vocab_size: cfg.vocab_size,
                num_hidden_layers: cfg.num_hidden_layers,
                max_position_embeddings: cfg.max_position_embeddings,
                num_attention_heads: cfg.num_attention_heads,
                num_key_value_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
            };
            let deltanet = Some(DeltaNetPoolConfig {
                num_deltanet_layers: (0..cfg.num_hidden_layers)
                    .filter(|i| (i + 1) % cfg.full_attention_interval != 0)
                    .count(),
                num_v_heads: cfg.linear_num_value_heads,
                head_k_dim: cfg.linear_key_head_dim,
                head_v_dim: cfg.linear_value_head_dim,
                conv_dim: cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
                    + cfg.linear_num_value_heads * cfg.linear_value_head_dim,
                conv_kernel: cfg.linear_conv_kernel_dim,
            });
            Ok(crate::models::registry::GgufLoadResult {
                model: Box::new(model),
                common,
                deltanet,
                eos_token_ids: cfg.eos_token_ids,
            })
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GGUF support (from qwen3_5/gguf.rs)
// ═══════════════════════════════════════════════════════════════════════

pub mod gguf {
    //! Quantized Qwen3.5 model loaded from GGUF format.
    //!
    //! Reference: llama.cpp `src/models/qwen35.cpp` + `src/models/delta-net-base.cpp`.
    
    use crate::tensor::quantized::gguf_file;
    use crate::tensor::{DType, Device, Result, Tensor};
    use std::io::{Read, Seek};
    use std::sync::Arc;
    
    use crate::constants::GGUF_INTERMEDIATE_SIZE_MULTIPLIER;
    use crate::models::commons::{Linear, RmsNorm};
    use crate::models::commons::embedding::Embedding;
    
    // ── Config ──────────────────────────────────────────────────────────────
    
    #[derive(Debug, Clone)]
    pub struct Qwen3_5GgufConfig {
        pub num_hidden_layers: usize,
        pub hidden_size: usize,
        pub intermediate_size: usize,
        pub num_attention_heads: usize,
        pub num_key_value_heads: usize,
        pub head_dim: usize,
        pub max_position_embeddings: usize,
        pub rms_norm_eps: f64,
        pub rope_theta: f64,
        pub vocab_size: usize,
        pub eos_token_ids: Vec<u32>,
        // DeltaNet
        pub linear_num_key_heads: usize,
        pub linear_num_value_heads: usize,
        pub linear_key_head_dim: usize,
        pub linear_value_head_dim: usize,
        pub linear_conv_kernel_dim: usize,
        pub full_attention_interval: usize,
        pub partial_rotary_factor: f64,
    }
    
    impl Qwen3_5GgufConfig {
        fn key_dim(&self) -> usize {
            self.linear_num_key_heads * self.linear_key_head_dim
        }
    
        fn value_dim(&self) -> usize {
            self.linear_num_value_heads * self.linear_value_head_dim
        }
    
        fn conv_dim(&self) -> usize {
            self.key_dim() * 2 + self.value_dim()
        }
    
        fn rotary_dim(&self) -> usize {
            (self.head_dim as f64 * self.partial_rotary_factor) as usize
        }
    
        fn is_recurrent(&self, layer_idx: usize) -> bool {
            (layer_idx + 1) % self.full_attention_interval != 0
        }
    }
    
    // ── Model ───────────────────────────────────────────────────────────────
    
    pub struct Qwen3_5GgufModel {
        inner: super::Qwen3_5ForCausalLM,
    }
    
    impl Qwen3_5GgufModel {
        fn load_linear<R: Read + Seek>(
            ct: &gguf_file::Content,
            reader: &mut R,
            name: &str,
            device: &Device,
        ) -> Result<Linear> {
            let qtensor = ct.tensor(reader, name, device)?;
            Linear::from_qtensor(Arc::new(qtensor))
        }
    
        fn load_tensor<R: Read + Seek>(
            ct: &gguf_file::Content,
            reader: &mut R,
            name: &str,
            device: &Device,
        ) -> Result<Tensor> {
            let qtensor = ct.tensor(reader, name, device)?;
            qtensor.dequantize(device)
        }
    
        pub fn from_gguf<R: Read + Seek>(
            ct: gguf_file::Content,
            reader: &mut R,
            device: &Device,
        ) -> Result<(Self, Qwen3_5GgufConfig)> {
            let config = parse_gguf_config(&ct)?;
    
            // Embedding (dequantize for lookup table)
            let embed_weight = Self::load_tensor(&ct, reader, "token_embd.weight", device)?;
            let embed_tokens = Embedding::new(embed_weight, config.hidden_size);
    
            // Build PartialRotaryEmbedding from GGUF config
            let build_rope = |dtype: DType| -> Result<super::PartialRotaryEmbedding> {
                let rotary_dim = config.rotary_dim();
                let half = rotary_dim / 2;
                let inv_freq: Vec<f32> = (0..rotary_dim)
                    .step_by(2)
                    .map(|i| 1.0 / config.rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
                    .collect();
                let inv_freq =
                    Tensor::from_vec(inv_freq, (1, half), device)?.to_dtype(DType::F32)?;
                let positions =
                    Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
                        .to_dtype(DType::F32)?
                        .reshape((config.max_position_embeddings, 1))?;
                let freqs = positions.broadcast_mul(&inv_freq)?;
                let cos = freqs.cos()?.to_dtype(dtype)?;
                let sin = freqs.sin()?.to_dtype(dtype)?;
                Ok(super::PartialRotaryEmbedding {
                    cos,
                    sin,
                    rotary_dim,
                })
            };
    
            // Build layers
            let mut layers = Vec::with_capacity(config.num_hidden_layers);
            let dtype = DType::F32;
    
            for i in 0..config.num_hidden_layers {
                let prefix = format!("blk.{i}");
    
                // Layer norms (GGUF already has +1 applied for residual RMSNorm)
                let ln1_weight =
                    Self::load_tensor(&ct, reader, &format!("{prefix}.attn_norm.weight"), device)?;

                let ln2_weight = Self::load_tensor(
                    &ct,
                    reader,
                    &format!("{prefix}.post_attention_norm.weight"),
                    device,
                )?;
    
                // Token mixer
                let token_mixer = if config.is_recurrent(i) {
                    // DeltaNet layer
                    let in_proj_qkv = Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.attn_qkv.weight"),
                        device,
                    )?;
                    let in_proj_z = Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.attn_gate.weight"),
                        device,
                    )?;
                    let in_proj_b = Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.ssm_beta.weight"),
                        device,
                    )?;
                    let in_proj_a = Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.ssm_alpha.weight"),
                        device,
                    )?;
    
                    // Conv1d weight: GGUF stores [d_conv, conv_channels], transpose if needed
                    let conv_weight_raw = Self::load_tensor(
                        &ct,
                        reader,
                        &format!("{prefix}.ssm_conv1d.weight"),
                        device,
                    )?;
                    let conv_weight =
                        if conv_weight_raw.dim(0)? == config.linear_conv_kernel_dim {
                            conv_weight_raw.t()?.contiguous()?
                        } else {
                            conv_weight_raw
                        };
    
                    let dt_bias = Self::load_tensor(
                        &ct,
                        reader,
                        &format!("{prefix}.ssm_dt.bias"),
                        device,
                    )?;
    
                    // GGUF stores -exp(A_log); convert back: A_log = ln(-ssm_a)
                    let ssm_a =
                        Self::load_tensor(&ct, reader, &format!("{prefix}.ssm_a"), device)?;
                    let a_log = ssm_a.neg()?.log()?;
    
                    // Gated RMSNorm (NO +1 for ssm_norm)
                    let norm_weight = Self::load_tensor(
                        &ct,
                        reader,
                        &format!("{prefix}.ssm_norm.weight"),
                        device,
                    )?;
                    let norm = super::RmsNormGated {
                        weight: norm_weight,
                        eps: config.rms_norm_eps,
                        num_heads: config.linear_num_value_heads,
                        head_dim: config.linear_value_head_dim,
                    };
    
                    let out_proj = Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.ssm_out.weight"),
                        device,
                    )?;
    
                    let key_dim = config.key_dim();
                    let value_dim = config.value_dim();
                    let conv_dim = config.conv_dim();
    
                    super::TokenMixer::LinearAttention(super::Qwen3_5GatedDeltaNet {
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
                        num_k_heads: config.linear_num_key_heads,
                        num_v_heads: config.linear_num_value_heads,
                        head_k_dim: config.linear_key_head_dim,
                        head_v_dim: config.linear_value_head_dim,
                        key_dim,
                        value_dim,
                        conv_dim,
                        conv_kernel: config.linear_conv_kernel_dim,
                    })
                } else {
                    // Full attention layer
                    let q_proj = Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.attn_q.weight"),
                        device,
                    )?;
                    let k_proj = Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.attn_k.weight"),
                        device,
                    )?;
                    let v_proj = Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.attn_v.weight"),
                        device,
                    )?;
                    let o_proj = Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.attn_output.weight"),
                        device,
                    )?;
    
                    // QK norms (GGUF already has +1 applied)
                    let q_norm_weight = Self::load_tensor(
                        &ct,
                        reader,
                        &format!("{prefix}.attn_q_norm.weight"),
                        device,
                    )?;
                    let k_norm_weight = Self::load_tensor(
                        &ct,
                        reader,
                        &format!("{prefix}.attn_k_norm.weight"),
                        device,
                    )?;
    
                    let q_norm =
                        RmsNorm::from_weight(q_norm_weight.clone(), config.rms_norm_eps);
                    let k_norm =
                        RmsNorm::from_weight(k_norm_weight.clone(), config.rms_norm_eps);
    
                    let rope = build_rope(dtype)?;
    
                    super::TokenMixer::FullAttention(super::Qwen3_5Attention {
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
                        k_cache: Vec::new(),
                        v_cache: Vec::new(),
                        num_heads: config.num_attention_heads,
                        num_kv_heads: config.num_key_value_heads,
                        head_dim: config.head_dim,
                        rms_norm_eps: config.rms_norm_eps,
                        softmax_scale: 1.0 / (config.head_dim as f64).sqrt(),
                        attn_output_gate: true,
                    })
                };
    
                // MLP (same for both layer types)
                let mlp = super::MlpVariant::Dense(super::Qwen3_5Mlp {
                    gate_proj: Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.ffn_gate.weight"),
                        device,
                    )?,
                    up_proj: Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.ffn_up.weight"),
                        device,
                    )?,
                    down_proj: Self::load_linear(
                        &ct,
                        reader,
                        &format!("{prefix}.ffn_down.weight"),
                        device,
                    )?,
                });
    
                layers.push(super::Qwen3_5DecoderLayer {
                    token_mixer,
                    mlp,
                    ln1_weight,
                    ln2_weight,
                    rms_norm_eps: config.rms_norm_eps as f32,
                });
            }
    
            // Final norm (GGUF already has +1 applied)
            let norm_weight = Self::load_tensor(&ct, reader, "output_norm.weight", device)?;
            let norm = RmsNorm::from_weight(norm_weight.clone(), config.rms_norm_eps);
    
            // LM head
            let lm_head = if ct.tensor_infos.get("output.weight").is_some() {
                Self::load_linear(&ct, reader, "output.weight", device)?
            } else {
                // Tied embeddings: use token_embd.weight
                Self::load_linear(&ct, reader, "token_embd.weight", device)?
            };
    
            let model = super::Qwen3_5Model {
                embed_tokens,
                layers,
                norm,
                norm_weight,
                rms_norm_eps: config.rms_norm_eps,
            };
    
            let inner = super::Qwen3_5ForCausalLM { model, lm_head };
    
            Ok((Self { inner }, config))
        }
    
        pub fn clear_kv_cache(&mut self) {
            self.inner.clear_kv_cache();
        }
    
        pub fn forward_with_cache(
            &mut self,
            input_ids: &Tensor,
            position_offset: usize,
        ) -> Result<Tensor> {
            self.inner.forward_with_cache(input_ids, position_offset)
        }
    
        pub fn forward(
            &mut self,
            packed_input: &Tensor,
            ctx: &mut crate::models::commons::BatchAttnContext,
        ) -> Result<Tensor> {
            self.inner.forward(packed_input, ctx)
        }
    }
    
    impl crate::models::KvCacheModel for Qwen3_5GgufModel {
        fn forward_with_cache(
            &mut self,
            input_ids: &Tensor,
            position_offset: usize,
        ) -> Result<Tensor> {
            Qwen3_5GgufModel::forward_with_cache(self, input_ids, position_offset)
        }
    }
    
    impl crate::models::ModelForward for Qwen3_5GgufModel {
        fn forward(
            &mut self,
            packed_input: &Tensor,
            ctx: &mut crate::models::commons::BatchAttnContext,
        ) -> Result<Tensor> {
            self.inner.forward(packed_input, ctx)
        }
    
        fn clear_kv_cache(&mut self) {
            self.clear_kv_cache();
        }
    
        fn as_kv_cache_model(&mut self) -> Option<&mut dyn crate::models::KvCacheModel> {
            Some(self)
        }
    }
    
    // ── GGUF Config Parsing ─────────────────────────────────────────────────
    
    fn parse_gguf_config(ct: &gguf_file::Content) -> Result<Qwen3_5GgufConfig> {
        let md = &ct.metadata;
    
        let get_u32 = |key: &str| -> Result<usize> {
            let val = md.get(key)
                .ok_or_else(|| crate::tensor::Error::Msg(format!("missing GGUF metadata: {key}")))?;
            Ok(val.to_u32().map(|v| v as usize)?)
        };
    
        let get_u32_or = |key: &str, default: usize| -> usize {
            md.get(key)
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or_else(|| {
                    tracing::warn!(
                        "Qwen3.5 GGUF: '{key}' not found, using default: {default}"
                    );
                    default
                })
        };
    
        let get_f32 = |key: &str| -> Result<f64> {
            let val = md.get(key)
                .ok_or_else(|| crate::tensor::Error::Msg(format!("missing GGUF metadata: {key}")))?;
            Ok(val.to_f32().map(|v| v as f64)?)
        };
    
        let get_f32_or = |key: &str, default: f64| -> f64 {
            md.get(key)
                .and_then(|v| v.to_f32().ok())
                .map(|v| v as f64)
                .unwrap_or_else(|| {
                    tracing::warn!(
                        "Qwen3.5 GGUF: '{key}' not found, using default: {default}"
                    );
                    default
                })
        };
    
        // Auto-detect architecture prefix
        let default_arch = "qwen35".to_string();
        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .unwrap_or_else(|| {
                tracing::warn!(
                    "Qwen3.5 GGUF: 'general.architecture' not found, using default: qwen35"
                );
                &default_arch
            });
    
        let num_hidden_layers = get_u32(&format!("{arch}.block_count"))?;
        let hidden_size = get_u32(&format!("{arch}.embedding_length"))?;
        let num_attention_heads = get_u32(&format!("{arch}.attention.head_count"))?;
        let num_key_value_heads = get_u32(&format!("{arch}.attention.head_count_kv"))?;
        let head_dim = get_u32(&format!("{arch}.attention.key_length"))?;
        let max_position_embeddings = get_u32(&format!("{arch}.context_length"))?;
        let rms_norm_eps = get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))?;
        let rope_theta = get_f32(&format!("{arch}.rope.freq_base"))?;
    
        let default_intermediate = hidden_size * GGUF_INTERMEDIATE_SIZE_MULTIPLIER;
        let intermediate_size = get_u32_or(
            &format!("{arch}.feed_forward_length"),
            default_intermediate,
        );
    
        // SSM / DeltaNet specific metadata
        let ssm_d_conv = get_u32_or(&format!("{arch}.ssm.conv_kernel"), 4);
        let ssm_d_inner = get_u32_or(&format!("{arch}.ssm.inner_size"), hidden_size);
        let ssm_d_state = get_u32_or(&format!("{arch}.ssm.state_size"), 128);
        let ssm_dt_rank = get_u32_or(&format!("{arch}.ssm.time_step_rank"), 16);
        let ssm_n_group = get_u32_or(&format!("{arch}.ssm.group_count"), 16);
        let full_attention_interval =
            get_u32_or(&format!("{arch}.full_attention_interval"), 4);
    
        // Derive linear attention dimensions from SSM params (llama.cpp convention)
        let linear_num_key_heads = ssm_n_group;
        let linear_num_value_heads = ssm_dt_rank;
        let linear_key_head_dim = ssm_d_state;
        let linear_value_head_dim = if ssm_dt_rank > 0 {
            ssm_d_inner / ssm_dt_rank
        } else {
            ssm_d_state
        };
    
        // Partial rotary factor from rope_dimension_sections or default
        let partial_rotary_factor = get_f32_or(
            &format!("{arch}.rope.partial_rotary_factor"),
            0.25,
        );
    
        let vocab_size = ct
            .tensor_infos
            .get("token_embd.weight")
            .map(|t| t.shape.dims()[0])
            .unwrap_or_else(|| {
                tracing::warn!(
                    "Qwen3.5 GGUF: 'token_embd.weight' tensor not found, using default vocab_size: 248320"
                );
                248320
            });
    
        let eos_token_ids = md
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .map(|id| vec![id])
            .unwrap_or_else(|| {
                tracing::warn!(
                    "Qwen3.5 GGUF: 'tokenizer.ggml.eos_token_id' not found, using empty list"
                );
                vec![]
            });
    
        Ok(Qwen3_5GgufConfig {
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            vocab_size,
            eos_token_ids,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim: ssm_d_conv,
            full_attention_interval,
            partial_rotary_factor,
        })
    }
    
}
