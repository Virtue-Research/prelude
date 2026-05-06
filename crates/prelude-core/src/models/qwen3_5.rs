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

use crate::loading::var_builder::VarBuilder;
use crate::models::commons::embedding::Embedding;
use crate::models::commons::linear::DenseLinear;
use crate::tensor::{D, DType, Device, Module, Result, Tensor};

use crate::models::commons::{
    BatchAttnContext, BatchState, LayerAttnContext, Linear, last_token_select,
};
use crate::models::resolve_or_warn;
use crate::ops::{MaskType, PagedParams, VarlenParams};
use std::collections::BTreeMap;

const MAX_GROUPED_PREFILL_BATCH: usize = 8;
const MAX_ZERO_GROUPED_PREFILL_BATCH: usize = 16;

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
        let weight = vb.get_with_hints_dtype(head_dim, "weight", Default::default(), DType::F32)?;
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

        // Flatten to 2D [...*num_heads, head_dim] for the fused kernel
        let flat_rows = x.elem_count() / self.head_dim;
        let x_2d = x.reshape((flat_rows, self.head_dim))?;
        let z_2d = z.reshape((flat_rows, self.head_dim))?;

        // Fused: RMSNorm(x) * weight * SiLU(gate) in one kernel
        let result = if let Some(r) = ops.rmsnorm_gated(&x_2d, &z_2d, &self.weight, self.eps as f32)
        {
            r?
        } else {
            // Decomposed fallback (CPU or non-BF16)
            let x_f32 = x_2d.to_dtype(DType::F32)?;
            let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
            let normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
            let normed = normed.broadcast_mul(&self.weight)?;
            let silu_gate = ops.silu(&z_2d.to_dtype(DType::F32)?)?;
            normed.broadcast_mul(&silu_gate)?.to_dtype(x.dtype())?
        };

        result.reshape(orig_shape)
    }
}

// ── Gated DeltaNet (Linear Attention) ────────────────────────────────────

pub(super) struct Qwen3_5GatedDeltaNet {
    // Qwen3.5 keeps QKV/Z/B/A as split projections. Fusing QKV/Z forces large
    // contiguous copies before conv/norm in this implementation, and fusing
    // only B/A did not improve the 35B prefill path in profiling.
    pub(super) in_proj_qkv: Linear, // hidden → key_dim*2 + value_dim
    pub(super) in_proj_z: Linear,   // hidden → value_dim
    pub(super) in_proj_b: Linear,   // hidden → num_v_heads
    pub(super) in_proj_a: Linear,   // hidden → num_v_heads
    pub(super) conv_weight: Tensor, // [conv_dim, kernel_size] reshaped for dot product
    dt_bias_f32: Tensor,
    dt_bias_f32_expanded: Tensor,
    a_log_f32: Tensor,
    prefill_cu_seqlens: Option<(usize, Tensor)>,
    pub(super) norm: RmsNormGated,
    pub(super) out_proj: Linear,
    // State
    pub(super) conv_state: Option<Tensor>, // [conv_dim, kernel-1]
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

        let in_proj_qkv = Linear::load(
            vb.pp("in_proj_qkv"),
            cfg.hidden_size,
            key_dim * 2 + value_dim, // Q + K + V
            false,
        )?;
        let in_proj_z = Linear::load(vb.pp("in_proj_z"), cfg.hidden_size, value_dim, false)?;
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
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;
        let dt_bias_f32_expanded = dt_bias_f32
            .reshape((cfg.linear_num_value_heads, 1))?
            .broadcast_as((cfg.linear_num_value_heads, cfg.linear_key_head_dim))?
            .contiguous()?;
        let a_log_f32 = if a_log.dtype() == DType::F32 {
            a_log.clone()
        } else {
            a_log.to_dtype(DType::F32)?
        };

        let norm = RmsNormGated::new(
            cfg.linear_value_head_dim,
            cfg.linear_num_value_heads,
            cfg.rms_norm_eps,
            vb.pp("norm"),
        )?;
        let out_proj = Linear::load(vb.pp("out_proj"), value_dim, cfg.hidden_size, false)?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv_weight,
            dt_bias_f32,
            dt_bias_f32_expanded,
            a_log_f32,
            prefill_cu_seqlens: None,
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

    fn single_prefill_cu_seqlens(&mut self, seq_len: usize, device: &Device) -> Result<Tensor> {
        if let Some((cached_len, cached)) = &self.prefill_cu_seqlens {
            if *cached_len == seq_len {
                return Ok(cached.clone());
            }
        }
        let cu = Tensor::from_vec(vec![0i64, seq_len as i64], (2,), device)?;
        self.prefill_cu_seqlens = Some((seq_len, cu.clone()));
        Ok(cu)
    }

    /// Forward pass for a single token (decode) or a sequence (prefill).
    fn forward(&mut self, x: &Tensor, _offset: usize, ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        assert_eq!(b, 1, "Qwen3.5 DeltaNet only supports batch_size=1");
        let bst = BatchState::no_lora();

        // Split projections. See the comment in `new()` for why the
        // four are kept separate rather than fused into one matmul.
        let qkv = self.in_proj_qkv.forward(x, &bst, ops)?; // [1, L, key_dim*2 + value_dim]
        let z = self.in_proj_z.forward(x, &bst, ops)?; // [1, L, value_dim]
        let b_param = self.in_proj_b.forward(x, &bst, ops)?; // [1, L, num_v_heads]
        let a_param = self.in_proj_a.forward(x, &bst, ops)?; // [1, L, num_v_heads]

        // `qkv` already has the `[Q | K | V]` layout conv1d wants.
        let qkv_for_conv = qkv; // [B, L, conv_dim]

        // Apply causal conv1d. Both `conv1d_decode` and `conv1d_prefill`
        // fuse the SiLU activation into the kernel (fast path) or the
        // fallback loop, so we don't apply a separate SiLU here.
        let qkv_conv = if seq_len == 1 {
            self.conv1d_decode(&qkv_for_conv.squeeze(0)?.squeeze(0)?, ops)?
                .unsqueeze(0)?
                .unsqueeze(0)?
        } else {
            self.conv1d_prefill(&qkv_for_conv, ops)?
        };

        // Squeeze batch dim and pass the packed mixed_qkv straight
        // through to `delta_rule_prefill`. The fast path wants the
        // un-split tensor so it can hand it to `gdn_post_conv` as one
        // contiguous channel blob; the fallback still splits it inside.
        let mixed_qkv = qkv_conv.get(0)?; // [L, conv_dim]
        let b_param = b_param.get(0)?;
        let a_param = a_param.get(0)?;

        // Batched delta-rule prefill. For seq_len==1 we still go through this
        // path; the single-step "loop" is cheap and keeps the hot code in one
        // place.
        let device = x.device();
        let output = self
            .delta_rule_prefill(&mixed_qkv, &b_param, &a_param, device, ops)?
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
        mixed_qkv: &Tensor, // [T, 2*key_dim + value_dim] — post-conv1d
        b_in: &Tensor,      // [T, num_v_heads]
        a_in: &Tensor,      // [T, num_v_heads]
        device: &Device,
        ops: &dyn crate::ops::Ops,
    ) -> Result<Tensor> {
        let t = mixed_qkv.dim(0)?;
        let hv = self.num_v_heads;
        let hk = self.num_k_heads;
        let kdim = self.head_k_dim;
        let vdim = self.head_v_dim;
        let out_dtype = mixed_qkv.dtype();

        // Fast path: FlashInfer `gdn_prefill` kernel — the one
        // Qwen3.5 / Qwen3-next / FLA's `chunk_gated_delta_rule` were
        // designed for. Scalar-per-head linear-space decay, no chunk
        // cumsum / no safe_gate clamp — bit-exact (within BF16) to the
        // HF reference.
        if device.is_cuda() && kdim == vdim && kdim == 128 {
            if let Some(out) = self.delta_rule_prefill_gdn(mixed_qkv, b_in, a_in, device, ops)? {
                return Ok(out.to_dtype(out_dtype)?);
            }
        }

        // ── Fallback: batched-matmul recurrence on whatever device we have ──
        //
        // Split `mixed_qkv` back into per-head Q / K / V slices.
        // `mixed_qkv` layout is `[Q (HK*K) | K (HK*K) | V (HV*V)]`.
        let q_in = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k_in = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v_in = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

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
        let neg_a_exp = self.a_log_f32.exp()?.neg()?; // [HV]
        let a_f32 = a_in.to_dtype(DType::F32)?; // [T, HV]
        let a_plus_dt = a_f32.broadcast_add(&self.dt_bias_f32)?; // [T, HV]
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
            let state_k = state_decayed.broadcast_mul(&k_row_1xk)?.sum(D::Minus1)?; // [HV, V]

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
            let out = state.broadcast_mul(&q_row_1xk)?.sum(D::Minus1)?; // [HV, V]
            outputs.push(out);
        }

        self.recurrent_state = Some(state);

        // Stack outputs: [T, HV, V] → [T, value_dim] → caller dtype
        let out_stacked = Tensor::stack(&outputs, 0)?; // [T, HV, V]
        out_stacked
            .reshape((t, self.value_dim))?
            .to_dtype(out_dtype)
    }

    /// FlashInfer-backed GDN prefill fast path for Qwen3.5 DeltaNet.
    ///
    /// Two-stage fast path:
    ///  1. `ops.gdn_post_conv` — one fused CUDA kernel produces Q, K
    ///     (L2-normalised), V (raw), `alpha = exp(g_scalar)` and
    ///     `beta = sigmoid(b_raw)` directly from the mixed_qkv channel
    ///     blob + the raw gate inputs. Replaces ~20 candle ops per layer.
    ///  2. `ops.gdn_prefill_varlen` — FlashInfer's fused gdn_prefill
    ///     kernel consuming the prepped tensors.
    ///
    /// This matches HF transformers' `chunk_gated_delta_rule` semantics
    /// bit-for-bit modulo BF16 rounding: scalar-per-head **linear-space**
    /// decay, no `RCP_LN2` rescaling, no `safe_gate` clamp, no per-element
    /// broadcast. The kernel math is the same one Qwen3.5 was trained
    /// against.
    ///
    /// Returns `Ok(None)` when any backend can't serve the call
    /// (unsupported architecture, shape mismatch, kernel not compiled);
    /// caller falls back.
    fn delta_rule_prefill_gdn(
        &mut self,
        mixed_qkv: &Tensor, // [T, 2*HK*D + HV*D] BF16
        b_in: &Tensor,      // [T, HV] BF16
        a_in: &Tensor,      // [T, HV] BF16
        device: &Device,
        ops: &dyn crate::ops::Ops,
    ) -> Result<Option<Tensor>> {
        let t = mixed_qkv.dim(0)?;
        let hk = self.num_k_heads;
        let hv = self.num_v_heads;
        let kdim = self.head_k_dim;
        // FlashInfer uses `num_sab_heads = max(num_q, num_v)` for the
        // state / gate / output head axis. Qwen3.5 is GVA (num_v > num_k),
        // so alpha/beta/output live on the V-head axis.
        debug_assert!(hv >= hk, "Qwen3.5 DeltaNet is GVA: num_v >= num_k");

        // ── Stage 1: fused post-conv1d prep ────────────────────────────
        // `gdn_post_conv` wants dt_bias and A_log as F32. Keep cached F32
        // copies on the layer to avoid per-request cast kernels.
        let Some(prep) = ops.gdn_post_conv(
            mixed_qkv,
            a_in,
            b_in,
            &self.a_log_f32,
            &self.dt_bias_f32,
            hk,
            hv,
            kdim,
        ) else {
            // No CUDA fast path for gdn_post_conv — fall back to the
            // composed recurrence.
            return Ok(None);
        };
        let (q_bf16, k_bf16, v_bf16, alpha, beta) = prep?;

        // ── cu_seqlens = [0, T] for a single packed sequence (I64) ───────
        // Note: flashinfer expects I64, unlike cuLA's I32.
        let cu_seqlens = self.single_prefill_cu_seqlens(t, device)?;

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
            &q_bf16,
            &k_bf16,
            &v_bf16,
            &alpha,
            &beta,
            &cu_seqlens,
            initial_state_4d.as_ref(),
            scale,
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
    ///
    /// Prefers the fused `Ops::causal_conv1d_update` kernel (Dao-AILab
    /// mamba kernel), falls back to a broadcast+sum in-place update
    /// loop on non-CUDA / unsupported-width devices. Always fuses the
    /// SiLU tail so callers must NOT apply SiLU on top.
    fn conv1d_decode(&mut self, x: &Tensor, ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        // x: [conv_dim]
        let device = x.device();
        let dtype = x.dtype();
        let pad_len = self.conv_kernel - 1;

        // Lazy-init conv_state. The fast path expects `[B, D, W-1]`;
        // the fallback keeps the 2-D `[D, W-1]` layout the old code used.
        if self.conv_state.is_none() {
            self.conv_state = Some(Tensor::zeros((self.conv_dim, pad_len), dtype, device)?);
        }

        // Fast path: Ops::causal_conv1d_update
        //   wants `x: [B=1, D]`, `conv_state: [B=1, D, W-1]`,
        //   `weight: [D, W]`, returns `[B=1, D]`.
        //
        // Important: Dao's update kernel mutates `conv_state` in
        // place. If the state tensor we hand it aliases some OTHER
        // tensor's storage (e.g. `deltanet_varlen_pooled` loads the
        // state as a view into `pool.conv_states[layer]`), that other
        // tensor would be silently mutated too, and a later
        // `slice_set` write-back from our owned copy would refuse with
        // "cannot use slice_set when self and src share their
        // storage". Force a fresh allocation with `x + 0` so the
        // kernel's in-place update lands on our own buffer.
        let x_bd = x.unsqueeze(0)?; // [1, conv_dim]
        let state_flat = self.conv_state.as_ref().unwrap();
        let state_fresh = (state_flat + 0.0f64)?.contiguous()?; // [conv_dim, W-1] fresh
        let state_bd = state_fresh.unsqueeze(0)?.contiguous()?; // [1, conv_dim, W-1]
        if let Some(res) = ops.causal_conv1d_update(
            &x_bd,
            &state_bd,
            &self.conv_weight,
            None,
            /*silu_activation=*/ true,
            None, // conv_state_indices — single-seq decode, not pool-indexed
        ) {
            let out_bd = res?; // [1, conv_dim]
            // `state_bd` was mutated in place. Save it back to
            // `self.conv_state` — squeezed to our 2-D on-disk layout.
            self.conv_state = Some(state_bd.squeeze(0)?.contiguous()?);
            return out_bd.squeeze(0);
        }

        // ── Fallback: manual shift + broadcast * sum ────────────────
        let state = self.conv_state.as_ref().unwrap();
        let x_col = x.unsqueeze(D::Minus1)?; // [conv_dim, 1]
        let full_window = Tensor::cat(&[state, &x_col], 1)?; // [conv_dim, kernel]
        let out_raw = (full_window * &self.conv_weight)?.sum(D::Minus1)?; // [conv_dim]
        // Manual SiLU fusion parity with the fast path.
        let out = ops.silu(&out_raw)?;

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
    ///
    /// Prefers Dao-AILab's `causal_conv1d_fn` (fused kernel, SiLU fused
    /// in) and falls back to a per-kernel-position shift+sum loop on
    /// non-CUDA / unsupported-width paths.
    ///
    /// **Cross-chunk left context**: upstream's channel-first kernel
    /// silently ignores the `initial_states` pointer (only the
    /// channel-last variant honors it). To keep cross-chunk state
    /// correct we pre-pad `x` on the time axis with the saved
    /// `conv_state` ourselves (adding `W-1` tokens), then call the
    /// kernel with `L + W - 1` timesteps and slice the trailing `L`
    /// outputs. This is mathematically identical to what the upstream
    /// channel-last kernel would do via `initial_states_ptr`.
    fn conv1d_prefill(&mut self, x: &Tensor, ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        // x: [1, L, conv_dim]
        let (b, seq_len, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();
        let pad_len = self.conv_kernel - 1;

        // Transpose to `[B, D, L]` for the Dao-AILab kernel convention.
        let x_t = x.transpose(1, 2)?.contiguous()?; // [1, conv_dim, L]

        // Pre-pad left context with saved conv_state (or zeros on the
        // first chunk). The kernel processes `[B, D, pad_len + L]` and
        // emits `[B, D, pad_len + L]` whose last `L` timesteps are the
        // correct causal conv outputs; we drop the leading pad.
        let prefix = if let Some(ref state) = self.conv_state {
            state.unsqueeze(0)?.contiguous()? // [1, D, W-1]
        } else {
            Tensor::zeros((b, self.conv_dim, pad_len), dtype, device)?
        };
        let x_padded = Tensor::cat(&[&prefix, &x_t], 2)?.contiguous()?; // [1, D, pad_len+L]
        let padded_len = seq_len + pad_len;

        // ── Fast path: Ops::causal_conv1d_fn ─────────────────────────
        //
        // We pass `None` for `initial_states` — the prefix we prepended
        // above is the semantic equivalent and is guaranteed to be
        // honored by the kernel (unlike `initial_states` which upstream
        // silently drops in channel-first layout).
        //
        // SiLU is fused so the caller must NOT apply SiLU on top.
        if let Some(res) = ops.causal_conv1d_fn(
            &x_padded,
            &self.conv_weight,
            None,
            None,
            /*silu_activation=*/ true,
        ) {
            let result_padded = res?; // [1, D, pad_len+L]
            // Drop the leading `pad_len` outputs: they correspond to
            // timesteps in the pre-pad zone which don't belong to the
            // output sequence.
            let result_t = result_padded.narrow(2, pad_len, seq_len)?; // [1, D, L]
            let result = result_t.transpose(1, 2)?.contiguous()?; // [1, L, conv_dim]
            debug_assert_eq!(padded_len - pad_len, seq_len);

            // Save the last `W-1` raw inputs as the new conv_state.
            let x_t_2d = x_t.squeeze(0)?; // [conv_dim, L]
            if seq_len >= pad_len {
                self.conv_state = Some(x_t_2d.narrow(1, seq_len - pad_len, pad_len)?.contiguous()?);
            } else {
                let old = if let Some(ref state) = self.conv_state {
                    state.narrow(1, seq_len, pad_len - seq_len)?
                } else {
                    Tensor::zeros((self.conv_dim, pad_len - seq_len), dtype, device)?
                };
                self.conv_state = Some(Tensor::cat(&[old, x_t_2d], 1)?.contiguous()?);
            }
            return Ok(result);
        }

        // ── Fallback: per-kernel-position shift + sum loop ───────────
        //
        // For each `k_i ∈ 0..width`, the output contribution at time
        // `t` is `weight[:, k_i] * padded[:, t + k_i]`. We accumulate
        // across the 4 kernel offsets with simple tensor ops (no
        // matmul → CUDA-safe even when the CUTLASS wrapper can't serve
        // the NN-mode GEMM candle's conv1d path would emit).
        let prefix = if let Some(ref state) = self.conv_state {
            state.unsqueeze(0)?.contiguous()?
        } else {
            Tensor::zeros((b, self.conv_dim, pad_len), dtype, device)?
        };
        let padded = Tensor::cat(&[&prefix, &x_t], 2)?.contiguous()?; // [1, D, L+W-1]

        let mut acc: Option<Tensor> = None;
        for k_i in 0..self.conv_kernel {
            let shifted = padded.narrow(2, k_i, seq_len)?; // [1, D, L]
            // weight[:, k_i] → [D], reshape to [1, D, 1] for broadcast.
            let w_slice = self
                .conv_weight
                .narrow(1, k_i, 1)?
                .reshape((1, self.conv_dim, 1))?;
            let term = shifted.broadcast_mul(&w_slice)?;
            acc = Some(match acc {
                None => term,
                Some(a) => (a + term)?,
            });
        }
        let out_t = acc.unwrap(); // [1, D, L]
        // Fuse SiLU here so the caller can drop its separate silu call.
        let out_t = ops.silu(&out_t)?;
        let result = out_t.transpose(1, 2)?.contiguous()?; // [1, L, conv_dim]

        // Save state, same as fast path.
        let x_t_2d = x_t.squeeze(0)?;
        if seq_len >= pad_len {
            self.conv_state = Some(x_t_2d.narrow(1, seq_len - pad_len, pad_len)?.contiguous()?);
        } else {
            let old = if let Some(ref state) = self.conv_state {
                state.narrow(1, seq_len, pad_len - seq_len)?
            } else {
                Tensor::zeros((self.conv_dim, pad_len - seq_len), dtype, device)?
            };
            self.conv_state = Some(Tensor::cat(&[old, x_t_2d], 1)?.contiguous()?);
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

        let k_norm_weight = (vb
            .pp("k_norm")
            .get(cfg.head_dim, "weight")?
            .to_dtype(DType::F32)?
            + 1.0)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
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
        let q = ctx
            .ops
            .rms_norm(
                &q.reshape((total_tokens * self.num_heads, self.head_dim))?,
                &self.q_norm_weight,
                self.rms_norm_eps as f32,
            )?
            .reshape((total_tokens, self.num_heads, self.head_dim))?;
        let k = ctx
            .ops
            .rms_norm(
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
            ctx.ops
                .reshape_and_cache(&k, &v, kv.key_cache, kv.value_cache, kv.slot_mapping)?;
            ctx.ops.paged_attention(
                &q,
                kv.key_cache,
                kv.value_cache,
                &PagedParams {
                    block_tables: kv.block_tables,
                    cu_seqlens_q: ctx.cu_seqlens_q,
                    cu_seqlens_k: kv.cu_seqlens_k,
                    max_seqlen_q: ctx.max_seqlen_q,
                    max_seqlen_k: kv.max_seqlen_k,
                    scale: softmax_scale,
                    mask: MaskType::Causal,
                    softcap: None,
                },
            )?
        } else {
            ctx.ops.varlen_attention(
                &q,
                &k,
                &v,
                &VarlenParams {
                    cu_seqlens_q: ctx.cu_seqlens_q,
                    cu_seqlens_k: ctx.cu_seqlens_q,
                    max_seqlen_q: ctx.max_seqlen_q,
                    max_seqlen_k: ctx.max_seqlen_q,
                    scale: softmax_scale,
                    mask: MaskType::Causal,
                    softcap: None,
                },
            )?
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
    fn forward_with_cache(
        &mut self,
        ops: &dyn crate::ops::Ops,
        x: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
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
        let q = ops
            .rms_norm(
                &q.reshape((seq_len * self.num_heads, self.head_dim))?,
                &self.q_norm_weight,
                self.rms_norm_eps as f32,
            )?
            .reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = ops
            .rms_norm(
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
            let causal_mask = Tensor::from_vec(mask_data, (1, seq_len, total_kv_len), x.device())?;
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

pub(super) struct Qwen3_5SparseMoeBlock {
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
        // CUTLASS Swiglu expects [up|gate] order. Swap in-place at load time
        // via the Ops trait (2MB GPU temp, done once).
        if experts_gate_up.device().is_cuda() {
            let ops = crate::ops::select_ops(experts_gate_up.device());
            if let Some(result) = ops.swap_moe_gate_up(&experts_gate_up, moe_intermediate_size) {
                result?;
            }
        }
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
        let (n_tokens, hidden_dim) = xs.dims2()?;

        // Routing: fused kernel when available, decomposed fallback
        let router_logits = xs.apply(&self.gate)?;
        let (topk_weights, experts_per_tok) =
            self.compute_routing(ops, &router_logits, n_tokens)?;

        // Expert dispatch: fused CUTLASS pipeline when available,
        // composed grouped_gemm + silu_mul fallback
        let mut routed_out = if let Some(r) = ops.cutlass_fused_moe(
            xs,
            &experts_per_tok,
            &topk_weights,
            &self.experts_gate_up,
            &self.experts_down,
        ) {
            match r {
                Ok(out) => out,
                Err(error) => {
                    tracing::debug!(%error, "CUTLASS fused MoE failed; falling back to composed MoE");
                    self.forward_composed(
                        ops,
                        xs,
                        &topk_weights,
                        &experts_per_tok,
                        n_tokens,
                        hidden_dim,
                    )?
                }
            }
        } else {
            self.forward_composed(
                ops,
                xs,
                &topk_weights,
                &experts_per_tok,
                n_tokens,
                hidden_dim,
            )?
        };

        // 3. Shared expert
        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(ops, xs)?;
            let shared_out = if let Some(ref gate_w) = self.shared_expert_gate_weight {
                if let Some(result) = ops.shared_expert_gate(xs, &shared_out, gate_w) {
                    result?
                } else {
                    let logits = xs
                        .broadcast_mul(&gate_w.unsqueeze(0)?)?
                        .sum_keepdim(D::Minus1)?;
                    shared_out.broadcast_mul(&ops.sigmoid(&logits)?)?
                }
            } else {
                shared_out
            };
            routed_out = (routed_out + shared_out)?;
        }

        Ok(routed_out)
    }

    /// Routing: fused kernel on CUDA, decomposed on CPU.
    fn compute_routing(
        &self,
        ops: &dyn crate::ops::Ops,
        router_logits: &Tensor,
        n_tokens: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Try fused routing (single CUDA kernel, F32 softmax internally).
        // norm_topk_prob is now part of the kernel call instead of an
        // outer gate; passing the wrong value would silently change the
        // routing math.
        if let Some(result) =
            ops.fused_moe_routing(router_logits, self.num_experts_per_tok, self.norm_topk_prob)
        {
            let (tw, topk_ids, _, _) = result?;
            return Ok((tw, topk_ids.reshape((n_tokens, self.num_experts_per_tok))?));
        }
        // Decomposed fallback (CPU or non-BF16)
        let routing_weights = ops.softmax(
            &router_logits.to_dtype(DType::F32)?,
            router_logits.rank() - 1,
        )?;
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
        Ok((topk_weights, experts_per_tok))
    }

    /// Fused MoE dispatch (CUDA fast path).
    ///
    /// One `grouped_gemm` over the stacked `[E, 2*inter, hidden]` gate_up
    /// weight handles all experts in a single kernel, then the activation is
    /// applied to the split `[up|gate]` halves, then `fused_moe_gemm` applies
    /// the down projection with topk-weighted accumulation. This replaces a sequential
    /// `for t { for k in 0..topk { two tiny matmuls } }` loop that did two
    /// D2H syncs per layer via `to_vec2()` and launched ~`n_tokens * topk * 2`
    /// tiny GEMMs — catastrophic for a 40-layer 256-expert top-8 model.
    fn forward_composed(
        &self,
        ops: &dyn crate::ops::Ops,
        xs: &Tensor,
        topk_weights: &Tensor,
        experts_per_tok: &Tensor,
        n_tokens: usize,
        hidden_dim: usize,
    ) -> Result<Tensor> {
        let flat = experts_per_tok.flatten_all()?;
        let (sorted_expert_ids, sorted_token_ids) =
            if let Some(result) = ops.moe_sort_experts(&flat) {
                result?
            } else {
                sort_expert_assignments(experts_per_tok, xs.device())?
            };

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

        // Weights are in [up|gate] order (swapped for CUTLASS Swiglu at load time).
        // silu(gate) * up: gate is second half, up is first half.
        let inter = self.moe_intermediate_size;
        let down_input = {
            let up = gate_up.narrow(D::Minus1, 0, inter)?.contiguous()?;
            let gate = gate_up.narrow(D::Minus1, inter, inter)?.contiguous()?;
            ops.silu_mul(&gate, &up)?
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
}

/// Sort topk expert assignments so that all tokens routed to the same expert
/// are contiguous in the output. Returns `(sorted_expert_ids, sorted_token_ids)`
/// both as flat `[num_tokens * topk]` u32 tensors. The `sorted_token_ids[i]`
/// gives the original flat assignment index (token * topk + k) for the i-th
/// entry in the sorted expert stream, which is what `grouped_gemm` expects.
fn sort_expert_assignments(experts_per_tok: &Tensor, device: &Device) -> Result<(Tensor, Tensor)> {
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
    if let Some(out) = deltanet_varlen_grouped_zero(gdn, packed, seq_lens, ops)? {
        return Ok(out);
    }

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

fn deltanet_varlen_grouped_zero(
    gdn: &mut Qwen3_5GatedDeltaNet,
    packed: &Tensor,
    seq_lens: &[usize],
    ops: &dyn crate::ops::Ops,
) -> Result<Option<Tensor>> {
    if seq_lens.len() < 2 || !packed.device().is_cuda() {
        return Ok(None);
    }
    if gdn.head_k_dim != 128 || gdn.head_v_dim != 128 || gdn.head_k_dim != gdn.head_v_dim {
        return Ok(None);
    }

    let mut offsets = Vec::with_capacity(seq_lens.len());
    let mut off = 0usize;
    for &len in seq_lens {
        offsets.push(off);
        off += len;
    }

    let mut prefill_by_len: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (idx, &len) in seq_lens.iter().enumerate() {
        prefill_by_len.entry(len).or_default().push(idx);
    }
    if !prefill_by_len
        .iter()
        .any(|(&seq_len, indices)| seq_len > 1 && indices.len() > 1)
    {
        return Ok(None);
    }

    if prefill_by_len.len() == 1 {
        let (&seq_len, indices) = prefill_by_len.iter().next().unwrap();
        if seq_len > 1 && indices.len() > 1 {
            let mut flattened = Vec::new();
            let mut start = 0usize;
            while start < indices.len() {
                let remaining = indices.len() - start;
                let take = grouped_prefill_take_with(remaining, MAX_ZERO_GROUPED_PREFILL_BATCH);
                let chunk = &indices[start..start + take];
                let Some(out) =
                    deltanet_prefill_group_fused_zero(gdn, packed, &offsets, seq_len, chunk, ops)?
                else {
                    gdn.clear_state();
                    return Ok(None);
                };
                flattened.push(flatten_grouped_prefill_output(out, take, seq_len)?);
                start += take;
            }
            gdn.clear_state();
            return Ok(Some(if flattened.len() == 1 {
                flattened.pop().unwrap()
            } else {
                Tensor::cat(&flattened, 0)?
            }));
        }
    }

    let mut outputs: Vec<Option<Tensor>> = vec![None; seq_lens.len()];
    for (&seq_len, indices) in prefill_by_len.iter() {
        let mut start = 0usize;
        while start < indices.len() {
            let remaining = indices.len() - start;
            let take = grouped_prefill_take_with(remaining, MAX_ZERO_GROUPED_PREFILL_BATCH);
            if take == 1 || seq_len <= 1 {
                let req_idx = indices[start];
                gdn.clear_state();
                let seq = packed.narrow(0, offsets[req_idx], seq_len)?.unsqueeze(0)?;
                outputs[req_idx] = Some(gdn.forward(&seq, 0, ops)?.squeeze(0)?);
                gdn.clear_state();
                start += 1;
                continue;
            }

            let chunk = &indices[start..start + take];
            let Some(out) =
                deltanet_prefill_group_fused_zero(gdn, packed, &offsets, seq_len, chunk, ops)?
            else {
                gdn.clear_state();
                return Ok(None);
            };
            for (group_pos, &req_idx) in chunk.iter().enumerate() {
                outputs[req_idx] = Some(out.get(group_pos)?);
            }
            start += take;
        }
    }

    let mut ordered = Vec::with_capacity(seq_lens.len());
    for item in outputs {
        let Some(t) = item else {
            gdn.clear_state();
            return Ok(None);
        };
        ordered.push(t);
    }
    gdn.clear_state();
    Ok(Some(Tensor::cat(&ordered, 0)?))
}

fn grouped_prefill_take(remaining: usize) -> usize {
    grouped_prefill_take_with(remaining, MAX_GROUPED_PREFILL_BATCH)
}

fn grouped_prefill_take_with(remaining: usize, max_batch: usize) -> usize {
    if remaining == 1 {
        1
    } else if remaining <= max_batch {
        remaining
    } else if remaining - max_batch == 1 {
        max_batch - 1
    } else {
        max_batch
    }
}

fn grouped_prefill_input(
    packed: &Tensor,
    offsets: &[usize],
    seq_len: usize,
    indices: &[usize],
) -> Result<Tensor> {
    let batch = indices.len();
    if batch == 0 {
        crate::tensor::bail!("grouped_prefill_input called with empty batch")
    }

    let first_idx = indices[0];
    let first_offset = offsets[first_idx];
    let contiguous_group = indices
        .iter()
        .enumerate()
        .all(|(group_pos, &req_idx)| offsets[req_idx] == first_offset + group_pos * seq_len);

    if contiguous_group && packed.is_contiguous() {
        let (_, hidden) = packed.dims2()?;
        return packed
            .narrow(0, first_offset, batch * seq_len)?
            .reshape((batch, seq_len, hidden));
    }

    let mut seqs = Vec::with_capacity(batch);
    for &req_idx in indices {
        seqs.push(packed.narrow(0, offsets[req_idx], seq_len)?);
    }
    Tensor::stack(&seqs, 0)?.contiguous()
}

fn flatten_grouped_prefill_output(out: Tensor, batch: usize, seq_len: usize) -> Result<Tensor> {
    let (out_batch, out_seq_len, hidden) = out.dims3()?;
    debug_assert_eq!(out_batch, batch);
    debug_assert_eq!(out_seq_len, seq_len);
    out.reshape((batch * seq_len, hidden))
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
    slot_ids_gpu: Option<&Tensor>,
    state_is_zero: Option<&[bool]>,
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
        (16, 16, 128, 128)   // MHA H=16
        | (32, 32, 128, 128) // MHA H=32
        | (64, 64, 128, 128) // MHA H=64
        | (16, 32, 128, 128) // GQA H=16/HV=32 (Qwen3.5-35B-A3B)
    );
    let fused_eligible = all_decode
        && packed.device().is_cuda()
        && fused_supported
        && ops.kda_decode_available_for(
            slot_ids.len(),
            gdn.num_k_heads,
            gdn.num_v_heads,
            gdn.head_k_dim,
            gdn.head_v_dim,
        );
    if fused_eligible {
        if let Some(out) = deltanet_decode_batched_fused(
            gdn,
            packed,
            pool,
            slot_ids,
            slot_ids_gpu,
            dn_layer_idx,
            ops,
        )? {
            return Ok(out);
        }
    }
    if packed.device().is_cuda() && fused_supported {
        if let Some(out) = deltanet_mixed_grouped_fused(
            gdn,
            packed,
            seq_lens,
            pool,
            slot_ids,
            state_is_zero,
            dn_layer_idx,
            ops,
        )? {
            return Ok(out);
        }
    }

    let mut outputs = Vec::new();
    let mut offset = 0usize;
    for (i, &len) in seq_lens.iter().enumerate() {
        let slot = slot_ids[i] as usize;
        let starts_from_zero = state_is_zero
            .and_then(|flags| flags.get(i))
            .copied()
            .unwrap_or(false);

        if starts_from_zero {
            // New prefill request: the pool slot was just allocated/reset, so
            // avoid copying an all-zero recurrent row into layer scratch. The
            // GDN and conv paths already treat `None` as zero initial state.
            gdn.recurrent_state = None;
            gdn.conv_state = None;
        } else {
            // Load this request's state from the pool into the layer scratch.
            // Force contiguous so downstream ops (matmul, slice_set on write-back)
            // see a dense layout. `get(slot)` strips the leading max_slots dim.
            gdn.recurrent_state = Some(
                pool.recurrent_states[dn_layer_idx]
                    .get(slot)?
                    .contiguous()?,
            );
            gdn.conv_state = Some(pool.conv_states[dn_layer_idx].get(slot)?.contiguous()?);
        }

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

fn deltanet_mixed_grouped_fused(
    gdn: &mut Qwen3_5GatedDeltaNet,
    packed: &Tensor,
    seq_lens: &[usize],
    pool: &mut crate::deltanet_pool::DeltaNetPool,
    slot_ids: &[u32],
    state_is_zero: Option<&[bool]>,
    dn_layer_idx: usize,
    ops: &dyn crate::ops::Ops,
) -> Result<Option<Tensor>> {
    if seq_lens.is_empty() || !packed.device().is_cuda() {
        return Ok(None);
    }
    if gdn.head_k_dim != 128 || gdn.head_v_dim != 128 || gdn.head_k_dim != gdn.head_v_dim {
        return Ok(None);
    }

    let mut offsets = Vec::with_capacity(seq_lens.len());
    let mut off = 0usize;
    for &len in seq_lens {
        offsets.push(off);
        off += len;
    }

    let mut decode_indices = Vec::new();
    let mut prefill_by_len: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (idx, &len) in seq_lens.iter().enumerate() {
        if len == 1 {
            decode_indices.push(idx);
        } else {
            prefill_by_len.entry(len).or_default().push(idx);
        }
    }

    if decode_indices.is_empty() && prefill_by_len.is_empty() {
        return Ok(None);
    }
    if !decode_indices.is_empty()
        && !ops.kda_decode_available_for(
            decode_indices.len(),
            gdn.num_k_heads,
            gdn.num_v_heads,
            gdn.head_k_dim,
            gdn.head_v_dim,
        )
    {
        return Ok(None);
    }

    // Larger buckets are chunked so we still reduce the per-layer GDN/conv
    // launch count without feeding an oversized grouped prefill into
    // FlashInfer.
    if decode_indices.is_empty() && prefill_by_len.len() == 1 {
        let (&seq_len, indices) = prefill_by_len.iter().next().unwrap();
        if seq_len > 1 && indices.len() > 1 {
            let mut flattened = Vec::new();
            let mut start = 0usize;
            while start < indices.len() {
                let remaining = indices.len() - start;
                let take = grouped_prefill_take(remaining);
                let chunk = &indices[start..start + take];
                let out = if indices_all_state_zero(state_is_zero, chunk) {
                    deltanet_prefill_group_fused_zero_pooled(
                        gdn,
                        packed,
                        &offsets,
                        seq_len,
                        chunk,
                        pool,
                        slot_ids,
                        dn_layer_idx,
                        ops,
                    )?
                } else {
                    deltanet_prefill_group_fused(
                        gdn,
                        packed,
                        &offsets,
                        seq_len,
                        chunk,
                        pool,
                        slot_ids,
                        dn_layer_idx,
                        ops,
                    )?
                };
                let Some(out) = out else { return Ok(None) };
                flattened.push(flatten_grouped_prefill_output(out, take, seq_len)?);
                start += take;
            }
            return Ok(Some(if flattened.len() == 1 {
                flattened.pop().unwrap()
            } else {
                Tensor::cat(&flattened, 0)?
            }));
        }
    }

    let mut outputs: Vec<Option<Tensor>> = vec![None; seq_lens.len()];
    for (&seq_len, indices) in prefill_by_len.iter() {
        let mut start = 0usize;
        while start < indices.len() {
            let remaining = indices.len() - start;
            let take = grouped_prefill_take(remaining);
            if take == 1 {
                let req_idx = indices[start];
                outputs[req_idx] = Some(deltanet_single_pooled(
                    gdn,
                    packed,
                    offsets[req_idx],
                    seq_len,
                    pool,
                    slot_ids[req_idx],
                    dn_layer_idx,
                    ops,
                )?);
                start += 1;
                continue;
            }
            let chunk = &indices[start..start + take];
            let out = if indices_all_state_zero(state_is_zero, chunk) {
                deltanet_prefill_group_fused_zero_pooled(
                    gdn,
                    packed,
                    &offsets,
                    seq_len,
                    chunk,
                    pool,
                    slot_ids,
                    dn_layer_idx,
                    ops,
                )?
            } else {
                deltanet_prefill_group_fused(
                    gdn,
                    packed,
                    &offsets,
                    seq_len,
                    chunk,
                    pool,
                    slot_ids,
                    dn_layer_idx,
                    ops,
                )?
            };
            let Some(out) = out else {
                return Ok(None);
            };
            for (group_pos, &req_idx) in chunk.iter().enumerate() {
                outputs[req_idx] = Some(out.get(group_pos)?);
            }
            start += take;
        }
    }

    if !decode_indices.is_empty() {
        let mut rows = Vec::with_capacity(decode_indices.len());
        let mut decode_slots = Vec::with_capacity(decode_indices.len());
        for &req_idx in decode_indices.iter() {
            rows.push(packed.narrow(0, offsets[req_idx], 1)?.squeeze(0)?);
            decode_slots.push(slot_ids[req_idx]);
        }
        let decode_packed = Tensor::stack(&rows, 0)?.contiguous()?;
        let decode_slots_gpu =
            Tensor::from_vec(decode_slots.clone(), (decode_slots.len(),), packed.device())?;
        let out = deltanet_decode_batched_fused(
            gdn,
            &decode_packed,
            pool,
            &decode_slots,
            Some(&decode_slots_gpu),
            dn_layer_idx,
            ops,
        )?;
        let Some(out) = out else {
            return Ok(None);
        };
        for (group_pos, &req_idx) in decode_indices.iter().enumerate() {
            outputs[req_idx] = Some(out.narrow(0, group_pos, 1)?);
        }
    }

    let mut ordered = Vec::with_capacity(seq_lens.len());
    for item in outputs {
        let Some(t) = item else {
            return Ok(None);
        };
        ordered.push(t);
    }
    Ok(Some(Tensor::cat(&ordered, 0)?))
}

fn indices_all_state_zero(state_is_zero: Option<&[bool]>, indices: &[usize]) -> bool {
    let Some(flags) = state_is_zero else {
        return false;
    };
    indices
        .iter()
        .all(|&req_idx| flags.get(req_idx).copied().unwrap_or(false))
}

#[allow(clippy::too_many_arguments)]
fn deltanet_single_pooled(
    gdn: &mut Qwen3_5GatedDeltaNet,
    packed: &Tensor,
    offset: usize,
    seq_len: usize,
    pool: &mut crate::deltanet_pool::DeltaNetPool,
    slot_id: u32,
    dn_layer_idx: usize,
    ops: &dyn crate::ops::Ops,
) -> Result<Tensor> {
    let slot = slot_id as usize;
    gdn.recurrent_state = Some(
        pool.recurrent_states[dn_layer_idx]
            .get(slot)?
            .contiguous()?,
    );
    gdn.conv_state = Some(pool.conv_states[dn_layer_idx].get(slot)?.contiguous()?);

    let seq = packed.narrow(0, offset, seq_len)?.unsqueeze(0)?;
    let out = gdn.forward(&seq, 0, ops)?.squeeze(0)?;

    if let Some(ref state) = gdn.recurrent_state {
        let row = state.contiguous()?.unsqueeze(0)?.contiguous()?;
        pool.recurrent_states[dn_layer_idx].slice_set(&row, 0, slot)?;
    }
    if let Some(ref state) = gdn.conv_state {
        let row = state.contiguous()?.unsqueeze(0)?.contiguous()?;
        pool.conv_states[dn_layer_idx].slice_set(&row, 0, slot)?;
    }
    gdn.clear_state();
    Ok(out)
}

fn deltanet_prefill_group_fused_zero(
    gdn: &mut Qwen3_5GatedDeltaNet,
    packed: &Tensor,
    offsets: &[usize],
    seq_len: usize,
    indices: &[usize],
    ops: &dyn crate::ops::Ops,
) -> Result<Option<Tensor>> {
    let batch = indices.len();
    if batch == 0 || seq_len <= 1 {
        return Ok(None);
    }

    let bst = BatchState::no_lora();
    let x = grouped_prefill_input(packed, offsets, seq_len, indices)?; // [B, L, hidden]

    let qkv = gdn.in_proj_qkv.forward(&x, &bst, ops)?; // [B, L, conv_dim]
    let z = gdn.in_proj_z.forward(&x, &bst, ops)?; // [B, L, value_dim]
    let b_raw = gdn.in_proj_b.forward(&x, &bst, ops)?; // [B, L, HV]
    let a_raw = gdn.in_proj_a.forward(&x, &bst, ops)?; // [B, L, HV]

    let qkv_conv = if let Some(conv) = ops.causal_conv1d_fn_channellast(
        &qkv,
        &gdn.conv_weight,
        None,
        None,
        /*silu_activation=*/ true,
    ) {
        conv?
    } else {
        let x_t = qkv.transpose(1, 2)?.contiguous()?; // [B, conv_dim, L]
        let Some(conv) = ops.causal_conv1d_fn(
            &x_t,
            &gdn.conv_weight,
            None,
            None,
            /*silu_activation=*/ true,
        ) else {
            return Ok(None);
        };
        conv?.transpose(1, 2)?.contiguous()? // [B, L, conv_dim]
    };

    let total_tokens = batch * seq_len;
    let mixed = qkv_conv.reshape((total_tokens, gdn.conv_dim))?;
    let a_2d = a_raw.reshape((total_tokens, gdn.num_v_heads))?;
    let b_2d = b_raw.reshape((total_tokens, gdn.num_v_heads))?;

    let Some(prep) = ops.gdn_post_conv(
        &mixed,
        &a_2d,
        &b_2d,
        &gdn.a_log_f32,
        &gdn.dt_bias_f32,
        gdn.num_k_heads,
        gdn.num_v_heads,
        gdn.head_k_dim,
    ) else {
        return Ok(None);
    };
    let (q_bf16, k_bf16, v_bf16, alpha, beta) = prep?;

    let mut cu = Vec::with_capacity(batch + 1);
    for i in 0..=batch {
        cu.push((i * seq_len) as i64);
    }
    let cu_seqlens = Tensor::from_vec(cu, (batch + 1,), packed.device())?;

    let scale = (gdn.head_k_dim as f32).powf(-0.5);
    let Some(result) = ops.gdn_prefill_varlen(
        &q_bf16,
        &k_bf16,
        &v_bf16,
        &alpha,
        &beta,
        &cu_seqlens,
        None,
        scale,
    ) else {
        return Ok(None);
    };
    let (out, _) = result?;

    let out = out.reshape((batch, seq_len, gdn.value_dim))?;
    let normed = gdn.norm.forward(&out, &z.contiguous()?, ops)?;
    let projected = gdn.out_proj.forward(&normed, &bst, ops)?;
    Ok(Some(projected))
}

#[allow(clippy::too_many_arguments)]
fn deltanet_prefill_group_fused_zero_pooled(
    gdn: &mut Qwen3_5GatedDeltaNet,
    packed: &Tensor,
    offsets: &[usize],
    seq_len: usize,
    indices: &[usize],
    pool: &mut crate::deltanet_pool::DeltaNetPool,
    slot_ids: &[u32],
    dn_layer_idx: usize,
    ops: &dyn crate::ops::Ops,
) -> Result<Option<Tensor>> {
    let batch = indices.len();
    if batch == 0 || seq_len <= 1 {
        return Ok(None);
    }

    let bst = BatchState::no_lora();
    let x = grouped_prefill_input(packed, offsets, seq_len, indices)?; // [B, L, hidden]

    let qkv = gdn.in_proj_qkv.forward(&x, &bst, ops)?; // [B, L, conv_dim]
    let z = gdn.in_proj_z.forward(&x, &bst, ops)?; // [B, L, value_dim]
    let b_raw = gdn.in_proj_b.forward(&x, &bst, ops)?; // [B, L, HV]
    let a_raw = gdn.in_proj_a.forward(&x, &bst, ops)?; // [B, L, HV]

    let qkv_conv = if let Some(conv) = ops.causal_conv1d_fn_channellast(
        &qkv,
        &gdn.conv_weight,
        None,
        None,
        /*silu_activation=*/ true,
    ) {
        conv?
    } else {
        let x_t = qkv.transpose(1, 2)?.contiguous()?; // [B, conv_dim, L]
        let Some(conv) = ops.causal_conv1d_fn(
            &x_t,
            &gdn.conv_weight,
            None,
            None,
            /*silu_activation=*/ true,
        ) else {
            return Ok(None);
        };
        conv?.transpose(1, 2)?.contiguous()? // [B, L, conv_dim]
    };

    let total_tokens = batch * seq_len;
    let mixed = qkv_conv.reshape((total_tokens, gdn.conv_dim))?;
    let a_2d = a_raw.reshape((total_tokens, gdn.num_v_heads))?;
    let b_2d = b_raw.reshape((total_tokens, gdn.num_v_heads))?;

    let Some(prep) = ops.gdn_post_conv(
        &mixed,
        &a_2d,
        &b_2d,
        &gdn.a_log_f32,
        &gdn.dt_bias_f32,
        gdn.num_k_heads,
        gdn.num_v_heads,
        gdn.head_k_dim,
    ) else {
        return Ok(None);
    };
    let (q_bf16, k_bf16, v_bf16, alpha, beta) = prep?;

    let mut cu = Vec::with_capacity(batch + 1);
    for i in 0..=batch {
        cu.push((i * seq_len) as i64);
    }
    let cu_seqlens = Tensor::from_vec(cu, (batch + 1,), packed.device())?;

    let scale = (gdn.head_k_dim as f32).powf(-0.5);
    let Some(result) = ops.gdn_prefill_varlen(
        &q_bf16,
        &k_bf16,
        &v_bf16,
        &alpha,
        &beta,
        &cu_seqlens,
        None,
        scale,
    ) else {
        return Ok(None);
    };
    let (out, final_state) = result?;

    let pad_len = gdn.conv_kernel - 1;
    let pool_conv = &pool.conv_states[dn_layer_idx];
    let conv_tail = if seq_len >= pad_len {
        qkv.narrow(1, seq_len - pad_len, pad_len)?
            .transpose(1, 2)?
            .contiguous()?
    } else {
        let mut rows = Vec::with_capacity(batch);
        for group_pos in 0..batch {
            let tail = qkv.get(group_pos)?.transpose(0, 1)?.contiguous()?;
            let zeros =
                Tensor::zeros((gdn.conv_dim, pad_len - seq_len), qkv.dtype(), qkv.device())?;
            rows.push(Tensor::cat(&[&zeros, &tail], 1)?);
        }
        Tensor::stack(&rows, 0)?.contiguous()?
    };

    let pool_state = &pool.recurrent_states[dn_layer_idx];
    for (group_pos, &req_idx) in indices.iter().enumerate() {
        let conv_row = conv_tail.get(group_pos)?.unsqueeze(0)?.contiguous()?;
        pool_conv.slice_set(&conv_row, 0, slot_ids[req_idx] as usize)?;

        let state_row = final_state.get(group_pos)?.unsqueeze(0)?.contiguous()?;
        pool_state.slice_set(&state_row, 0, slot_ids[req_idx] as usize)?;
    }

    let out = out.reshape((batch, seq_len, gdn.value_dim))?;
    let normed = gdn.norm.forward(&out, &z.contiguous()?, ops)?;
    let projected = gdn.out_proj.forward(&normed, &bst, ops)?;
    Ok(Some(projected))
}

#[allow(clippy::too_many_arguments)]
fn deltanet_prefill_group_fused(
    gdn: &mut Qwen3_5GatedDeltaNet,
    packed: &Tensor,
    offsets: &[usize],
    seq_len: usize,
    indices: &[usize],
    pool: &mut crate::deltanet_pool::DeltaNetPool,
    slot_ids: &[u32],
    dn_layer_idx: usize,
    ops: &dyn crate::ops::Ops,
) -> Result<Option<Tensor>> {
    let batch = indices.len();
    if batch == 0 || seq_len <= 1 {
        return Ok(None);
    }

    let bst = BatchState::no_lora();
    let x = grouped_prefill_input(packed, offsets, seq_len, indices)?; // [B, L, hidden]

    let qkv = gdn.in_proj_qkv.forward(&x, &bst, ops)?; // [B, L, conv_dim]
    let z = gdn.in_proj_z.forward(&x, &bst, ops)?; // [B, L, value_dim]
    let b_raw = gdn.in_proj_b.forward(&x, &bst, ops)?; // [B, L, HV]
    let a_raw = gdn.in_proj_a.forward(&x, &bst, ops)?; // [B, L, HV]

    let pad_len = gdn.conv_kernel - 1;
    let x_t = qkv.transpose(1, 2)?.contiguous()?; // [B, conv_dim, L]
    let pool_conv = &pool.conv_states[dn_layer_idx];
    let mut prefixes = Vec::with_capacity(batch);
    for &req_idx in indices {
        prefixes.push(pool_conv.get(slot_ids[req_idx] as usize)?.contiguous()?);
    }
    let prefix = Tensor::stack(&prefixes, 0)?.contiguous()?; // [B, conv_dim, W-1]
    let x_padded = Tensor::cat(&[&prefix, &x_t], 2)?.contiguous()?;

    let Some(conv) = ops.causal_conv1d_fn(
        &x_padded,
        &gdn.conv_weight,
        None,
        None,
        /*silu_activation=*/ true,
    ) else {
        return Ok(None);
    };
    let conv = conv?;
    let qkv_conv = conv
        .narrow(2, pad_len, seq_len)?
        .transpose(1, 2)?
        .contiguous()?; // [B, L, conv_dim]

    let total_tokens = batch * seq_len;
    let mixed = qkv_conv.reshape((total_tokens, gdn.conv_dim))?;
    let a_2d = a_raw.reshape((total_tokens, gdn.num_v_heads))?;
    let b_2d = b_raw.reshape((total_tokens, gdn.num_v_heads))?;

    let Some(prep) = ops.gdn_post_conv(
        &mixed,
        &a_2d,
        &b_2d,
        &gdn.a_log_f32,
        &gdn.dt_bias_f32,
        gdn.num_k_heads,
        gdn.num_v_heads,
        gdn.head_k_dim,
    ) else {
        return Ok(None);
    };
    let (q_bf16, k_bf16, v_bf16, alpha, beta) = prep?;

    let mut cu = Vec::with_capacity(batch + 1);
    for i in 0..=batch {
        cu.push((i * seq_len) as i64);
    }
    let cu_seqlens = Tensor::from_vec(cu, (batch + 1,), packed.device())?;

    let pool_state = &pool.recurrent_states[dn_layer_idx];
    let mut states = Vec::with_capacity(batch);
    for &req_idx in indices {
        states.push(pool_state.get(slot_ids[req_idx] as usize)?.contiguous()?);
    }
    let initial_state = Tensor::stack(&states, 0)?.contiguous()?;

    let scale = (gdn.head_k_dim as f32).powf(-0.5);
    let Some(result) = ops.gdn_prefill_varlen(
        &q_bf16,
        &k_bf16,
        &v_bf16,
        &alpha,
        &beta,
        &cu_seqlens,
        Some(&initial_state),
        scale,
    ) else {
        return Ok(None);
    };
    let (out, final_state) = result?;

    for (group_pos, &req_idx) in indices.iter().enumerate() {
        let x_row = x_t.get(group_pos)?;
        let new_conv_state = if seq_len >= pad_len {
            x_row.narrow(1, seq_len - pad_len, pad_len)?.contiguous()?
        } else {
            let old = prefix
                .get(group_pos)?
                .narrow(1, seq_len, pad_len - seq_len)?;
            Tensor::cat(&[&old, &x_row], 1)?.contiguous()?
        };
        let conv_row = new_conv_state.unsqueeze(0)?.contiguous()?;
        pool_conv.slice_set(&conv_row, 0, slot_ids[req_idx] as usize)?;

        let row = final_state.get(group_pos)?.unsqueeze(0)?.contiguous()?;
        pool_state.slice_set(&row, 0, slot_ids[req_idx] as usize)?;
    }

    let out = out.reshape((batch, seq_len, gdn.value_dim))?;
    let normed = gdn.norm.forward(&out, &z.contiguous()?, ops)?;
    let projected = gdn.out_proj.forward(&normed, &bst, ops)?;
    Ok(Some(projected))
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
    slot_ids_gpu: Option<&Tensor>,
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
    if !ops.kda_decode_available_for(
        n,
        gdn.num_k_heads,
        gdn.num_v_heads,
        gdn.head_k_dim,
        gdn.head_v_dim,
    ) {
        return Ok(None);
    }
    // `packed` is `[total_tokens, hidden]` and all seq_lens == 1, so
    // `total_tokens == n`. Fold the batch dim into the time dim for the
    // projections: `gdn.in_proj_*` expect the existing `[B, L, hidden]`
    // layout, so wrap `packed` as `[1, N, hidden]`.
    let bst = BatchState::no_lora();
    let x = packed.unsqueeze(0)?; // [1, N, hidden]

    let qkv = gdn.in_proj_qkv.forward(&x, &bst, ops)?; // [1, N, key*2+val]
    let z = gdn.in_proj_z.forward(&x, &bst, ops)?; // [1, N, value_dim]
    let b_raw_full = gdn.in_proj_b.forward(&x, &bst, ops)?; // [1, N, HV]
    let a_raw_full = gdn.in_proj_a.forward(&x, &bst, ops)?; // [1, N, HV]

    let qkv_for_conv = qkv; // [1, N, conv_dim]
    let qkv_for_conv_2d = qkv_for_conv.squeeze(0)?; // [N, conv_dim]

    // ── Batched conv1d_decode via pool-indexed kernel ────────────────
    // Uses `conv_state_indices` to let the kernel index directly into
    // pool.conv_states[layer] (shape `[pool_size, conv_dim, state_len]`),
    // eliminating the per-request CPU loop. One kernel launch for all N
    // sequences, CUDA-graph compatible.
    let device = packed.device();
    // Use pre-allocated GPU tensor when available (CUDA graph compatible).
    // Otherwise create on-the-fly (eager path, allocates memory).
    let slot_ids_owned: Option<Tensor>;
    let slot_ids_gpu: &Tensor = match slot_ids_gpu {
        Some(t) => t,
        None => {
            slot_ids_owned = Some(Tensor::from_vec(slot_ids.to_vec(), (n,), device)?);
            slot_ids_owned.as_ref().unwrap()
        }
    };
    let qkv_conv = {
        let x_bd = qkv_for_conv_2d.contiguous()?; // [N, conv_dim]
        let pool_conv = &pool.conv_states[dn_layer_idx]; // [pool_size, conv_dim, state_len]
        // causal_conv1d_update wants x: [N, dim, 1], conv_state: [pool, dim, sl]
        // We need to check if the fused kernel path is available
        if let Some(res) = ops.causal_conv1d_update(
            &x_bd,            // [N, conv_dim]
            pool_conv,        // [pool_size, conv_dim, state_len]
            &gdn.conv_weight, // [conv_dim, width]
            None,             // bias
            /*silu_activation=*/ true,
            Some(slot_ids_gpu), // conv_state_indices [N] U32 (reinterpreted as I32)
        ) {
            res? // [N, conv_dim]
        } else {
            // Fallback: per-request sequential (CPU loop, non-graph-compatible)
            let mut conv_outs: Vec<Tensor> = Vec::with_capacity(n);
            for (i, &slot) in slot_ids.iter().enumerate() {
                let slot_us = slot as usize;
                gdn.conv_state = Some(pool_conv.get(slot_us)?.contiguous()?);
                let x_row = qkv_for_conv_2d.get(i)?.contiguous()?;
                let out = gdn.conv1d_decode(&x_row, ops)?;
                conv_outs.push(out);
                if let Some(ref state) = gdn.conv_state {
                    let row = state.contiguous()?.unsqueeze(0)?.contiguous()?;
                    pool_conv.slice_set(&row, 0, slot_us)?;
                }
            }
            gdn.conv_state = None;
            Tensor::stack(&conv_outs, 0)?
        }
    };

    // Split into q, k, v with head layout.
    let q_flat = qkv_conv.narrow(D::Minus1, 0, gdn.key_dim)?;
    let k_flat = qkv_conv.narrow(D::Minus1, gdn.key_dim, gdn.key_dim)?;
    let v_flat = qkv_conv.narrow(D::Minus1, gdn.key_dim * 2, gdn.value_dim)?;
    let q_nhk = q_flat
        .reshape((n, gdn.num_k_heads, gdn.head_k_dim))?
        .contiguous()?;
    let k_nhk = k_flat
        .reshape((n, gdn.num_k_heads, gdn.head_k_dim))?
        .contiguous()?;
    let v_nhv = v_flat
        .reshape((n, gdn.num_v_heads, gdn.head_v_dim))?
        .contiguous()?;

    // Scalar-per-head a/b drop the batch-of-1 dim.
    let a_2d = a_raw_full.squeeze(0)?.contiguous()?; // [N, HV]
    let b_2d = b_raw_full.squeeze(0)?.contiguous()?; // [N, HV]

    let decode_out = match ops.kda_decode_batched(
        &q_nhk,
        &k_nhk,
        &v_nhv,
        &a_2d,
        &b_2d,
        &gdn.a_log_f32,
        &gdn.dt_bias_f32_expanded,
        &pool.recurrent_states[dn_layer_idx],
        slot_ids_gpu,
    ) {
        Some(Ok(o)) => o, // [N, HV, V]
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
            token_mixer,
            mlp,
            ln1_weight,
            ln2_weight,
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

        let mlp = self.mlp.forward(ops, &h2)?;
        ops.add_or_fused(&x_res, &mlp)
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
        slot_ids_gpu: Option<&Tensor>,
        state_is_zero: Option<&[bool]>,
        dn_layer_idx: usize,
    ) -> Result<Tensor> {
        let h = ops.rms_norm(x, &self.ln1_weight, self.rms_norm_eps)?;
        let h = match &mut self.token_mixer {
            TokenMixer::LinearAttention(gdn) => deltanet_varlen_pooled(
                gdn,
                &h,
                seq_lens,
                pool,
                slot_ids,
                slot_ids_gpu,
                state_is_zero,
                dn_layer_idx,
                ops,
            )?,
            TokenMixer::FullAttention(_) => {
                crate::tensor::bail!(
                    "forward_with_paged_prefix_pooled called on FullAttention layer"
                )
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

    fn forward_with_cache(
        &mut self,
        ops: &dyn crate::ops::Ops,
        x: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
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

        Ok(Self {
            embed_tokens,
            layers,
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
                                ctx.deltanet_slots_gpu,
                                ctx.deltanet_state_is_zero,
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
        ctx.ops
            .rms_norm(&h, &self.norm_weight, self.rms_norm_eps as f32)
    }

    fn clear_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }

    pub(super) fn forward_with_cache(
        &mut self,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
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
            &BatchState::no_lora(),
            ctx.ops,
        )
    }

    /// Cached forward: returns logits `[L, vocab_size]` for all input tokens.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
        let hidden = self.model.forward_with_cache(input_ids, position_offset)?;
        self.lm_head.forward(
            &hidden,
            &BatchState::no_lora(),
            crate::ops::select_ops(hidden.device()),
        )
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
        self.lm_head.forward(
            hidden,
            &BatchState::no_lora(),
            crate::ops::select_ops(hidden.device()),
        )
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
    use crate::models::registry::{ArchSpec, ParsedModelConfig, candle_model_err, parse_json};

    const ARCHITECTURE_ALIASES: &[&str] = &["Qwen3_5", "Qwen35", "Qwen3_5Moe", "Qwen35Moe"];
    const MODEL_TYPE_ALIASES: &[&str] =
        &["qwen3_5_text", "qwen3_5", "qwen3_5_moe_text", "qwen3_5_moe"];
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
    inventory::submit!(crate::models::registry::ArchSpecEntry::new(
        &QWEN3_5_ARCH_SPEC
    ));

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
                supports_cuda_graph: device.is_cuda() && is_safetensors,
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
    use crate::models::commons::Linear;
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
                    // DeltaNet layer. We keep the four input
                    // projections split rather than fusing into one
                    // matmul — see the comment in
                    // `Qwen3_5GatedDeltaNet::new` (safetensors loader)
                    // for the GEMM tile-alignment rationale.
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
                    let conv_weight = if conv_weight_raw.dim(0)? == config.linear_conv_kernel_dim {
                        conv_weight_raw.t()?.contiguous()?
                    } else {
                        conv_weight_raw
                    };

                    let dt_bias =
                        Self::load_tensor(&ct, reader, &format!("{prefix}.ssm_dt.bias"), device)?;

                    // GGUF stores -exp(A_log); convert back: A_log = ln(-ssm_a)
                    let ssm_a = Self::load_tensor(&ct, reader, &format!("{prefix}.ssm_a"), device)?;
                    let a_log = ssm_a.neg()?.log()?;
                    let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;
                    let dt_bias_f32_expanded = dt_bias_f32
                        .reshape((config.linear_num_value_heads, 1))?
                        .broadcast_as((config.linear_num_value_heads, config.linear_key_head_dim))?
                        .contiguous()?;
                    let a_log_f32 = if a_log.dtype() == DType::F32 {
                        a_log.clone()
                    } else {
                        a_log.to_dtype(DType::F32)?
                    };

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
                        dt_bias_f32,
                        dt_bias_f32_expanded,
                        a_log_f32,
                        prefill_cu_seqlens: None,
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
                    let q_proj =
                        Self::load_linear(&ct, reader, &format!("{prefix}.attn_q.weight"), device)?;
                    let k_proj =
                        Self::load_linear(&ct, reader, &format!("{prefix}.attn_k.weight"), device)?;
                    let v_proj =
                        Self::load_linear(&ct, reader, &format!("{prefix}.attn_v.weight"), device)?;
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

                    let rope = build_rope(dtype)?;

                    super::TokenMixer::FullAttention(super::Qwen3_5Attention {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
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
            let val = md.get(key).ok_or_else(|| {
                crate::tensor::Error::Msg(format!("missing GGUF metadata: {key}"))
            })?;
            Ok(val.to_u32().map(|v| v as usize)?)
        };

        let get_u32_or = |key: &str, default: usize| -> usize {
            md.get(key)
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or_else(|| {
                    tracing::warn!("Qwen3.5 GGUF: '{key}' not found, using default: {default}");
                    default
                })
        };

        let get_f32 = |key: &str| -> Result<f64> {
            let val = md.get(key).ok_or_else(|| {
                crate::tensor::Error::Msg(format!("missing GGUF metadata: {key}"))
            })?;
            Ok(val.to_f32().map(|v| v as f64)?)
        };

        let get_f32_or = |key: &str, default: f64| -> f64 {
            md.get(key)
                .and_then(|v| v.to_f32().ok())
                .map(|v| v as f64)
                .unwrap_or_else(|| {
                    tracing::warn!("Qwen3.5 GGUF: '{key}' not found, using default: {default}");
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
        let intermediate_size =
            get_u32_or(&format!("{arch}.feed_forward_length"), default_intermediate);

        // SSM / DeltaNet specific metadata
        let ssm_d_conv = get_u32_or(&format!("{arch}.ssm.conv_kernel"), 4);
        let ssm_d_inner = get_u32_or(&format!("{arch}.ssm.inner_size"), hidden_size);
        let ssm_d_state = get_u32_or(&format!("{arch}.ssm.state_size"), 128);
        let ssm_dt_rank = get_u32_or(&format!("{arch}.ssm.time_step_rank"), 16);
        let ssm_n_group = get_u32_or(&format!("{arch}.ssm.group_count"), 16);
        let full_attention_interval = get_u32_or(&format!("{arch}.full_attention_interval"), 4);

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
        let partial_rotary_factor = get_f32_or(&format!("{arch}.rope.partial_rotary_factor"), 0.25);

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
