use std::sync::Arc;

use crate::loading::var_builder::VarBuilder;
use crate::models::commons::embedding::Embedding;
use crate::tensor::{DType, Device, Module, Result, Tensor};
use serde::Deserialize;

use crate::models::commons::{BatchState, Linear, RmsNorm};
use crate::models::commons::{PagedKvBatchContext, PagedKvContext};
use crate::ops::{MaskType, PagedParams, VarlenParams};

use crate::models::model_config;

// ── Gemma4 Config ────────────────────────────────────────────────────────

/// Per-layer-type RoPE parameters (deserialized from `rope_parameters` dict).
#[derive(Debug, Clone, Deserialize)]
pub struct RopeLayerParams {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_rope_type() -> String {
    "default".into()
}
fn default_partial_rotary_factor() -> f64 {
    1.0
}

model_config! {
    /// Gemma4 text model configuration
    pub struct Gemma4Config("Gemma4") {
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
            rope_parameters: Option<std::collections::HashMap<String, RopeLayerParams>>,
            global_head_dim: Option<usize>,
            enable_moe_block: bool,
            num_experts: Option<usize>,
            top_k_experts: Option<usize>,
            moe_intermediate_size: Option<usize>,
            expert_intermediate_size: Option<usize>,
            num_kv_shared_layers: usize,
            attention_k_eq_v: bool,
            num_global_key_value_heads: Option<usize>,
            hidden_size_per_layer_input: Option<usize>,
            vocab_size_per_layer_input: Option<usize>,
            use_double_wide_mlp: bool,
        }
        warn_default {
            vocab_size: usize = 262144,
            max_position_embeddings: usize = 32768,
            rms_norm_eps: f64 = 1e-6,
            rope_theta: f64 = 10_000.0,
            rope_local_base_freq: f64 = 10_000.0,
            tie_word_embeddings: bool = true,
        }
    }
}

impl Gemma4Config {
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    fn layer_type(&self, layer_idx: usize) -> &str {
        self.layer_types
            .as_ref()
            .and_then(|lt| lt.get(layer_idx))
            .map(String::as_str)
            .unwrap_or("full_attention")
    }

    fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.layer_type(layer_idx) == "sliding_attention"
    }

    fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        if !self.is_sliding_layer(layer_idx) {
            self.global_head_dim.unwrap_or(self.head_dim)
        } else {
            self.head_dim
        }
    }

    fn rope_theta_for_layer(&self, layer_idx: usize) -> f64 {
        let lt = self.layer_type(layer_idx);
        if let Some(params) = self.rope_parameters.as_ref().and_then(|rp| rp.get(lt)) {
            params.rope_theta
        } else if self.is_sliding_layer(layer_idx) {
            self.rope_local_base_freq
        } else {
            self.rope_theta
        }
    }

    fn partial_rotary_factor_for_layer(&self, layer_idx: usize) -> f64 {
        let lt = self.layer_type(layer_idx);
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.get(lt))
            .map(|p| p.partial_rotary_factor)
            .unwrap_or(1.0)
    }

    fn num_kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        let is_full = !self.is_sliding_layer(layer_idx);
        if is_full && self.attention_k_eq_v {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    fn is_kv_shared_layer(&self, layer_idx: usize) -> bool {
        if self.num_kv_shared_layers == 0 {
            return false;
        }
        let first = self
            .num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers);
        layer_idx >= first && first > 0
    }

    /// For a KV-shared layer, find the source layer (last non-shared layer of the same type).
    /// Returns None for non-shared layers.
    fn kv_sharing_source(&self, layer_idx: usize) -> Option<usize> {
        if !self.is_kv_shared_layer(layer_idx) {
            return None;
        }
        let first = self
            .num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers);
        let current_type = self.layer_type(layer_idx);
        (0..first)
            .rev()
            .find(|&i| self.layer_type(i) == current_type)
    }

    fn layer_intermediate_size(&self, layer_idx: usize) -> usize {
        let double = self.use_double_wide_mlp && self.is_kv_shared_layer(layer_idx);
        self.intermediate_size * if double { 2 } else { 1 }
    }

    fn ple_dim(&self) -> usize {
        self.hidden_size_per_layer_input.unwrap_or(0)
    }

    fn ple_vocab_size(&self) -> usize {
        self.vocab_size_per_layer_input.unwrap_or(self.vocab_size)
    }

    fn moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size
            .or(self.expert_intermediate_size)
            .unwrap_or(self.intermediate_size)
    }
}

// ── Rotary Embedding ─────────────────────────────────────────────────────

/// Gemma4 proportional RoPE: frequency exponent uses head_size as denominator,
/// and partial_rotary_factor controls how many dimensions are rotated.
/// Non-rotated dimensions get cos=1, sin=0 (identity rotation).
#[derive(Debug, Clone)]
struct Gemma4RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Gemma4RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        dev: &Device,
    ) -> Result<Self> {
        let rotary_dim = ((head_dim as f64 * partial_rotary_factor) as usize / 2) * 2;
        let rope_angles = rotary_dim / 2;
        let nope_angles = (head_dim / 2).saturating_sub(rope_angles);

        // Gemma4 proportional RoPE: denominator is head_dim (not rotary_dim)
        let mut inv_freq: Vec<f32> = (0..rope_angles)
            .map(|i| 1f32 / rope_theta.powf((2 * i) as f64 / head_dim as f64) as f32)
            .collect();

        // Zero-pad for non-rotated dimensions (identity: cos=1, sin=0)
        for _ in 0..nope_angles {
            inv_freq.push(0.0);
        }

        let half_dim = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.broadcast_mul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }
}

// ── RmsNorm without learnable weight (pure normalization) ───────────────

/// RmsNorm without learnable weight — creates weight = ones.
fn rms_norm_no_weight(dim: usize, eps: f64, dev: &Device, dtype: DType) -> Result<RmsNorm> {
    let weight = Tensor::ones((dim,), dtype, dev)?;
    Ok(RmsNorm::from_weight(weight, eps))
}

// ── MLP ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Gemma4Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Gemma4Mlp {
    fn new(cfg: &Gemma4Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let intermediate = cfg.layer_intermediate_size(layer_idx);
        Ok(Self {
            gate_proj: Linear::load(vb.pp("gate_proj"), cfg.hidden_size, intermediate, false)?,
            up_proj: Linear::load(vb.pp("up_proj"), cfg.hidden_size, intermediate, false)?,
            down_proj: Linear::load(vb.pp("down_proj"), intermediate, cfg.hidden_size, false)?,
        })
    }

    fn forward(&self, ops: &dyn crate::ops::Ops, x: &Tensor) -> Result<Tensor> {
        let bs = BatchState::no_lora();
        let gate = self.gate_proj.forward(x, &bs, ops)?;
        let up = self.up_proj.forward(x, &bs, ops)?;
        // GeluAndMul: GELU(gate) * up — uses fused kernel when available
        let activated = ops.gelu_mul(&gate, &up)?;
        self.down_proj.forward(&activated, &bs, ops)
    }
}

// ── MoE ──────────────────────────────────────────────────────────────────

/// A single MoE expert (same structure as Gemma4Mlp).
#[derive(Debug, Clone)]
struct Gemma4Expert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Gemma4Expert {
    fn new(cfg: &Gemma4Config, vb: VarBuilder) -> Result<Self> {
        let expert_intermediate = cfg.moe_intermediate_size();
        Ok(Self {
            gate_proj: Linear::load(
                vb.pp("gate_proj"),
                cfg.hidden_size,
                expert_intermediate,
                false,
            )?,
            up_proj: Linear::load(
                vb.pp("up_proj"),
                cfg.hidden_size,
                expert_intermediate,
                false,
            )?,
            down_proj: Linear::load(
                vb.pp("down_proj"),
                expert_intermediate,
                cfg.hidden_size,
                false,
            )?,
        })
    }

    fn forward(&self, ops: &dyn crate::ops::Ops, x: &Tensor) -> Result<Tensor> {
        let bs = BatchState::no_lora();
        let gate = self.gate_proj.forward(x, &bs, ops)?;
        let up = self.up_proj.forward(x, &bs, ops)?;
        let activated = ops.gelu_mul(&gate, &up)?;
        self.down_proj.forward(&activated, &bs, ops)
    }
}

/// Router for Gemma4 MoE: RMSNorm(no weight) → scale by hidden_size^{-0.5} → learned scale → proj.
#[derive(Debug, Clone)]
struct Gemma4Router {
    norm: RmsNorm,
    scale: Tensor,
    root_size: f64,
    proj: Linear,
}

impl Gemma4Router {
    fn new(cfg: &Gemma4Config, vb: VarBuilder) -> Result<Self> {
        let num_experts = cfg.num_experts.unwrap_or(0);
        let norm = rms_norm_no_weight(cfg.hidden_size, cfg.rms_norm_eps, vb.device(), vb.dtype())?;
        let scale = vb.get(cfg.hidden_size, "scale")?;
        let proj = Linear::load(vb.pp("proj"), cfg.hidden_size, num_experts, false)?;
        Ok(Self {
            norm,
            scale,
            root_size: (cfg.hidden_size as f64).powf(-0.5),
            proj,
        })
    }

    /// Returns router logits [total_tokens, num_experts] in F32.
    fn forward(&self, ops: &dyn crate::ops::Ops, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = x.affine(self.root_size, 0.0)?;
        let x = (&x * &self.scale)?;
        let logits = self.proj.forward(&x, &BatchState::no_lora(), ops)?;
        logits.to_dtype(DType::F32)
    }
}

/// Gemma4 MoE block: sparse expert dispatch.
#[derive(Debug, Clone)]
struct Gemma4Moe {
    experts: Vec<Gemma4Expert>,
    per_expert_scale: Tensor,
    top_k: usize,
}

impl Gemma4Moe {
    fn new(cfg: &Gemma4Config, vb: VarBuilder) -> Result<Self> {
        let num_experts = cfg.num_experts.unwrap_or(0);
        let top_k = cfg.top_k_experts.unwrap_or(2);
        let per_expert_scale = vb.get(num_experts, "per_expert_scale")?;
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(Gemma4Expert::new(cfg, vb.pp(&format!("experts.{i}")))?);
        }
        Ok(Self {
            experts,
            per_expert_scale,
            top_k,
        })
    }

    /// Forward: given router_logits [T, E], dispatch to top-k experts.
    fn forward(
        &self,
        ops: &dyn crate::ops::Ops,
        x: &Tensor,
        router_logits: &Tensor,
    ) -> Result<Tensor> {
        let (total_tokens, _num_experts) = router_logits.dims2()?;

        // Softmax over all experts
        let router_probs = candle_nn::ops::softmax_last_dim(router_logits)?;

        // Top-k selection: get indices and weights
        let router_logits_vec: Vec<Vec<f32>> = router_logits.to_dtype(DType::F32)?.to_vec2()?;
        let router_probs_vec: Vec<Vec<f32>> = router_probs.to_vec2()?;
        let per_expert_scale: Vec<f32> = self.per_expert_scale.to_dtype(DType::F32)?.to_vec1()?;

        let mut output = Tensor::zeros(x.shape(), x.dtype(), x.device())?;

        for t in 0..total_tokens {
            // Find top-k experts for this token
            let mut expert_scores: Vec<(usize, f32)> = router_logits_vec[t]
                .iter()
                .enumerate()
                .map(|(i, &s)| (i, s))
                .collect();
            expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            expert_scores.truncate(self.top_k);

            // Compute dispatch weights: softmax prob of selected, renormalized
            let selected_probs: Vec<(usize, f32)> = expert_scores
                .iter()
                .map(|&(idx, _)| (idx, router_probs_vec[t][idx]))
                .collect();
            let prob_sum: f32 = selected_probs.iter().map(|(_, p)| p).sum();
            let renorm = if prob_sum > 0.0 { prob_sum } else { 1.0 };

            let token_input = x.narrow(0, t, 1)?;
            let mut token_output = Tensor::zeros(token_input.shape(), x.dtype(), x.device())?;

            for &(expert_idx, prob) in &selected_probs {
                let weight = (prob / renorm) * per_expert_scale[expert_idx];
                let expert_out = self.experts[expert_idx].forward(ops, &token_input)?;
                token_output = (token_output + expert_out.affine(weight as f64, 0.0)?)?;
            }

            // Scatter token output into result
            let indices = Tensor::from_vec(vec![t as u32], (1,), x.device())?;
            output = output.index_add(&indices, &token_output, 0)?;
        }

        Ok(output)
    }
}

// ── Attention ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Gemma4Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm, // No learnable weight
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Gemma4RotaryEmbedding>,
    is_sliding: bool,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
}

impl Gemma4Attention {
    fn new(
        cfg: &Gemma4Config,
        layer_idx: usize,
        rotary_emb: Arc<Gemma4RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim_for_layer(layer_idx);
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_kv_heads_for_layer(layer_idx);
        let is_sliding = cfg.is_sliding_layer(layer_idx);
        let is_full = !is_sliding;
        let use_k_eq_v = is_full && cfg.attention_k_eq_v;

        let q_proj = Linear::load(
            vb.pp("q_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
        )?;
        let k_proj = Linear::load(
            vb.pp("k_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
        )?;

        // For k_eq_v layers, V reuses K weights (loaded at weight-loading time by duplicating k_proj → v_proj).
        // In our case we just load v_proj normally — weight loading will have placed the right weights there.
        let v_proj = if use_k_eq_v {
            // For k_eq_v, v_proj will have same weights as k_proj (handled in weight loading).
            // But we still need to load it. If the checkpoint doesn't have v_proj, we clone k_proj.
            match Linear::load(
                vb.pp("v_proj"),
                cfg.hidden_size,
                num_kv_heads * head_dim,
                cfg.attention_bias,
            ) {
                Ok(v) => v,
                Err(_) => {
                    // Clone k_proj weights as v_proj
                    Linear::from_weight(
                        vb.pp("k_proj")
                            .get((num_kv_heads * head_dim, cfg.hidden_size), "weight")?,
                        None,
                    )?
                }
            }
        } else {
            Linear::load(
                vb.pp("v_proj"),
                cfg.hidden_size,
                num_kv_heads * head_dim,
                cfg.attention_bias,
            )?
        };

        let o_proj = Linear::load(
            vb.pp("o_proj"),
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
        )?;

        // Q/K norms have learnable weights
        let q_norm = RmsNorm::load(vb.pp("q_norm"), head_dim, cfg.rms_norm_eps)?;
        let k_norm = RmsNorm::load(vb.pp("k_norm"), head_dim, cfg.rms_norm_eps)?;
        // V norm: no learnable weight (pure normalization)
        let v_norm = rms_norm_no_weight(head_dim, cfg.rms_norm_eps, vb.device(), vb.dtype())?;

        let sliding_window = if is_sliding { cfg.sliding_window } else { None };
        let softcap = cfg.attn_logit_softcapping.map(|c| c as f32);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            is_sliding,
            sliding_window,
            softcap,
        })
    }

    /// Forward pass. For KV-shared layers, `shared_kv` provides the source layer's
    /// post-norm/RoPE K and post-norm V. Returns (output, Option<(K, V)>) where
    /// the K/V are returned for source layers so they can be shared during prefill.
    fn forward(
        &mut self,
        ops: &dyn crate::ops::Ops,
        packed_input: &Tensor,
        cu_seqlens: &Tensor,
        max_seqlen: usize,
        position_ids: &Tensor,
        shared_kv: Option<(&Tensor, &Tensor)>,
        paged_kv: Option<&PagedKvContext<'_>>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (total_tokens, _) = packed_input.dims2()?;
        let bs = BatchState::no_lora();
        let is_shared = shared_kv.is_some();

        // Q: always computed (proj + norm + RoPE)
        let q = self.q_proj.forward(packed_input, &bs, ops)?;
        let q = q.reshape((total_tokens, self.num_heads, self.head_dim))?;
        let q = ops.rms_norm(&q, self.q_norm.weight(), self.q_norm.eps() as f32)?;

        let q_cos = self.rotary_emb.cos.index_select(position_ids, 0)?;
        let q_sin = self.rotary_emb.sin.index_select(position_ids, 0)?;
        let (total, hq, d2) = q.dims3()?;
        let q = crate::ops::rope_thd(&q.reshape((1, total, hq, d2))?, &q_cos, &q_sin)?
            .reshape((total, hq, d2))?;

        // K/V: shared layers reuse source layer's K/V, non-shared compute their own
        let (k, v, kv_to_share) = if let Some((sk, sv)) = shared_kv {
            (sk.clone(), sv.clone(), None)
        } else {
            let k = self.k_proj.forward(packed_input, &bs, ops)?;
            let v = self.v_proj.forward(packed_input, &bs, ops)?;
            let k = k.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;
            let k = ops.rms_norm(&k, self.k_norm.weight(), self.k_norm.eps() as f32)?;
            let hk = k.dim(1)?;
            let k = crate::ops::rope_thd(&k.reshape((1, total, hk, d2))?, &q_cos, &q_sin)?
                .reshape((total, hk, d2))?;
            let v = v.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;
            let v = self.v_norm.forward(&v)?;
            (k.clone(), v.clone(), Some((k, v)))
        };

        // Attention dispatch: paged KV cache (decode) or varlen (prefill)
        let mask = if self.is_sliding {
            let ws = self.sliding_window.unwrap_or(max_seqlen);
            MaskType::SlidingWindow {
                left: ws.saturating_sub(1),
                right: 0,
            }
        } else {
            MaskType::Causal
        };

        let attn_out = if let Some(kv) = paged_kv {
            // Paged decode: write K/V to cache (skip for shared layers — cache is aliased),
            // then read from cache for attention.
            if !is_shared {
                ops.reshape_and_cache(&k, &v, kv.key_cache, kv.value_cache, kv.slot_mapping)?;
            }
            ops.paged_attention(
                &q,
                kv.key_cache,
                kv.value_cache,
                &PagedParams {
                    block_tables: kv.block_tables,
                    cu_seqlens_q: cu_seqlens,
                    cu_seqlens_k: kv.cu_seqlens_k,
                    max_seqlen_q: max_seqlen,
                    max_seqlen_k: kv.max_seqlen_k,
                    scale: 1.0,
                    mask,
                    softcap: self.softcap,
                },
            )?
        } else {
            // Varlen prefill
            ops.varlen_attention(
                &q,
                &k,
                &v,
                &VarlenParams {
                    cu_seqlens_q: cu_seqlens,
                    cu_seqlens_k: cu_seqlens,
                    max_seqlen_q: max_seqlen,
                    max_seqlen_k: max_seqlen,
                    scale: 1.0,
                    mask,
                    softcap: self.softcap,
                },
            )?
        };

        let attn_dim = self.num_heads * self.head_dim;
        let output = self
            .o_proj
            .forward(&attn_out.reshape((total_tokens, attn_dim))?, &bs, ops)?;
        Ok((output, kv_to_share))
    }

    fn clear_kv_cache(&mut self) {}
}

// ── Decoder Layer ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Gemma4DecoderLayer {
    self_attn: Gemma4Attention,
    mlp: Gemma4Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    // MoE (optional)
    router: Option<Gemma4Router>,
    moe: Option<Gemma4Moe>,
    post_feedforward_layernorm_1: Option<RmsNorm>,
    pre_feedforward_layernorm_2: Option<RmsNorm>,
    post_feedforward_layernorm_2: Option<RmsNorm>,
    // PLE (optional)
    per_layer_input_gate: Option<Linear>,
    per_layer_projection: Option<Linear>,
    post_per_layer_input_norm: Option<RmsNorm>,
    // Layer scalar
    layer_scalar: Tensor,
}

impl Gemma4DecoderLayer {
    fn new(
        cfg: &Gemma4Config,
        layer_idx: usize,
        rotary_emb: Arc<Gemma4RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Gemma4Attention::new(cfg, layer_idx, rotary_emb, vb.pp("self_attn"))?;
        let mlp = Gemma4Mlp::new(cfg, layer_idx, vb.pp("mlp"))?;

        let input_layernorm =
            RmsNorm::load(vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let post_attention_layernorm = RmsNorm::load(
            vb.pp("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let pre_feedforward_layernorm = RmsNorm::load(
            vb.pp("pre_feedforward_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let post_feedforward_layernorm = RmsNorm::load(
            vb.pp("post_feedforward_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        // MoE
        let (router, moe, pf_ln1, pf_ln2_pre, pf_ln2) = if cfg.enable_moe_block {
            let r = Gemma4Router::new(cfg, vb.pp("router"))?;
            let m = Gemma4Moe::new(cfg, vb.pp("moe"))?;
            let ln1 = RmsNorm::load(
                vb.pp("post_feedforward_layernorm_1"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?;
            let ln2_pre = RmsNorm::load(
                vb.pp("pre_feedforward_layernorm_2"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?;
            let ln2 = RmsNorm::load(
                vb.pp("post_feedforward_layernorm_2"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?;
            (Some(r), Some(m), Some(ln1), Some(ln2_pre), Some(ln2))
        } else {
            (None, None, None, None, None)
        };

        // PLE
        let ple_dim = cfg.ple_dim();
        let (ple_gate, ple_proj, ple_norm) = if ple_dim > 0 {
            let gate = Linear::load(
                vb.pp("per_layer_input_gate"),
                cfg.hidden_size,
                ple_dim,
                false,
            )?;
            let proj = Linear::load(
                vb.pp("per_layer_projection"),
                ple_dim,
                cfg.hidden_size,
                false,
            )?;
            let norm = RmsNorm::load(
                vb.pp("post_per_layer_input_norm"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?;
            (Some(gate), Some(proj), Some(norm))
        } else {
            (None, None, None)
        };

        // Layer scalar (defaults to 1.0 if not in checkpoint)
        let layer_scalar = match vb.get(1, "layer_scalar") {
            Ok(s) => s,
            Err(_) => Tensor::ones((1,), vb.dtype(), vb.device())?,
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            router,
            moe,
            post_feedforward_layernorm_1: pf_ln1,
            pre_feedforward_layernorm_2: pf_ln2_pre,
            post_feedforward_layernorm_2: pf_ln2,
            per_layer_input_gate: ple_gate,
            per_layer_projection: ple_proj,
            post_per_layer_input_norm: ple_norm,
            layer_scalar,
        })
    }

    /// Forward pass. Returns (output, Option<(K, V)>) — K/V returned for source layers.
    fn forward(
        &mut self,
        ops: &dyn crate::ops::Ops,
        xs: &Tensor,
        cu_seqlens: &Tensor,
        max_seqlen: usize,
        position_ids: &Tensor,
        per_layer_input: Option<&Tensor>,
        shared_kv: Option<(&Tensor, &Tensor)>,
        paged_kv: Option<&PagedKvContext<'_>>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        // 1. Self-attention with residual
        let residual = xs;
        let hidden = self.input_layernorm.forward(residual)?;
        let (hidden, kv_to_share) = self.self_attn.forward(
            ops,
            &hidden,
            cu_seqlens,
            max_seqlen,
            position_ids,
            shared_kv,
            paged_kv,
        )?;
        let hidden = self.post_attention_layernorm.forward(&hidden)?;
        let xs = ops.add_or_fused(&hidden, residual)?;
        let residual = &xs;

        // 2. MLP
        let hidden = self.pre_feedforward_layernorm.forward(residual)?;
        let hidden = self.mlp.forward(ops, &hidden)?;

        // 3. MoE (optional, parallel to MLP)
        let hidden =
            if let (Some(router), Some(moe), Some(pf_ln1), Some(pf_ln2_pre), Some(pf_ln2)) = (
                &self.router,
                &self.moe,
                &self.post_feedforward_layernorm_1,
                &self.pre_feedforward_layernorm_2,
                &self.post_feedforward_layernorm_2,
            ) {
                let mlp_normed = pf_ln1.forward(&hidden)?;
                let router_logits = router.forward(ops, residual)?;
                let moe_input = pf_ln2_pre.forward(residual)?;
                let moe_out = moe.forward(ops, &moe_input, &router_logits)?;
                let moe_normed = pf_ln2.forward(&moe_out)?;
                ops.add_or_fused(&mlp_normed, &moe_normed)?
            } else {
                hidden
            };

        // 4. Post-FFN norm + residual
        let hidden = self.post_feedforward_layernorm.forward(&hidden)?;
        let mut xs = ops.add_or_fused(&hidden, residual)?;

        // 5. PLE (optional)
        if let (Some(gate_linear), Some(proj_linear), Some(norm)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
        ) {
            if let Some(ple_input) = per_layer_input {
                let gate = gate_linear.forward(&xs, &BatchState::no_lora(), ops)?;
                let gate = gate.gelu()?;
                let gated = (gate * ple_input)?;
                let contribution = proj_linear.forward(&gated, &BatchState::no_lora(), ops)?;
                let contribution = norm.forward(&contribution)?;
                xs = (xs + contribution)?;
            }
        }

        // 6. Layer scalar
        let xs = xs.broadcast_mul(&self.layer_scalar)?;
        Ok((xs, kv_to_share))
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Base Model ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Gemma4Model {
    embed_tokens: Embedding,
    layers: Vec<Gemma4DecoderLayer>,
    norm: RmsNorm,
    hidden_size: usize,
    // PLE infrastructure
    embed_tokens_per_layer: Option<Embedding>,
    per_layer_model_projection: Option<Linear>,
    per_layer_projection_norm: Option<RmsNorm>,
    ple_dim: usize,
    num_hidden_layers: usize,
    /// KV sharing: kv_sharing_map[layer_idx] = Some(source_layer_idx) for shared layers.
    kv_sharing_map: Vec<Option<usize>>,
}

impl Gemma4Model {
    fn new(cfg: &Gemma4Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = {
            let emb_vb = vb.pp("embed_tokens");
            let weight = emb_vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            Embedding::new(weight, cfg.hidden_size)
        };

        // PLE embeddings
        let ple_dim = cfg.ple_dim();
        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if ple_dim > 0 {
                let total_ple_dim = ple_dim * cfg.num_hidden_layers;
                let ple_vocab = cfg.ple_vocab_size();
                let ple_emb_weight = vb
                    .pp("embed_tokens_per_layer")
                    .get((ple_vocab, total_ple_dim), "weight")?;
                let ple_emb = Embedding::new(ple_emb_weight, total_ple_dim);
                let proj = Linear::load(
                    vb.pp("per_layer_model_projection"),
                    cfg.hidden_size,
                    total_ple_dim,
                    false,
                )?;
                let norm = RmsNorm::load(
                    vb.pp("per_layer_projection_norm"),
                    ple_dim,
                    cfg.rms_norm_eps,
                )?;
                (Some(ple_emb), Some(proj), Some(norm))
            } else {
                (None, None, None)
            };

        // Build per-layer rotary embeddings (each layer type may have different params)
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        // Cache rotary embeddings by (rope_theta, partial_rotary_factor, head_dim) to share
        let mut rotary_cache: std::collections::HashMap<
            (u64, u64, usize),
            Arc<Gemma4RotaryEmbedding>,
        > = std::collections::HashMap::new();

        for i in 0..cfg.num_hidden_layers {
            let rope_theta = cfg.rope_theta_for_layer(i);
            let partial_factor = cfg.partial_rotary_factor_for_layer(i);
            let head_dim = cfg.head_dim_for_layer(i);
            let key = (rope_theta.to_bits(), partial_factor.to_bits(), head_dim);

            let rotary = rotary_cache
                .entry(key)
                .or_insert_with(|| {
                    Arc::new(
                        Gemma4RotaryEmbedding::new(
                            vb.dtype(),
                            head_dim,
                            cfg.max_position_embeddings,
                            rope_theta,
                            partial_factor,
                            vb.device(),
                        )
                        .unwrap(),
                    )
                })
                .clone();

            layers.push(Gemma4DecoderLayer::new(
                cfg,
                i,
                rotary,
                vb.pp(&format!("layers.{}", i)),
            )?);
        }

        let norm = RmsNorm::load(vb.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            hidden_size: cfg.hidden_size,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            ple_dim,
            num_hidden_layers: cfg.num_hidden_layers,
            kv_sharing_map: (0..cfg.num_hidden_layers)
                .map(|i| cfg.kv_sharing_source(i))
                .collect(),
        })
    }

    /// Compute per-layer inputs from PLE embeddings + projection.
    fn compute_per_layer_inputs(
        &self,
        ops: &dyn crate::ops::Ops,
        hidden_states: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Option<Tensor>> {
        if self.ple_dim == 0 {
            return Ok(None);
        }

        let (embed_ple, proj, norm) = match (
            &self.embed_tokens_per_layer,
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) {
            (Some(e), Some(p), Some(n)) => (e, p, n),
            _ => return Ok(None),
        };

        let total_tokens = input_ids.dim(0)?;

        // PLE embeddings: [total_tokens, total_ple_dim]
        let ple_embeds = embed_ple.forward(input_ids)?;
        let embed_scale = (self.ple_dim as f64).sqrt();
        let ple_embeds = (ple_embeds * embed_scale)?;
        // Reshape: [total_tokens, num_layers, ple_dim]
        let ple_embeds =
            ple_embeds.reshape((total_tokens, self.num_hidden_layers, self.ple_dim))?;

        // Projection from hidden_states: [total_tokens, total_ple_dim]
        let projection = proj.forward(hidden_states, &BatchState::no_lora(), ops)?;
        let proj_scale = (self.hidden_size as f64).powf(-0.5);
        let projection = (projection * proj_scale)?;
        // Reshape: [total_tokens, num_layers, ple_dim]
        let projection =
            projection.reshape((total_tokens, self.num_hidden_layers, self.ple_dim))?;
        // Normalize each per-layer slice
        let projection = norm.forward(&projection)?;

        // Combine: (projection + ple_embeds) * 1/sqrt(2)
        let input_scale = (2.0f64).powf(-0.5);
        let combined = ((projection + ple_embeds)? * input_scale)?;
        Ok(Some(combined))
    }

    fn forward(
        &mut self,
        ops: &dyn crate::ops::Ops,
        packed_input: &Tensor,
        cu_seqlens: &Tensor,
        max_seqlen: usize,
        position_ids: &Tensor,
        paged_kv: Option<&PagedKvBatchContext<'_>>,
    ) -> Result<Tensor> {
        let embed_scale = (self.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(packed_input)? * embed_scale)?;

        // Compute PLE inputs
        let per_layer_inputs = self.compute_per_layer_inputs(ops, &xs, packed_input)?;

        // KV sharing: source layers store their K/V for shared layers to reuse (prefill path).
        // In paged path, sharing is handled by cache aliasing — shared layers read from
        // the source layer's cache automatically.
        let mut shared_kv_store: std::collections::HashMap<usize, (Tensor, Tensor)> =
            std::collections::HashMap::new();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let ple_input = per_layer_inputs
                .as_ref()
                .map(|pli| pli.narrow(1, i, 1).unwrap().squeeze(1).unwrap());

            // For KV-shared layers in prefill: get source layer's K/V
            let shared_kv = if paged_kv.is_none() {
                self.kv_sharing_map[i]
                    .and_then(|src| shared_kv_store.get(&src).map(|(k, v)| (k, v)))
            } else {
                None // Paged path: sharing handled by cache aliasing
            };

            // Per-layer paged KV context
            let layer_kv = paged_kv.map(|p| p.layer(i));

            let (out, kv_to_share) = layer.forward(
                ops,
                &xs,
                cu_seqlens,
                max_seqlen,
                position_ids,
                ple_input.as_ref(),
                shared_kv,
                layer_kv.as_ref(),
            )?;
            xs = out;

            // Source layers: store K/V for future shared layers (prefill only)
            if let Some(kv) = kv_to_share {
                shared_kv_store.insert(i, kv);
            }
        }

        self.norm.forward(&xs)
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

// ── Causal LM Model ──────────────────────────────────────────────────────

/// Gemma4 model for causal language modeling
#[derive(Debug, Clone)]
pub struct Gemma4ForCausalLM {
    base: Gemma4Model,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
}

impl Gemma4ForCausalLM {
    pub fn new(cfg: &Gemma4Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with_parts(cfg, vb.pp("model"), vb)
    }

    pub fn new_with_parts(
        cfg: &Gemma4Config,
        model_vb: VarBuilder,
        head_vb: VarBuilder,
    ) -> Result<Self> {
        let base = Gemma4Model::new(cfg, model_vb.clone())?;

        let lm_head = if cfg.tie_word_embeddings {
            let embed_weight = model_vb
                .pp("embed_tokens")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            Linear::from_weight(embed_weight, None)?
        } else {
            Linear::load(
                head_vb.pp("lm_head"),
                cfg.hidden_size,
                cfg.vocab_size,
                false,
            )?
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
            ctx.paged_kv,
        )?;
        let last_hidden =
            crate::models::commons::last_token_select(&hidden, ctx.seq_lens)?.contiguous()?;

        let logits =
            self.lm_head
                .forward(&last_hidden.unsqueeze(1)?, &BatchState::no_lora(), ctx.ops)?;

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

impl crate::models::LogitsSplitModel for Gemma4ForCausalLM {
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
            ctx.paged_kv,
        )
    }

    fn compute_logits(&self, hidden: &Tensor) -> crate::tensor::Result<Tensor> {
        let logits = self.lm_head.forward(
            hidden,
            &BatchState::no_lora(),
            crate::ops::select_ops(hidden.device()),
        )?;
        if let Some(cap) = self.final_logit_softcapping {
            let scaled = (&logits / cap)?;
            let tanh = scaled.tanh()?;
            tanh * cap
        } else {
            Ok(logits)
        }
    }
}

impl crate::models::ModelForward for Gemma4ForCausalLM {
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

    fn kv_cache_sharing(&self) -> Vec<Option<usize>> {
        self.base.kv_sharing_map.clone()
    }

    fn as_logits_model(&self) -> Option<&dyn crate::models::LogitsSplitModel> {
        Some(self)
    }

    fn as_logits_model_mut(&mut self) -> Option<&mut dyn crate::models::LogitsSplitModel> {
        Some(self)
    }
}

// ── Registry / meta ─────────────────────────────────────────────────────

pub(crate) mod meta {
    use super::{Gemma4Config, Gemma4ForCausalLM};
    use crate::engine::EngineError;
    use crate::engine::{CommonModelConfig, RuntimeCaps, TaskKind, WeightsBackend};
    use crate::loading::var_builder::VarBuilder;
    use crate::models::registry::{ArchSpec, ParsedModelConfig, candle_model_err, parse_value};

    const ARCHITECTURE_ALIASES: &[&str] = &["Gemma4", "Gemma4Text"];
    const MODEL_TYPE_ALIASES: &[&str] = &["gemma4", "gemma4_text", "gemma3n"];
    const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate];

    #[derive(Debug, Clone, Copy)]
    enum Gemma4WeightLayout {
        FlatText,
        NestedLanguageModel,
    }

    enum Gemma4ArchConfig {
        Dense {
            cfg: Gemma4Config,
            layout: Gemma4WeightLayout,
        },
    }

    fn common_from_gemma4(cfg: &Gemma4Config) -> CommonModelConfig {
        CommonModelConfig {
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            max_position_embeddings: cfg.max_position_embeddings,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        }
    }

    fn infer_weight_layout(raw: &serde_json::Value) -> Gemma4WeightLayout {
        if raw.get("text_config").is_some() {
            Gemma4WeightLayout::NestedLanguageModel
        } else {
            Gemma4WeightLayout::FlatText
        }
    }

    fn parse_gemma4_text_config(
        raw: &serde_json::Value,
        description: &str,
    ) -> Result<Gemma4Config, EngineError> {
        if let Some(text_config) = raw.get("text_config") {
            parse_value(text_config.clone(), description)
        } else {
            parse_value(raw.clone(), description)
        }
    }

    pub(crate) struct Gemma4ArchSpec;

    pub(crate) static GEMMA4_ARCH_SPEC: Gemma4ArchSpec = Gemma4ArchSpec;
    inventory::submit!(crate::models::registry::ArchSpecEntry::new(
        &GEMMA4_ARCH_SPEC
    ));

    impl ArchSpec for Gemma4ArchSpec {
        fn name(&self) -> &'static str {
            "gemma4"
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
                    let cfg = parse_gemma4_text_config(raw, "Gemma4 config")?;
                    let common = common_from_gemma4(&cfg);
                    Ok(ParsedModelConfig {
                        common,
                        deltanet: None,
                        arch_config: Box::new(Gemma4ArchConfig::Dense { cfg, layout }),
                    })
                }
                _ => Err(EngineError::InvalidRequest(format!(
                    "Gemma4 does not support task {:?}",
                    task
                ))),
            }
        }

        fn build_model(
            &self,
            arch_config: &dyn std::any::Any,
            vb: VarBuilder<'_>,
        ) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
            let cfg = arch_config
                .downcast_ref::<Gemma4ArchConfig>()
                .ok_or_else(|| {
                    EngineError::Internal("unexpected arch config type for Gemma4".into())
                })?;

            match cfg {
                Gemma4ArchConfig::Dense { cfg, layout } => {
                    // Gemma4 weight layout:
                    //   FlatText: model.layers.0... (text-only checkpoint)
                    //   Nested:   model.language_model.layers.0... (multimodal checkpoint)
                    // Note: unlike Gemma3, Gemma4 does NOT have an extra ".model" level.
                    let model_vb = match layout {
                        Gemma4WeightLayout::FlatText => vb.clone().pp("model"),
                        Gemma4WeightLayout::NestedLanguageModel => {
                            vb.clone().pp("model").pp("language_model")
                        }
                    };
                    Ok(Box::new(
                        Gemma4ForCausalLM::new_with_parts(cfg, model_vb, vb)
                            .map_err(candle_model_err)?,
                    ))
                }
            }
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
                supports_paged_attn: device.is_cuda() && is_safetensors,
                supports_varlen: device.is_cuda() && is_safetensors,
                supports_deltanet: false,
                supports_cuda_graph: false,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::engine::TaskKind;
        use crate::models::registry::find_arch_spec_by_architecture_prefix;

        #[test]
        fn gemma4_registry_lookup() {
            let spec = find_arch_spec_by_architecture_prefix("Gemma4");
            assert!(spec.is_some(), "Gemma4 should be registered");
            assert_eq!(spec.unwrap().name(), "gemma4");
        }

        #[test]
        fn gemma4_registry_lookup_text() {
            let spec = find_arch_spec_by_architecture_prefix("Gemma4Text");
            assert!(spec.is_some(), "Gemma4Text should be registered");
        }

        #[test]
        fn gemma4_parse_config_generate() {
            let spec = find_arch_spec_by_architecture_prefix("Gemma4").unwrap();
            let json = serde_json::json!({
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
            });
            let result = spec.parse_config(TaskKind::Generate, &json, "");
            assert!(result.is_ok(), "should parse: {:?}", result.err());
        }

        #[test]
        fn gemma4_parse_nested_text_config() {
            let spec = find_arch_spec_by_architecture_prefix("Gemma4").unwrap();
            let json = serde_json::json!({
                "text_config": {
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 32,
                }
            });
            let result = spec.parse_config(TaskKind::Generate, &json, "");
            assert!(result.is_ok(), "should parse nested: {:?}", result.err());
        }
    }
}
