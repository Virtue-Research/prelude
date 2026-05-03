use std::sync::{Arc, OnceLock};

use crate::config::{MoeBackendPolicy, global_runtime};
use crate::loading::var_builder::VarBuilder;
use crate::models::commons::activation::Activation;
use crate::models::commons::embedding::Embedding;
use crate::models::commons::linear::DenseLinear;
use crate::models::config::Qwen3Config;
use crate::profiling::{nvtx_pop, nvtx_push};
use crate::tensor::{D, DType, Module, Result, Tensor};

// Shared layer primitives (extracted from qwen3 into reusable modules)
use crate::models::commons::{
    BatchAttnContext, BatchState, LayerAttnContext, Linear, RmsNorm, RotaryEmbedding,
    last_token_select,
};

// Model-specific attention (still lives in qwen3)
use super::qwen3::Qwen3Attention;

// ── Config ──────────────────────────────────────────────────────────────

fn moe_backend_policy() -> MoeBackendPolicy {
    global_runtime()
        .map(|runtime| runtime.moe_backend)
        .unwrap_or_default()
}

fn log_moe_backend_once(requested: MoeBackendPolicy, actual: &'static str) {
    static CUTLASS: OnceLock<()> = OnceLock::new();
    static SEQUENTIAL: OnceLock<()> = OnceLock::new();

    let logged = match actual {
        "flashinfer_cutlass" => &CUTLASS,
        "sequential" => &SEQUENTIAL,
        _ => &SEQUENTIAL,
    };
    logged.get_or_init(|| {
        tracing::info!(
            requested = requested.as_str(),
            actual,
            "Qwen3 MoE backend selected"
        );
    });
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub moe_intermediate_size: usize,
    pub num_experts_per_tok: usize,
    pub num_experts: usize,
    pub attention_bias: bool,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub use_sliding_window: bool,
    pub norm_topk_prob: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    pub decoder_sparse_step: usize,
}

impl<'de> serde::Deserialize<'de> for Qwen3MoeConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RopeParameters {
            #[serde(default)]
            rope_theta: Option<f64>,
        }

        #[derive(serde::Deserialize)]
        struct Raw {
            vocab_size: usize,
            hidden_size: usize,
            intermediate_size: usize,
            num_hidden_layers: usize,
            num_attention_heads: usize,
            head_dim: usize,
            num_key_value_heads: usize,
            max_position_embeddings: usize,
            moe_intermediate_size: usize,
            num_experts_per_tok: usize,
            #[serde(alias = "num_local_experts")]
            num_experts: usize,
            #[serde(default)]
            attention_bias: bool,
            #[serde(default)]
            sliding_window: Option<usize>,
            #[serde(default)]
            max_window_layers: usize,
            #[serde(default)]
            tie_word_embeddings: bool,
            #[serde(default)]
            use_sliding_window: bool,
            #[serde(default)]
            norm_topk_prob: bool,
            #[serde(default)]
            rope_theta: Option<f64>,
            #[serde(default)]
            rope_parameters: Option<RopeParameters>,
            #[serde(default)]
            rms_norm_eps: Option<f64>,
            #[serde(default)]
            hidden_act: Option<Activation>,
            #[serde(default)]
            decoder_sparse_step: Option<usize>,
        }

        let r = Raw::deserialize(deserializer)?;
        let rope_theta = r
            .rope_theta
            .or_else(|| r.rope_parameters.and_then(|p| p.rope_theta));
        const MODEL: &str = "Qwen3MoE";
        Ok(Self {
            vocab_size: r.vocab_size,
            hidden_size: r.hidden_size,
            intermediate_size: r.intermediate_size,
            num_hidden_layers: r.num_hidden_layers,
            num_attention_heads: r.num_attention_heads,
            head_dim: r.head_dim,
            num_key_value_heads: r.num_key_value_heads,
            max_position_embeddings: r.max_position_embeddings,
            moe_intermediate_size: r.moe_intermediate_size,
            num_experts_per_tok: r.num_experts_per_tok,
            num_experts: r.num_experts,
            attention_bias: r.attention_bias,
            sliding_window: r.sliding_window,
            max_window_layers: r.max_window_layers,
            tie_word_embeddings: r.tie_word_embeddings,
            use_sliding_window: r.use_sliding_window,
            norm_topk_prob: r.norm_topk_prob,
            rope_theta: crate::models::resolve_or_warn!(
                rope_theta,
                1_000_000.0,
                "rope_theta",
                MODEL
            ),
            rms_norm_eps: crate::models::resolve_or_warn!(
                r.rms_norm_eps,
                1e-6,
                "rms_norm_eps",
                MODEL
            ),
            hidden_act: crate::models::resolve_or_warn!(
                r.hidden_act,
                Activation::Silu,
                "hidden_act",
                MODEL
            ),
            decoder_sparse_step: crate::models::resolve_or_warn!(
                r.decoder_sparse_step,
                1,
                "decoder_sparse_step",
                MODEL
            ),
        })
    }
}

impl Qwen3MoeConfig {
    /// Convert to dense Qwen3Config for reusing attention code.
    pub fn to_dense_config(&self) -> Qwen3Config {
        Qwen3Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            head_dim: self.head_dim,
            attention_bias: self.attention_bias,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            sliding_window: self.sliding_window,
            max_window_layers: self.max_window_layers,
            tie_word_embeddings: self.tie_word_embeddings,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            use_sliding_window: self.use_sliding_window,
            hidden_act: self.hidden_act,
        }
    }
}

// ── Expert MLP ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Qwen3MoeExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3MoeExpert {
    fn new(cfg: &Qwen3MoeConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::load(
                vb.pp("gate_proj"),
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                false,
            )?,
            up_proj: Linear::load(
                vb.pp("up_proj"),
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                false,
            )?,
            down_proj: Linear::load(
                vb.pp("down_proj"),
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                false,
            )?,
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

#[derive(Debug, Clone)]
struct Qwen3SparseMoeBlock {
    gate: DenseLinear,
    experts: Vec<Qwen3MoeExpert>,
    // Pre-fused CUTLASS layout: [num_experts, 2*inter, hidden] ([up;gate])
    // and [num_experts, hidden, inter]. Enables FlashInfer's fused MoE
    // on supported CUDA architectures.
    experts_gate_up: Option<Tensor>,
    experts_down: Option<Tensor>,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl Qwen3SparseMoeBlock {
    fn new(cfg: &Qwen3MoeConfig, vb: VarBuilder) -> Result<Self> {
        let gate = {
            let gvb = vb.pp("gate");
            let w = gvb.get((cfg.num_experts, cfg.hidden_size), "weight")?;
            DenseLinear::new(w, None)
        };
        let mut experts = Vec::with_capacity(cfg.num_experts);
        let vb_e = vb.pp("experts");
        for idx in 0..cfg.num_experts {
            experts.push(Qwen3MoeExpert::new(cfg, vb_e.pp(idx))?);
        }

        // Stack expert weights once in FlashInfer CUTLASS layout (GPU only).
        // `experts_gate_up` is [num_experts, 2*inter, hidden] with [up; gate]
        // concatenation, matching CUTLASS's SwiGLU convention.
        let (experts_gate_up, experts_down) = if experts
            .first()
            .map_or(false, |e| e.gate_proj.weight().device().is_cuda())
        {
            let gate_ws: Vec<Tensor> = experts
                .iter()
                .map(|e| e.gate_proj.weight().clone())
                .collect();
            let up_ws: Vec<Tensor> = experts.iter().map(|e| e.up_proj.weight().clone()).collect();
            let down_ws: Vec<Tensor> = experts
                .iter()
                .map(|e| e.down_proj.weight().clone())
                .collect();

            let gate_w = Tensor::stack(&gate_ws, 0)?.contiguous()?;
            let up_w = Tensor::stack(&up_ws, 0)?.contiguous()?;
            let down_w = Tensor::stack(&down_ws, 0)?.contiguous()?;

            let experts_gate_up = Some(Tensor::cat(&[&up_w, &gate_w], 1)?.contiguous()?);
            let experts_down = Some(down_w);

            (experts_gate_up, experts_down)
        } else {
            (None, None)
        };

        Ok(Self {
            gate,
            experts,
            experts_gate_up,
            experts_down,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }

    /// Compute routing for 2D (varlen) input.
    ///
    /// Returns (topk_weights, experts_per_tok, hidden_dim).
    fn compute_routing_2d(
        &self,
        ops: &dyn crate::ops::Ops,
        xs: &Tensor,
    ) -> Result<(Tensor, Tensor, usize)> {
        let (n_tokens, hidden_dim) = xs.dims2()?;
        let router_logits = xs.apply(&self.gate)?;

        // Fast path: single fused CUDA kernel (softmax + top-k + optional
        // renorm, FP32 softmax internally). FlashInfer CUTLASS fused MoE
        // only needs weights and top-k ids, so the sort outputs are ignored.
        if let Some(result) = ops.fused_moe_routing(
            &router_logits,
            self.num_experts_per_tok,
            self.norm_topk_prob,
        ) {
            let (tw, topk_ids, _sorted_exp, _sorted_tok) = result?;
            return Ok((
                tw,
                topk_ids.reshape((n_tokens, self.num_experts_per_tok))?,
                hidden_dim,
            ));
        }

        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

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

        Ok((topk_weights, experts_per_tok, hidden_dim))
    }

    /// Sequential per-expert dispatch (CPU fallback).
    fn forward_sequential(
        &self,
        ops: &dyn crate::ops::Ops,
        xs: &Tensor,
        topk_weights: &Tensor,
        experts_per_tok: &Tensor,
        hidden_dim: usize,
    ) -> Result<Tensor> {
        let routing_weights = topk_weights.to_vec2::<f32>()?;
        let experts_per_tok_cpu = experts_per_tok.to_vec2::<u32>()?;

        let mut top_x: Vec<Vec<u32>> = vec![vec![]; self.experts.len()];
        let mut selected_weights: Vec<Vec<f32>> = vec![vec![]; self.experts.len()];
        for (row_idx, (rw, expert_idxs)) in routing_weights
            .iter()
            .zip(experts_per_tok_cpu.iter())
            .enumerate()
        {
            for (&rw, &expert_idx) in rw.iter().zip(expert_idxs.iter()) {
                top_x[expert_idx as usize].push(row_idx as u32);
                selected_weights[expert_idx as usize].push(rw);
            }
        }

        let mut ys = xs.zeros_like()?;
        for (expert_idx, expert_layer) in self.experts.iter().enumerate() {
            let top_x_expert = &top_x[expert_idx];
            if top_x_expert.is_empty() {
                continue;
            }
            let top_x_t = Tensor::new(top_x_expert.as_slice(), xs.device())?;
            let weights_t = Tensor::new(selected_weights[expert_idx].as_slice(), xs.device())?
                .reshape(((), 1))?
                .to_dtype(xs.dtype())?;

            let current_state = xs.index_select(&top_x_t, 0)?.reshape(((), hidden_dim))?;
            let current_hidden = expert_layer.forward(ops, &current_state)?;
            let current_hidden = current_hidden.broadcast_mul(&weights_t)?;
            ys = ys.index_add(&top_x_t, &current_hidden, 0)?;
        }

        Ok(ys)
    }

    /// Forward for varlen packed sequences: xs is (total_tokens, hidden_dim).
    #[allow(clippy::too_many_arguments)]
    fn forward_varlen(&self, ops: &dyn crate::ops::Ops, xs: &Tensor) -> Result<Tensor> {
        nvtx_push!("moe_routing");
        let (topk_weights, experts_per_tok, hidden_dim) = self.compute_routing_2d(ops, xs)?;
        nvtx_pop!();
        let backend_policy = moe_backend_policy();

        let input_is_cuda = xs.device().is_cuda();

        // Production path: FlashInfer's CUTLASS fused MoE handles
        // gate+up+silu+down+topk-weighted-sum in one kernel. On CUDA, `auto`
        // fails closed instead of silently measuring a slower fallback.
        // Forced `cutlass` also fails closed so benchmarks cannot
        // accidentally measure the reference path.
        if matches!(
            backend_policy,
            MoeBackendPolicy::Auto | MoeBackendPolicy::Cutlass
        ) {
            if !input_is_cuda && backend_policy == MoeBackendPolicy::Cutlass {
                return Err(candle_core::Error::Msg(
                    "MoE backend policy requires CUTLASS, but the input device is not CUDA".into(),
                ));
            }
            let require_cutlass = backend_policy == MoeBackendPolicy::Cutlass
                || (backend_policy == MoeBackendPolicy::Auto && input_is_cuda);
            if let (Some(egu), Some(ed)) =
                (self.experts_gate_up.as_ref(), self.experts_down.as_ref())
            {
                match ops.cutlass_fused_moe(xs, &experts_per_tok, &topk_weights, egu, ed) {
                    Some(Ok(out)) => {
                        log_moe_backend_once(backend_policy, "flashinfer_cutlass");
                        return Ok(out);
                    }
                    Some(Err(e)) if require_cutlass => {
                        return Err(candle_core::Error::Msg(format!(
                            "MoE backend policy '{}' requires FlashInfer CUTLASS, but fused MoE failed: {e}",
                            backend_policy.as_str(),
                        )));
                    }
                    Some(Err(e)) => {
                        tracing::warn!(error = %e, "CUTLASS fused MoE failed at runtime, falling back to reference MoE");
                    }
                    None if require_cutlass => {
                        return Err(candle_core::Error::Msg(format!(
                            "MoE backend policy '{}' requires FlashInfer CUTLASS, but the ops backend did not provide cutlass_fused_moe",
                            backend_policy.as_str(),
                        )));
                    }
                    None => {}
                }
            } else if backend_policy == MoeBackendPolicy::Cutlass
                || (backend_policy == MoeBackendPolicy::Auto && input_is_cuda)
            {
                return Err(candle_core::Error::Msg(format!(
                    "MoE backend policy '{}' requires FlashInfer CUTLASS, but CUTLASS expert weights are unavailable",
                    backend_policy.as_str(),
                )));
            }
        }

        // Reference path (CPU / explicit sequential / non-CUDA auto).
        log_moe_backend_once(backend_policy, "sequential");
        self.forward_sequential(ops, xs, &topk_weights, &experts_per_tok, hidden_dim)
    }
}

// ── Gated MLP (SiLU-gated FFN, for dense layers) ──────────────────────

#[derive(Debug, Clone)]
struct GatedMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    gate_up_proj: Option<Linear>,
}

impl GatedMlp {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = Linear::load(
            vb.pp("gate_proj"),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
        )?;
        let up_proj = Linear::load(
            vb.pp("up_proj"),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
        )?;
        let down_proj = Linear::load(
            vb.pp("down_proj"),
            cfg.intermediate_size,
            cfg.hidden_size,
            false,
        )?;

        let gate_up_proj = {
            let gw = gate_proj.weight();
            if gw.device().is_cpu() && gw.dtype() == DType::BF16 {
                let merged_w = Tensor::cat(&[gw, up_proj.weight()], 0)?;
                Linear::from_weight(merged_w, None).ok()
            } else {
                None
            }
        };

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_up_proj,
        })
    }

    fn forward(&self, x: &Tensor, ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        let bs = BatchState::no_lora();
        if let Some(ref gup) = self.gate_up_proj {
            let gate_up = gup.forward(x, &bs, ops)?;
            let dims = gate_up.dims();
            let dim = dims[dims.len() - 1] / 2;
            let gate = gate_up.narrow(dims.len() - 1, 0, dim)?;
            let up = gate_up.narrow(dims.len() - 1, dim, dim)?;
            return self.down_proj.forward(&ops.silu_mul(&gate, &up)?, &bs, ops);
        }
        let gate = self.gate_proj.forward(x, &bs, ops)?;
        let up = self.up_proj.forward(x, &bs, ops)?;
        self.down_proj.forward(&ops.silu_mul(&gate, &up)?, &bs, ops)
    }
}

// ── Feed-Forward dispatch ───────────────────────────────────────────────

#[derive(Debug, Clone)]
enum MoeFeedForward {
    Mlp(GatedMlp),
    SparseMoe(Qwen3SparseMoeBlock),
}

impl MoeFeedForward {
    fn forward_2d(&self, ops: &dyn crate::ops::Ops, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(x, ops),
            Self::SparseMoe(moe) => moe.forward_varlen(ops, x),
        }
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MoeDecoderLayer {
    self_attn: Qwen3Attention,
    feed_forward: MoeFeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MoeDecoderLayer {
    fn new(
        layer_idx: usize,
        cfg: &Qwen3MoeConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dense_cfg = cfg.to_dense_config();
        let self_attn = Qwen3Attention::new(&dense_cfg, rotary, vb.pp("self_attn"))?;

        let feed_forward = if cfg.num_experts > 0
            && cfg.decoder_sparse_step > 0
            && (layer_idx + 1) % cfg.decoder_sparse_step == 0
        {
            MoeFeedForward::SparseMoe(Qwen3SparseMoeBlock::new(cfg, vb.pp("mlp"))?)
        } else {
            MoeFeedForward::Mlp(GatedMlp::new(&dense_cfg, vb.pp("mlp"))?)
        };

        let input_layernorm =
            RmsNorm::load(vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let post_attention_layernorm = RmsNorm::load(
            vb.pp("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        Ok(Self {
            self_attn,
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        residual: Option<&Tensor>,
        ctx: &LayerAttnContext,
    ) -> Result<(Tensor, Tensor)> {
        let ops = ctx.ops;
        nvtx_push!("input_norm");
        let (residual, hidden) = self
            .input_layernorm
            .forward_residual(hidden, residual, ops)?;
        nvtx_pop!();
        nvtx_push!("attn");
        let hidden = self.self_attn.forward(&hidden, ctx)?;
        nvtx_pop!();
        nvtx_push!("post_attn_norm");
        let (residual, hidden) =
            self.post_attention_layernorm
                .forward_residual(&hidden, Some(&residual), ops)?;
        nvtx_pop!();
        nvtx_push!("ffn");
        let hidden = self.feed_forward.forward_2d(ops, &hidden)?;
        nvtx_pop!();
        Ok((hidden, residual))
    }
}

// ── Model ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MoeModel {
    embed_tokens: Embedding,
    layers: Vec<MoeDecoderLayer>,
    norm: RmsNorm,
}

impl MoeModel {
    fn new(cfg: &Qwen3MoeConfig, vb: VarBuilder) -> Result<Self> {
        let dense_cfg = cfg.to_dense_config();
        let embed_tokens = {
            let emb_vb = vb.pp("model.embed_tokens");
            let weight = emb_vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            Embedding::new(weight, cfg.hidden_size)
        };
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), &dense_cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MoeDecoderLayer::new(i, cfg, rotary.clone(), vb_l.pp(i))?);
        }

        let norm = RmsNorm::load(vb.pp("model.norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    fn clear_kv_cache(&mut self) {
        // No internal KV cache — paged KV is managed externally.
    }

    fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        nvtx_push!("embed");
        let mut hidden = self.embed_tokens.forward(packed_input)?;
        nvtx_pop!();
        let mut residual: Option<Tensor> = None;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            nvtx_push!("layer[{}]", i);
            let layer_kv = ctx.paged_kv.map(|kv| kv.layer(i));
            let layer_ctx = LayerAttnContext {
                ops: ctx.ops,
                cu_seqlens_q: ctx.cu_seqlens_q,
                max_seqlen_q: ctx.max_seqlen_q,
                position_ids: ctx.position_ids,
                paged_kv: layer_kv.as_ref(),
            };
            let (h, r) = layer.forward(&hidden, residual.as_ref(), &layer_ctx)?;
            hidden = h;
            residual = Some(r);
            nvtx_pop!();
        }
        nvtx_push!("final_norm");
        let (_, normed) = self
            .norm
            .forward_residual(&hidden, residual.as_ref(), ctx.ops)?;
        nvtx_pop!();
        Ok(normed)
    }
}

// ── ModelForCausalLM ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Qwen3MoeModelForCausalLM {
    base: MoeModel,
    lm_head: Linear,
}

impl Qwen3MoeModelForCausalLM {
    pub fn new(cfg: &Qwen3MoeConfig, vb: VarBuilder) -> Result<Self> {
        // GEMM dispatch (CUTLASS/DeepGEMM) is registered at startup, not per-model.

        let base = MoeModel::new(cfg, vb.clone())?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weight(base.embed_tokens.embeddings().clone(), None)?
        } else {
            Linear::load(vb.pp("lm_head"), cfg.hidden_size, cfg.vocab_size, false)?
        };
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let hidden = self.base.forward(packed_input, ctx)?;
        self.lm_head.forward(
            &last_token_select(&hidden, ctx.seq_lens)?.unsqueeze(1)?,
            &BatchState::no_lora(),
            ctx.ops,
        )
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}

impl crate::models::LogitsSplitModel for Qwen3MoeModelForCausalLM {
    fn forward_hidden_states(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> crate::tensor::Result<Tensor> {
        self.base.forward(packed_input, ctx)
    }

    fn compute_logits(&self, hidden: &Tensor) -> crate::tensor::Result<Tensor> {
        self.lm_head.forward(
            hidden,
            &BatchState::no_lora(),
            crate::ops::select_ops(hidden.device()),
        )
    }
}

impl crate::models::ModelForward for Qwen3MoeModelForCausalLM {
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
}

// ── Architecture registration ──────────────────────────────────────────

mod meta {
    use crate::loading::var_builder::VarBuilder;

    use super::{Qwen3MoeConfig, Qwen3MoeModelForCausalLM};
    use crate::engine::EngineError;
    use crate::engine::{CommonModelConfig, RuntimeCaps, TaskKind, WeightsBackend};
    use crate::models::registry::{ArchSpec, ParsedModelConfig, candle_model_err, parse_json};

    const ARCHITECTURE_ALIASES: &[&str] = &["Qwen3Moe", "Qwen3MoeModel"];
    const MODEL_TYPE_ALIASES: &[&str] = &["qwen3_moe"];
    const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate];

    pub(crate) struct Qwen3MoeArchSpec;

    pub(crate) static QWEN3_MOE_ARCH_SPEC: Qwen3MoeArchSpec = Qwen3MoeArchSpec;
    inventory::submit!(crate::models::registry::ArchSpecEntry::new(
        &QWEN3_MOE_ARCH_SPEC
    ));

    impl ArchSpec for Qwen3MoeArchSpec {
        fn name(&self) -> &'static str {
            "qwen3_moe"
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
            let cfg = parse_json::<Qwen3MoeConfig>(content, "Qwen3 MoE config")?;
            let common = CommonModelConfig {
                vocab_size: cfg.vocab_size,
                num_hidden_layers: cfg.num_hidden_layers,
                max_position_embeddings: cfg.max_position_embeddings,
                num_attention_heads: cfg.num_attention_heads,
                num_key_value_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
            };
            Ok(ParsedModelConfig {
                common,
                deltanet: None,
                arch_config: Box::new(cfg),
            })
        }

        fn build_model(
            &self,
            arch_config: &dyn std::any::Any,
            vb: VarBuilder<'_>,
        ) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
            let cfg = arch_config
                .downcast_ref::<Qwen3MoeConfig>()
                .ok_or_else(|| {
                    EngineError::Internal("unexpected arch config type for Qwen3Moe".into())
                })?;
            Ok(Box::new(
                Qwen3MoeModelForCausalLM::new(cfg, vb).map_err(candle_model_err)?,
            ))
        }

        fn runtime_caps(
            &self,
            task: TaskKind,
            backend: WeightsBackend,
            device: &crate::tensor::Device,
        ) -> RuntimeCaps {
            let is_safetensors = backend == WeightsBackend::Safetensors;
            let is_generate = task == TaskKind::Generate;

            let is_cuda = device.is_cuda();
            // CUDA graph capture for MoE decode. cudarc 0.19's
            // `CudaStream::alloc` uses `cuMemAllocAsync` whenever the
            // device supports memory pools (true on SM90+), which IS
            // graph-capturable — so most intermediate tensor allocations
            // are safe during capture.
            //
            // Known remaining blockers fire on PREFILL, not decode:
            //   - `moe_sort_experts_gpu` (thrust::sort_by_key) — only
            //     used when n > 1024 assignments; decode max_bs=32 caps
            //     n=128 so candle's bitonic sort is used instead.
            //   - `calculate_expert_offsets` (thrust::inclusive_scan) —
            //     only used when `is_prefill=true`; decode uses the
            //     custom `_light` kernel path.
            //
            // Default ON — bench measured 16ms → 10ms TPOT (and 3.55
            // → 5.33 RPS @ conc=32) on Qwen3-30B-A3B with no failures
            // across 200 requests. PRELUDE_MOE_CUDA_GRAPH=0 keeps the
            // historical opt-out for triage.
            let moe_graph = std::env::var("PRELUDE_MOE_CUDA_GRAPH")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .map(|v| v != 0)
                .unwrap_or(true);
            RuntimeCaps {
                supports_kv_cache: is_safetensors && is_generate,
                supports_prefix_cache: is_safetensors && is_cuda,
                supports_paged_attn: is_cuda && is_safetensors,
                supports_varlen: is_cuda && is_safetensors,
                supports_deltanet: false,
                // Must AND with `is_cuda`: the env-var check above doesn't
                // know what device we're on, and a CPU run with the flag
                // set would otherwise tell DecodeGraphCache the model
                // supports graphs on a non-CUDA device.
                supports_cuda_graph: is_cuda && moe_graph,
            }
        }
    }
}
