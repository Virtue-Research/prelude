use std::sync::Arc;

use crate::tensor::{DType, Device, Module, Result, Tensor, D};
use crate::loading::var_builder::VarBuilder;
use crate::models::commons::activation::Activation;
use crate::models::commons::embedding::Embedding;
use crate::models::commons::linear::NaiveLinear;
use crate::models::config::Qwen3Config;

// Shared layer primitives (extracted from qwen3 into reusable modules)
use crate::models::commons::{
    last_token_select,
    BatchAttnContext, BatchState, LayerAttnContext, RotaryEmbedding, Linear, RmsNorm,
};

// Model-specific attention (still lives in qwen3)
use super::qwen3::Qwen3Attention;

// ── Config ──────────────────────────────────────────────────────────────

use crate::models::model_config;

model_config! {
    pub struct Qwen3MoeConfig("Qwen3MoE") {
        required {
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
            num_experts: usize,
        }
        serde_default {
            attention_bias: bool,
            sliding_window: Option<usize>,
            max_window_layers: usize,
            tie_word_embeddings: bool,
            use_sliding_window: bool,
            norm_topk_prob: bool,
        }
        warn_default {
            rope_theta: f64 = 1_000_000.0,
            rms_norm_eps: f64 = 1e-6,
            hidden_act: Activation = Activation::Silu,
            decoder_sparse_step: usize = 1,
        }
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
            gate_proj: Linear::load(vb.pp("gate_proj"), cfg.hidden_size, cfg.moe_intermediate_size, false)?,
            up_proj: Linear::load(vb.pp("up_proj"), cfg.hidden_size, cfg.moe_intermediate_size, false)?,
            down_proj: Linear::load(vb.pp("down_proj"), cfg.moe_intermediate_size, cfg.hidden_size, false)?,
        })
    }

    fn forward(&self, ops: &dyn crate::ops::Ops, x: &Tensor) -> Result<Tensor> {
        let bs = BatchState::no_lora();
        let gate = self.gate_proj.forward(x, &bs, ops)?;
        let up = self.up_proj.forward(x, &bs, ops)?;
        self.down_proj.forward(&ops.silu_mul(&gate, &up)?, &bs, ops)
    }
}

fn count_tokens_per_expert(sorted_expert_ids: &Tensor, num_experts: usize, device: &Device) -> Result<Tensor> {
    let ids: Vec<u32> = sorted_expert_ids.to_vec1()?;
    let mut counts = vec![0u32; num_experts];
    for &id in &ids {
        if (id as usize) < num_experts { counts[id as usize] += 1; }
    }
    Tensor::from_vec(counts, (num_experts,), device)
}

// ── Sparse MoE Block ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Qwen3SparseMoeBlock {
    gate: NaiveLinear,
    experts: Vec<Qwen3MoeExpert>,
    // Stacked expert weights for fused GEMM [num_experts, N, K]
    gate_w: Option<Tensor>,
    up_w: Option<Tensor>,
    down_w: Option<Tensor>,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl Qwen3SparseMoeBlock {
    fn new(cfg: &Qwen3MoeConfig, vb: VarBuilder) -> Result<Self> {
        let gate = {
            let gvb = vb.pp("gate");
            let w = gvb.get((cfg.num_experts, cfg.hidden_size), "weight")?;
            NaiveLinear::new(w, None)
        };
        let mut experts = Vec::with_capacity(cfg.num_experts);
        let vb_e = vb.pp("experts");
        for idx in 0..cfg.num_experts {
            experts.push(Qwen3MoeExpert::new(cfg, vb_e.pp(idx))?);
        }

        // Stack expert weights for fused MoE GEMM (GPU only)
        let (gate_w, up_w, down_w) = if experts.first().map_or(false, |e| e.gate_proj.weight().device().is_cuda()) {
            let gate_ws: Vec<Tensor> = experts
                .iter()
                .map(|e| e.gate_proj.weight().clone())
                .collect();
            let up_ws: Vec<Tensor> = experts.iter().map(|e| e.up_proj.weight().clone()).collect();
            let down_ws: Vec<Tensor> = experts
                .iter()
                .map(|e| e.down_proj.weight().clone())
                .collect();
            (
                Some(Tensor::stack(&gate_ws, 0)?.contiguous()?),
                Some(Tensor::stack(&up_ws, 0)?.contiguous()?),
                Some(Tensor::stack(&down_ws, 0)?.contiguous()?),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            gate,
            experts,
            gate_w,
            up_w,
            down_w,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }

    /// Compute routing for 2D (varlen) input.
    /// Returns (topk_weights, experts_per_tok, hidden_dim).
    fn compute_routing_2d(&self, xs: &Tensor) -> Result<(Tensor, Tensor, usize)> {
        let (_total_tokens, hidden_dim) = xs.dims2()?;
        let router_logits = xs.apply(&self.gate)?;
        let routing_weights = router_logits.softmax(D::Minus1)?;

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

    /// Sort expert assignments by expert ID to produce sorted_token_ids and sorted_expert_ids.
    /// Uses GPU bitonic sort for small arrays (<=1024 elements, fits in shared memory),
    /// falls back to CPU argsort for larger arrays (prefill with many tokens).
    fn sort_expert_assignments(
        &self,
        experts_per_tok: &Tensor,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let flat = experts_per_tok.flatten_all()?;
        let n = flat.elem_count();

        // GPU sort for small arrays: shared_mem = next_power_of_2(n) * 4 bytes.
        // For n <= 1024, that's at most 4096 bytes — well within CUDA limits.
        if n <= 1024 && device.is_cuda() {
            let flat_2d = flat.reshape((1, n))?;
            let (sorted_vals, sorted_idx) = flat_2d.sort_last_dim(true)?;
            return Ok((sorted_vals.flatten_all()?, sorted_idx.flatten_all()?));
        }

        // CPU fallback for large prefills (>128 tokens × 8 experts = >1024 elements)
        let flat_vec = flat.to_vec1::<u32>()?;
        let mut indices: Vec<u32> = (0..n as u32).collect();
        indices.sort_by_key(|&i| flat_vec[i as usize]);
        let sorted_expert_ids: Vec<u32> = indices.iter().map(|&i| flat_vec[i as usize]).collect();
        Ok((
            Tensor::from_vec(sorted_expert_ids, (n,), device)?,
            Tensor::from_vec(indices, (n,), device)?,
        ))
    }

    /// Fused MoE forward for varlen (2D) input.
    fn forward_fused_varlen(
        &self,
        ops: &dyn crate::ops::Ops,
        xs: &Tensor,
        topk_weights: &Tensor,
        experts_per_tok: &Tensor,
        hidden_dim: usize,
    ) -> Result<Tensor> {
        let gate_w = self.gate_w.as_ref().unwrap();
        let up_w = self.up_w.as_ref().unwrap();
        let down_w = self.down_w.as_ref().unwrap();
        let (total_tokens, _) = xs.dims2()?;
        let (sorted_expert_ids, sorted_token_ids) =
            self.sort_expert_assignments(experts_per_tok, xs.device())?;

        let is_prefill = total_tokens > 1;
        let num_tokens_per_expert = count_tokens_per_expert(&sorted_expert_ids, self.experts.len(), xs.device())?;

        let gate = ops.grouped_gemm(
            xs, gate_w,
            &sorted_token_ids, &sorted_expert_ids, &num_tokens_per_expert,
        )?;

        let up = ops.grouped_gemm(
            xs, up_w,
            &sorted_token_ids, &sorted_expert_ids, &num_tokens_per_expert,
        )?;

        let down_input = ops.silu_mul(&gate, &up)?;

        let ys = match ops.fused_moe_gemm(
            &down_input, down_w, topk_weights,
            &sorted_token_ids, &sorted_expert_ids,
            self.num_experts_per_tok, is_prefill,
        ) {
            Some(r) => r?,
            None => {
                let raw = ops.grouped_gemm(
                    &down_input, down_w,
                    &sorted_token_ids, &sorted_expert_ids, &num_tokens_per_expert,
                )?;
                let raw = raw.reshape((total_tokens, self.num_experts_per_tok, hidden_dim))?;
                let w = topk_weights.unsqueeze(D::Minus1)?;
                return (raw * w)?.sum(D::Minus2);
            }
        };

        ys.reshape((total_tokens, self.num_experts_per_tok, hidden_dim))?
            .sum(D::Minus2)
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
    fn forward_varlen(&self, ops: &dyn crate::ops::Ops, xs: &Tensor) -> Result<Tensor> {
        let (topk_weights, experts_per_tok, hidden_dim) = self.compute_routing_2d(xs)?;

        if xs.device().is_cuda() && self.gate_w.is_some() {
            return self.forward_fused_varlen(ops, xs, &topk_weights, &experts_per_tok, hidden_dim);
        }

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
        let gate_proj = Linear::load(vb.pp("gate_proj"), cfg.hidden_size, cfg.intermediate_size, false)?;
        let up_proj = Linear::load(vb.pp("up_proj"), cfg.hidden_size, cfg.intermediate_size, false)?;
        let down_proj = Linear::load(vb.pp("down_proj"), cfg.intermediate_size, cfg.hidden_size, false)?;

        let gate_up_proj = {
            let gw = gate_proj.weight();
            if gw.device().is_cpu() && gw.dtype() == DType::BF16 {
                let merged_w = Tensor::cat(&[gw, up_proj.weight()], 0)?;
                Linear::from_weight(merged_w, None).ok()
            } else {
                None
            }
        };

        Ok(Self { gate_proj, up_proj, down_proj, gate_up_proj })
    }

    fn forward(&self, ctx: &BatchState, ops: &dyn crate::ops::Ops, x: &Tensor) -> Result<Tensor> {
        if let Some(ref gup) = self.gate_up_proj {
            let gate_up = gup.forward(x, ctx, ops)?;
            let dims = gate_up.dims();
            let dim = dims[dims.len() - 1] / 2;
            let gate = gate_up.narrow(dims.len() - 1, 0, dim)?;
            let up = gate_up.narrow(dims.len() - 1, dim, dim)?;
            return self.down_proj.forward(&ops.silu_mul(&gate, &up)?, ctx, ops);
        }
        let gate = self.gate_proj.forward(x, ctx, ops)?;
        let up = self.up_proj.forward(x, ctx, ops)?;
        self.down_proj.forward(&ops.silu_mul(&gate, &up)?, ctx, ops)
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
            Self::Mlp(mlp) => mlp.forward(&BatchState::no_lora(), ops, x),
            Self::SparseMoe(moe) => moe.forward_varlen(ops, x),
        }
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MoeDecoderLayer {
    self_attn: Qwen3Attention,
    feed_forward: MoeFeedForward,
    ln1_weight: Tensor,
    ln2_weight: Tensor,
    rms_norm_eps: f32,
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

        let ln1_weight = vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?;
        let ln2_weight = vb.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?;

        Ok(Self {
            self_attn, feed_forward, ln1_weight, ln2_weight,
            rms_norm_eps: cfg.rms_norm_eps as f32,
        })
    }

    fn forward(&self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
        let ops = ctx.ops;
        let h = ops.rms_norm(x, &self.ln1_weight, self.rms_norm_eps)?;
        let h = self.self_attn.forward(&h, ctx)?;
        let (x_res, h2) = ops.add_rmsnorm(x, &h, &self.ln2_weight, self.rms_norm_eps)?;
        ops.add_or_fused(&x_res, &self.feed_forward.forward_2d(ops, &h2)?)
    }
}

// ── Model ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MoeModel {
    embed_tokens: Embedding,
    layers: Vec<MoeDecoderLayer>,
    norm: RmsNorm,
    norm_weight: Tensor,
    rms_norm_eps: f64,
}

impl MoeModel {
    fn new(cfg: &Qwen3MoeConfig, vb: VarBuilder) -> Result<Self> {
        let dense_cfg = cfg.to_dense_config();
        let embed_tokens = {
            let emb_vb = vb.pp("model.embed_tokens");
            let weight = emb_vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            Embedding::new(weight, cfg.hidden_size)
        };
        let rotary = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            &dense_cfg,
            vb.device(),
        )?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MoeDecoderLayer::new(i, cfg, rotary.clone(), vb_l.pp(i))?);
        }

        let norm_weight = vb.pp("model.norm").get(cfg.hidden_size, "weight")?;

        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm::from_weight(norm_weight.clone(), cfg.rms_norm_eps),
            norm_weight,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    fn clear_kv_cache(&mut self) {
        // No internal KV cache — paged KV is managed externally.
    }

    fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let mut h = self.embed_tokens.forward(packed_input)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_kv = ctx.paged_kv.map(|kv| kv.layer(i));
            let layer_ctx = LayerAttnContext {
                ops: ctx.ops,
                cu_seqlens_q: ctx.cu_seqlens_q,
                max_seqlen_q: ctx.max_seqlen_q,
                position_ids: ctx.position_ids,
                paged_kv: layer_kv.as_ref(),
            };
            h = layer.forward(&h, &layer_ctx)?;
        }
        ctx.ops.rms_norm(&h, &self.norm_weight, self.rms_norm_eps as f32)
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
            &BatchState::no_lora(), ctx.ops,
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
        self.lm_head.forward(hidden, &BatchState::no_lora(), crate::ops::select_ops(hidden.device()))
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
    use crate::models::registry::{
        candle_model_err, parse_json, ArchSpec, ParsedModelConfig,
    };

    const ARCHITECTURE_ALIASES: &[&str] = &["Qwen3Moe", "Qwen3MoeModel"];
    const MODEL_TYPE_ALIASES: &[&str] = &["qwen3_moe"];
    const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate];

    pub(crate) struct Qwen3MoeArchSpec;

    pub(crate) static QWEN3_MOE_ARCH_SPEC: Qwen3MoeArchSpec = Qwen3MoeArchSpec;
    inventory::submit!(crate::models::registry::ArchSpecEntry::new(&QWEN3_MOE_ARCH_SPEC));

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
            RuntimeCaps {
                supports_kv_cache: is_safetensors && is_generate,
                supports_prefix_cache: is_safetensors && is_cuda,
                supports_paged_attn: is_cuda && is_safetensors,
                supports_varlen: is_cuda && is_safetensors,
                supports_deltanet: false,
                supports_cuda_graph: false,
            }
        }
    }
}
