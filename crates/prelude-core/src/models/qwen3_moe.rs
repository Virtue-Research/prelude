use std::sync::Arc;

use crate::tensor::{DType, Device, Module, Result, Tensor, D};
use crate::loading::var_builder::VarBuilder;
use crate::models::commons::activation::Activation;
use crate::models::commons::embedding::Embedding;
use crate::models::commons::linear::DenseLinear;
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
    gate: DenseLinear,
    experts: Vec<Qwen3MoeExpert>,
    // Stacked expert weights for the legacy WMMA grouped path.
    gate_w: Option<Tensor>,
    up_w: Option<Tensor>,
    down_w: Option<Tensor>,
    // Pre-fused CUTLASS layout: [num_experts, 2*inter, hidden] ([up;gate])
    // and [num_experts, hidden, inter]. Enables FlashInfer's fused MoE
    // on SM90+ (3-5× faster than WMMA on Blackwell).
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

        // Stack expert weights for fused MoE GEMM (GPU only).
        //
        // We build TWO layouts from the same source weights:
        //   (a) `gate_w`, `up_w`, `down_w` — [num_experts, N, K] stacks used
        //       by the legacy WMMA path.
        //   (b) `experts_gate_up`, `experts_down` — the CUTLASS layout used
        //       by FlashInfer's fused MoE. `experts_gate_up` is
        //       [num_experts, 2*inter, hidden] with [up; gate] concatenation
        //       (CUTLASS Swiglu convention — Qwen3.5 does the same via a
        //       pre-stored gate_up_proj + `swap_moe_gate_up`).
        //
        // Keeping both layouts is ~2× the MoE weight memory (~2GB for 15B-A3B
        // in BF16). Acceptable given B300's 275GB VRAM, and unlocks a 3–5×
        // fused-MoE speedup that WMMA can't match on SM100.
        let (gate_w, up_w, down_w, experts_gate_up, experts_down) =
            if experts.first().map_or(false, |e| e.gate_proj.weight().device().is_cuda()) {
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

                // Build the CUTLASS [up; gate] stack. `Tensor::cat` along
                // dim 1 gives [num_experts, 2*inter, hidden] which is what
                // `cutlass_fused_moe` expects. `.contiguous()` can OOM on
                // a doubled weight stack — propagate that with `?` instead
                // of swallowing into `None`, which would silently downgrade
                // to the WMMA fallback at every forward without any log.
                let experts_gate_up = Some(Tensor::cat(&[&up_w, &gate_w], 1)?.contiguous()?);
                let experts_down = Some(down_w.clone());

                (
                    Some(gate_w),
                    Some(up_w),
                    Some(down_w),
                    experts_gate_up,
                    experts_down,
                )
            } else {
                (None, None, None, None, None)
            };

        Ok(Self {
            gate,
            experts,
            gate_w,
            up_w,
            down_w,
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
        // renorm, FP32 softmax internally). Note: fused_moe_routing ALSO
        // emits sorted_expert_ids/sorted_token_ids, but that sort is only
        // *per-token* (within each token's topk assignments), not the
        // *global* sort-by-expert that the WMMA/GEMV grouped GEMM
        // downstream requires. Ignore those two outputs here; the caller
        // does a proper global sort via `sort_expert_assignments`.
        if let Some(result) = ops.fused_moe_routing(
            &router_logits, self.num_experts_per_tok, self.norm_topk_prob,
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

    /// Sort expert assignments by expert ID to produce sorted_token_ids and sorted_expert_ids.
    ///
    /// Prefer the `ops.moe_sort_experts` GPU path (thrust::sort_by_key) on
    /// CUDA — it's O(n log n) on-device with no sync. The CPU fallback
    /// below was being hit for every prefill step of > 1024 assignments
    /// (prefill of ≥ 256 tokens at top-4 from 64 experts), forcing a
    /// GPU→CPU copy + CPU sort + CPU→GPU copy per MoE layer × 48 layers.
    /// qwen3_5.rs has used the fast path for a while; qwen3_moe was
    /// overlooked.
    fn sort_expert_assignments(
        &self,
        ops: &dyn crate::ops::Ops,
        experts_per_tok: &Tensor,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let flat = experts_per_tok.flatten_all()?;
        let n = flat.elem_count();

        // For decode batches (n ≤ 1024 assignments), candle's
        // `sort_last_dim` is a small bitonic sort kernel with less fixed
        // overhead than thrust::sort_by_key — measured ~2ms/tok
        // regression when we forced thrust here for decode. For prefill
        // (n > 1024) candle's sort would fall back to CPU, so we use the
        // `ops.moe_sort_experts` thrust path there.
        if device.is_cuda() && n <= 1024 {
            let flat_2d = flat.reshape((1, n))?;
            let (sorted_vals, sorted_idx) = flat_2d.sort_last_dim(true)?;
            return Ok((sorted_vals.flatten_all()?, sorted_idx.flatten_all()?));
        }

        if let Some(result) = ops.moe_sort_experts(&flat) {
            return result;
        }

        // CPU fallback (non-CUDA).
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
            self.sort_expert_assignments(ops, experts_per_tok, xs.device())?;

        // `num_tokens_per_expert` used to be computed here via a GPU→CPU→GPU
        // histogram pass, but the CUDA impl of `grouped_gemm` ignores it.
        // Pass a zero-length placeholder to preserve the trait signature.
        let num_tokens_per_expert = Tensor::zeros((0,), DType::U32, xs.device())?;

        let gate = ops.grouped_gemm(
            xs, gate_w,
            &sorted_token_ids, &sorted_expert_ids, &num_tokens_per_expert,
        )?;
        let up = ops.grouped_gemm(
            xs, up_w,
            &sorted_token_ids, &sorted_expert_ids, &num_tokens_per_expert,
        )?;

        let down_input = ops.silu_mul(&gate, &up)?;

        // Always take the grouped-GEMM + external weighted-sum path.
        //
        // The alternative `fused_moe_gemm` call — which folds the topk
        // weighting and reduction into the MoE WMMA kernel — has a write
        // race when topk > 1: multiple expert blocks all store
        // `val * topk_weight` to `output[token_index * size_n + ..]`,
        // so the final value is the contribution of whichever warp wrote
        // last, rather than the sum of all four. That's bit-nondeterministic
        // across batch compositions (different scheduling → different
        // "winner"), which is what made prefix-cache reuse drift from
        // fresh compute by enough to flip the first predicted token.
        //
        // Using grouped_gemm (size_m = total_tokens * topk, one row per
        // (token, expert) pair, no collisions) + a Rust-side weighted
        // reduction is deterministic. A small extra launch + a multiply+sum
        // over (topk, hidden_dim) is cheap vs the kernel call itself.
        let raw = ops.grouped_gemm(
            &down_input, down_w,
            &sorted_token_ids, &sorted_expert_ids, &num_tokens_per_expert,
        )?;
        let raw = raw.reshape((total_tokens, self.num_experts_per_tok, hidden_dim))?;
        // topk_weights is F32 but raw is BF16. Candle's mul is strict on
        // both shape and dtype — use broadcast_mul + explicit cast.
        let w = topk_weights.unsqueeze(D::Minus1)?.to_dtype(raw.dtype())?;
        raw.broadcast_mul(&w)?.sum(D::Minus2)
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
        let (topk_weights, experts_per_tok, hidden_dim) = self.compute_routing_2d(ops, xs)?;

        // Preferred path: FlashInfer's CUTLASS fused MoE handles
        // gate+up+silu+down+topk-weighted-sum in one kernel. The ops impl
        // returns `None` on archs where it's not compiled (e.g. SM100
        // until upstream FlashInfer instantiates Blackwell MoE kernels);
        // those land in the WMMA grouped-GEMM fallback below.
        //
        // For `Some(Err(_))` — runtime kernel/launch failure on a
        // supported arch — we ALSO fall through to WMMA rather than
        // hard-failing the whole forward. Otherwise a transient
        // FlashInfer registration glitch turns into a model that never
        // generates, when the grouped-GEMM path was working fine before
        // the CUTLASS fast path was added.
        if xs.device().is_cuda() {
            if let (Some(egu), Some(ed)) = (self.experts_gate_up.as_ref(), self.experts_down.as_ref()) {
                match ops.cutlass_fused_moe(xs, &experts_per_tok, &topk_weights, egu, ed) {
                    Some(Ok(out)) => return Ok(out),
                    Some(Err(e)) => {
                        tracing::warn!(error = %e, "CUTLASS fused MoE failed at runtime, falling back to WMMA grouped-GEMM");
                    }
                    None => {}
                }
            }
        }

        // Fallback 1: WMMA / GEMV grouped-GEMM path (works on older archs / missing cutlass).
        if xs.device().is_cuda() && self.gate_w.is_some() {
            return self.forward_fused_varlen(ops, xs, &topk_weights, &experts_per_tok, hidden_dim);
        }

        // Fallback 2: per-expert loop (CPU / debug).
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

        let input_layernorm = RmsNorm::load(vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let post_attention_layernorm = RmsNorm::load(vb.pp("post_attention_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self { self_attn, feed_forward, input_layernorm, post_attention_layernorm })
    }

    fn forward(&self, hidden: &Tensor, residual: Option<&Tensor>, ctx: &LayerAttnContext) -> Result<(Tensor, Tensor)> {
        let ops = ctx.ops;
        let (residual, hidden) = self.input_layernorm.forward_residual(hidden, residual, ops)?;
        let hidden = self.self_attn.forward(&hidden, ctx)?;
        let (residual, hidden) = self.post_attention_layernorm.forward_residual(&hidden, Some(&residual), ops)?;
        let hidden = self.feed_forward.forward_2d(ops, &hidden)?;
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

        let norm = RmsNorm::load(vb.pp("model.norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self { embed_tokens, layers, norm })
    }

    fn clear_kv_cache(&mut self) {
        // No internal KV cache — paged KV is managed externally.
    }

    fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let mut hidden = self.embed_tokens.forward(packed_input)?;
        let mut residual: Option<Tensor> = None;
        for (i, layer) in self.layers.iter_mut().enumerate() {
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
        }
        let (_, normed) = self.norm.forward_residual(&hidden, residual.as_ref(), ctx.ops)?;
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
