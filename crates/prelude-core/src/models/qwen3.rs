
use std::collections::HashMap;
use std::sync::Arc;

use crate::tensor::{DType, Module, Result, Tensor};
use crate::loading::var_builder::VarBuilder;
use crate::models::commons::embedding::Embedding;
use crate::models::config::Qwen3Config;
use serde::Deserialize;

use crate::models::commons::{BatchAttnContext, BatchState, LayerAttnContext};
use crate::models::commons::{
    Linear, RmsNorm, RotaryEmbedding,
    last_token_select,
};
use crate::profiling::{nvtx_push, nvtx_pop};

use crate::models::{ClassifierModel, EmbeddingModel, KvCacheModel, LogitsSplitModel, ModelForward};
use crate::cache::kv_buf::KvBuf;

// ============================================================================
// Attention
// ============================================================================

#[derive(Debug, Clone)]
pub(crate) struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    rms_norm_eps: f64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    is_cuda: bool,
    softmax_scale: f32,
    qkv_proj: Option<Linear>,
    // CPU KV cache for decode (populated by forward_with_cache)
    k_cache: KvBuf,
    v_cache: KvBuf,
}

impl Qwen3Attention {
    pub(crate) fn new(
        cfg: &Qwen3Config,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        if cfg.use_sliding_window {
            crate::tensor::bail!("sliding window is not supported yet");
        }

        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let hidden_size = head_dim * num_heads;

        let q_norm_vb = vb.pp("q_norm");
        let q_norm_weight_raw = q_norm_vb.get(head_dim, "weight")?;
        let q_norm = RmsNorm::from_weight(q_norm_weight_raw.clone(), cfg.rms_norm_eps);
        let k_norm_vb = vb.pp("k_norm");
        let k_norm_weight = k_norm_vb.get(head_dim, "weight")?;
        let k_norm = RmsNorm::from_weight(k_norm_weight.clone(), cfg.rms_norm_eps);
        let is_cuda = vb.device().is_cuda();

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
        let v_proj = Linear::load(
            vb.pp("v_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
        )?;

        // Fused QKV: merge q_proj+k_proj+v_proj weights into single GEMM (saves 2 GEMM calls/layer)
        let qkv_proj = {
            let merged_w =
                Tensor::cat(&[q_proj.weight(), k_proj.weight(), v_proj.weight()], 0)?;
            match Linear::from_weight(merged_w, None) {
                Ok(l) => Some(l),
                Err(e) => {
                    tracing::warn!("Failed to create fused qkv_proj: {e}");
                    None
                }
            }
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj: Linear::load(
                vb.pp("o_proj"),
                num_heads * head_dim,
                cfg.hidden_size,
                cfg.attention_bias,
            )?,
            q_norm,
            k_norm,
            q_norm_weight: q_norm_weight_raw,
            k_norm_weight,
            rms_norm_eps: cfg.rms_norm_eps,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            rotary_emb,
            is_cuda,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            qkv_proj,
            k_cache: KvBuf::new(),
            v_cache: KvBuf::new(),
        })
    }

    // ── shared QKV / norm+rope helpers ──────────────────────────────────

    /// Project Q, K, V and reshape to `[total_tokens, H, D]` (varlen layout).
    #[inline]
    fn fused_qkv_projection(&self, x: &Tensor, total: usize, ctx: &BatchState, ops: &dyn crate::ops::Ops) -> Result<(Tensor, Tensor, Tensor)> {
        crate::models::commons::attn_utils::fused_qkv_projection(
            x,
            &self.q_proj, &self.k_proj, &self.v_proj,
            self.qkv_proj.as_ref(),
            total, self.num_heads, self.num_kv_heads, self.head_dim,
            ctx, ops,
        )
    }

    // ── Varlen forward (unified: GPU flash-attn / CPU cpu_ops) ──────────

    pub(crate) fn forward(&self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
        let total_q = x.dim(0)?;
        let bs = BatchState::no_lora();
        let ops = ctx.ops;

        let (q, k, v) = self.fused_qkv_projection(x, total_q, &bs, ops)?;

        // QK-norm + RoPE + optional cache write (OpsBundle picks optimal fuse path)
        let kv_cache = ctx.paged_kv.map(|kv| (kv.key_cache, kv.value_cache, kv.slot_mapping));
        let (q, k) = ops.qknorm_rope_and_cache(
            &q, &k, &v,
            &self.q_norm_weight, &self.k_norm_weight,
            &self.rotary_emb.cos, &self.rotary_emb.sin,
            ctx.position_ids, self.rms_norm_eps as f32,
            kv_cache,
        )?;

        // Attention
        let attn_out = if let Some(kv) = ctx.paged_kv {
            ops.paged_attention(&q, kv.key_cache, kv.value_cache, &crate::ops::PagedParams {
                block_tables: kv.block_tables,
                cu_seqlens_q: ctx.cu_seqlens_q, cu_seqlens_k: kv.cu_seqlens_k,
                max_seqlen_q: ctx.max_seqlen_q, max_seqlen_k: kv.max_seqlen_k,
                scale: self.softmax_scale, mask: crate::ops::MaskType::Causal, softcap: None,
            })?
        } else {
            ops.varlen_attention(&q, &k, &v, &crate::ops::VarlenParams {
                cu_seqlens_q: ctx.cu_seqlens_q, cu_seqlens_k: ctx.cu_seqlens_q,
                max_seqlen_q: ctx.max_seqlen_q, max_seqlen_k: ctx.max_seqlen_q,
                scale: self.softmax_scale, mask: crate::ops::MaskType::Causal, softcap: None,
            })?
        };

        self.o_proj.forward(&attn_out.reshape((total_q, self.hidden_size))?, &bs, ops)
    }

    /// Cached forward for CPU decode: handles both prefill (L=prompt_len) and
    /// decode (L=1). KV cache accumulates across calls; call `reset_kv_cache`
    /// between requests.
    ///
    /// KV cache is stored in varlen format `[total, H_kv, D]`.
    /// Uses matmul-based SDPA as a universal fallback.
    fn forward_with_cache(&mut self, x: &Tensor, position_offset: usize) -> Result<Tensor> {
        let seq_len = x.dim(0)?;
        let bs = BatchState::no_lora();

        // ── Tensor-based path ──
        let ops = crate::ops::select_ops(x.device());
        let (q, k, v) = self.fused_qkv_projection(x, seq_len, &bs, ops)?;
        let position_ids: Vec<u32> =
            (0..seq_len).map(|i| (position_offset + i) as u32).collect();
        let position_ids = Tensor::from_vec(position_ids, (seq_len,), x.device())?;
        let (q, k) = ops.qknorm_rope_and_cache(
            &q, &k, &v,
            &self.q_norm_weight, &self.k_norm_weight,
            &self.rotary_emb.cos, &self.rotary_emb.sin,
            &position_ids, self.rms_norm_eps as f32,
            None,  // no paged KV cache
        )?;

        self.k_cache.append(&k)?;
        self.v_cache.append(&v)?;
        let k_full = self.k_cache.view()?;
        let v_full = self.v_cache.view()?;

        let attn_out = self.matmul_cross_attention(
            &q, &k_full, &v_full, seq_len, position_offset,
        )?;

        self.o_proj.forward(
            &attn_out.reshape((seq_len, self.hidden_size))?,
            &bs, ops,
        )
    }

    /// Matmul-based cross-attention for decode: Q attends to full KV cache.
    /// Q: [seq_len, H, D], K: [total_kv, H_kv, D], V: [total_kv, H_kv, D]
    /// Returns: [seq_len, H, D]
    fn matmul_cross_attention(
        &self,
        q: &Tensor,
        k_full: &Tensor,
        v_full: &Tensor,
        seq_len: usize,
        position_offset: usize,
    ) -> Result<Tensor> {
        let total_kv_len = k_full.dim(0)?;
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let causal = seq_len > 1;

        // Convert to F32 for numeric stability
        let q_f32 = q.to_dtype(DType::F32)?;
        let k_f32 = k_full.to_dtype(DType::F32)?;
        let v_f32 = v_full.to_dtype(DType::F32)?;

        // Reshape to [H, seq_len, D] and [H_kv, total_kv, D]
        let q3 = q_f32
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let k3 = k_f32
            .reshape((total_kv_len, self.num_kv_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let v3 = v_f32
            .reshape((total_kv_len, self.num_kv_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;

        // Expand K,V for GQA: [H_kv, T, D] -> [H, T, D]
        let k3 = if num_kv_groups > 1 {
            k3.unsqueeze(1)?
                .expand((self.num_kv_heads, num_kv_groups, total_kv_len, self.head_dim))?
                .reshape((self.num_heads, total_kv_len, self.head_dim))?
                .contiguous()?
        } else {
            k3
        };
        let v3 = if num_kv_groups > 1 {
            v3.unsqueeze(1)?
                .expand((self.num_kv_heads, num_kv_groups, total_kv_len, self.head_dim))?
                .reshape((self.num_heads, total_kv_len, self.head_dim))?
                .contiguous()?
        } else {
            v3
        };

        // scores = Q @ K^T * scale  → [H, seq_len, total_kv]
        let scores = q3.matmul(&k3.transpose(1, 2)?)?;
        let scores = (scores * (self.softmax_scale as f64))?;

        // Apply causal mask if needed
        let scores = if causal {
            // Build lower-triangular causal mask with position_offset
            let mask_data: Vec<f32> = (0..seq_len)
                .flat_map(|qi| {
                    let qi_abs = position_offset + qi;
                    (0..total_kv_len).map(move |ki| {
                        if ki <= qi_abs { 0.0f32 } else { f32::NEG_INFINITY }
                    })
                })
                .collect();
            let mask = Tensor::from_vec(mask_data, (seq_len, total_kv_len), q.device())?;
            scores.broadcast_add(&mask)?
        } else {
            scores
        };

        // Softmax over last dim
        let _last_dim = scores.rank() - 1;
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;

        // output = attn_weights @ V → [H, seq_len, D]
        let out = attn_weights.matmul(&v3)?;

        // Transpose back: [H, seq_len, D] -> [seq_len, H, D]
        let out = out.transpose(0, 1)?.contiguous()?;
        out.to_dtype(v_full.dtype())
    }

    fn reset_kv_cache(&mut self) {
        self.k_cache.reset();
        self.v_cache.reset();
    }
}

// ============================================================================
// Gated MLP (SiLU-gated FFN: down_proj(SiLU(gate_proj(x)) * up_proj(x)))
// ============================================================================

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
            // Try silu_mul_concat: splits [tokens, 2*dim] internally, avoids narrow+copy
            if let Some(r) = ops.silu_mul_concat(&gate_up) {
                return self.down_proj.forward(&r?, &bs, ops);
            }
            // Fallback: narrow (triggers contiguous copy) + separate silu_mul
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

// ============================================================================
// Decoder Layer
// ============================================================================

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: GatedMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Qwen3Config,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
        _layer_idx: usize,
    ) -> Result<Self> {
        let self_attn = Qwen3Attention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = GatedMlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = RmsNorm::load(vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let post_attention_layernorm = RmsNorm::load(vb.pp("post_attention_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm })
    }

    fn forward(&self, hidden: &Tensor, residual: Option<&Tensor>, ctx: &LayerAttnContext) -> Result<(Tensor, Tensor)> {
        let ops = ctx.ops;
        let (residual, hidden) = self.input_layernorm.forward_residual(hidden, residual, ops)?;
        let hidden = self.self_attn.forward(&hidden, ctx)?;
        let (residual, hidden) = self.post_attention_layernorm.forward_residual(&hidden, Some(&residual), ops)?;
        let hidden = self.mlp.forward(&hidden, ops)?;
        Ok((hidden, residual))
    }

    fn forward_with_cache(&mut self, hidden: &Tensor, residual: Option<&Tensor>, ops: &dyn crate::ops::Ops, position_offset: usize) -> Result<(Tensor, Tensor)> {
        let (residual, hidden) = self.input_layernorm.forward_residual(hidden, residual, ops)?;
        let hidden = self.self_attn.forward_with_cache(&hidden, position_offset)?;
        let (residual, hidden) = self.post_attention_layernorm.forward_residual(&hidden, Some(&residual), ops)?;
        let hidden = self.mlp.forward(&hidden, ops)?;
        Ok((hidden, residual))
    }

    fn reset_kv_cache(&mut self) {
        self.self_attn.reset_kv_cache();
    }
}

// ============================================================================
// Model (backbone)
// ============================================================================

#[derive(Debug, Clone)]
struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
}

impl Model {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with_prefix(cfg, vb, "model")
    }

    fn new_with_prefix(cfg: &Qwen3Config, vb: VarBuilder, prefix: &str) -> Result<Self> {
        let (embed_vb, layers_vb, norm_vb) = if prefix.is_empty() {
            (vb.pp("embed_tokens"), vb.pp("layers"), vb.pp("norm"))
        } else {
            (
                vb.pp(format!("{}.embed_tokens", prefix)),
                vb.pp(format!("{}.layers", prefix)),
                vb.pp(format!("{}.norm", prefix)),
            )
        };
        let embed_tokens = {
            let weight = embed_vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            Embedding::new(weight, cfg.hidden_size)
        };
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                rotary.clone(),
                layers_vb.pp(i),
                i,
            )?);
        }
        let norm = RmsNorm::load(norm_vb, cfg.hidden_size, cfg.rms_norm_eps)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.reset_kv_cache();
        }
    }

    fn forward_with_cache(
        &mut self,
        ops: &dyn crate::ops::Ops,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let mut residual: Option<Tensor> = None;
        for (_i, layer) in self.layers.iter_mut().enumerate() {
            nvtx_push!("layer[{}]", _i);
            let (h, r) = layer.forward_with_cache(&hidden, residual.as_ref(), ops, position_offset)?;
            hidden = h;
            residual = Some(r);
            nvtx_pop!();
        }
        let (_, normed) = self.norm.forward_residual(&hidden, residual.as_ref(), ops)?;
        Ok(normed)
    }

    fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let mut hidden = self.embed_tokens.forward(packed_input)?;
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
        let (_, normed) = self.norm.forward_residual(&hidden, residual.as_ref(), ctx.ops)?;
        Ok(normed)
    }
}

// ============================================================================
// Task Heads
// ============================================================================

#[derive(Debug, Clone)]
pub struct Qwen3ModelForCausalLM {
    base: Model,
    lm_head: Linear,
}

impl Qwen3ModelForCausalLM {
    pub fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let base = Model::new(cfg, vb.clone())?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weight(base.embed_tokens.embeddings().clone(), None)?
        } else {
            Linear::load(vb.pp("lm_head"), cfg.hidden_size, cfg.vocab_size, false)?
        };
        Ok(Self { base, lm_head })
    }

    /// Helper: extract last-token hidden per varlen sequence → logits.
    fn last_token_logits(&self, hidden: &Tensor, seq_lens: &[usize]) -> Result<Tensor> {
        let bs = BatchState::no_lora();
        self.lm_head.forward(
            &last_token_select(hidden, seq_lens)?.unsqueeze(1)?,
            &bs, crate::ops::select_ops(hidden.device()),
        )
    }

    pub fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let hidden = self.base.forward(packed_input, ctx)?;
        self.last_token_logits(&hidden, ctx.seq_lens)
    }

    /// Cached forward: returns logits `[L, vocab_size]` for all input tokens.
    pub fn forward_with_cache(
        &mut self,
        ops: &dyn crate::ops::Ops,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
        let hidden = self.base.forward_with_cache(ops, input_ids, position_offset)?;
        self.lm_head.forward(&hidden, &BatchState::no_lora(), ops)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}

// ── Sequence Classification ────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3ClassifierConfig {
    #[serde(flatten)]
    pub base: Qwen3Config,
    pub num_labels: usize,
    #[serde(default)]
    pub label2id: Option<HashMap<String, usize>>,
    #[serde(default)]
    pub id2label: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone)]
pub struct Qwen3ForSequenceClassification {
    base: Model,
    score: Linear,
    num_labels: usize,
    id2label: Option<HashMap<usize, String>>,
}

impl Qwen3ForSequenceClassification {
    pub fn new(cfg: &Qwen3ClassifierConfig, vb: VarBuilder) -> Result<Self> {
        let base = Model::new(&cfg.base, vb.clone())?;
        let score = Linear::load(vb.pp("score"), cfg.base.hidden_size, cfg.num_labels, false)?;
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

    pub fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let hidden_states = self.base.forward(packed_input, ctx)?;
        self.score.forward(&last_token_select(&hidden_states, ctx.seq_lens)?, &BatchState::no_lora(), ctx.ops)
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

// ── Embedding ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Qwen3ForEmbedding {
    base: Model,
    hidden_size: usize,
}

impl Qwen3ForEmbedding {
    pub fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            base: Model::new_with_prefix(cfg, vb, "")?,
            hidden_size: cfg.hidden_size,
        })
    }

    pub fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let hidden_states = self.base.forward(packed_input, ctx)?;
        last_token_select(&hidden_states, ctx.seq_lens)?.contiguous()
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}

// ── ModelForward implementations ─────────────────────────────────────────

impl LogitsSplitModel for Qwen3ModelForCausalLM {
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

impl KvCacheModel for Qwen3ModelForCausalLM {
    fn forward_with_cache(
        &mut self,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> crate::tensor::Result<Tensor> {
        Qwen3ModelForCausalLM::forward_with_cache(self, crate::ops::select_ops(input_ids.device()), input_ids, position_offset)
    }
}

impl ModelForward for Qwen3ModelForCausalLM {
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

    fn as_logits_model(&self) -> Option<&dyn LogitsSplitModel> {
        Some(self)
    }

    fn as_logits_model_mut(&mut self) -> Option<&mut dyn LogitsSplitModel> {
        Some(self)
    }

    fn as_kv_cache_model(&mut self) -> Option<&mut dyn KvCacheModel> {
        Some(self)
    }
}

impl ClassifierModel for Qwen3ForSequenceClassification {
    fn num_labels(&self) -> usize {
        Qwen3ForSequenceClassification::num_labels(self)
    }

    fn get_label(&self, class_idx: usize) -> Option<String> {
        Qwen3ForSequenceClassification::get_label(self, class_idx)
    }
}

impl ModelForward for Qwen3ForSequenceClassification {
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> crate::tensor::Result<Tensor> {
        Qwen3ForSequenceClassification::forward(self, packed_input, ctx)
    }

    fn clear_kv_cache(&mut self) {
        Qwen3ForSequenceClassification::clear_kv_cache(self);
    }

    fn as_classifier(&self) -> Option<&dyn ClassifierModel> {
        Some(self)
    }
}

impl EmbeddingModel for Qwen3ForEmbedding {
    fn embedding_dim(&self) -> usize {
        self.hidden_size()
    }
}

impl ModelForward for Qwen3ForEmbedding {
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> crate::tensor::Result<Tensor> {
        Qwen3ForEmbedding::forward(self, packed_input, ctx)
    }

    fn clear_kv_cache(&mut self) {
        Qwen3ForEmbedding::clear_kv_cache(self);
    }

    fn as_embedding(&self) -> Option<&dyn EmbeddingModel> {
        Some(self)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;
    use crate::loading::var_builder::VarBuilder;
    use crate::models::commons::BatchAttnContext;

    fn tiny_config() -> Qwen3Config {
        serde_json::from_value(serde_json::json!({
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "head_dim": 16,
            "num_key_value_heads": 2,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "use_sliding_window": false,
            "attention_bias": false,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "max_window_layers": 0,
        }))
        .expect("tiny_config deserialization failed")
    }

    fn build_model(cfg: &Qwen3Config) -> Qwen3ModelForCausalLM {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        Qwen3ModelForCausalLM::new(cfg, vb).expect("model construction failed")
    }

    fn forward_standard(
        model: &mut Qwen3ModelForCausalLM,
        tokens: &[u32],
        device: &Device,
    ) -> Tensor {
        let seq_len = tokens.len();
        let input = Tensor::from_vec(tokens.to_vec(), (seq_len,), device).unwrap();
        let cu = Tensor::from_vec(vec![0u32, seq_len as u32], (2,), device).unwrap();
        let pos = Tensor::from_vec((0..seq_len as u32).collect::<Vec<_>>(), (seq_len,), device).unwrap();
        let seq_lens = vec![seq_len];
        let ops = crate::ops::select_ops(device);
        let mut ctx = BatchAttnContext {
            ops, cu_seqlens_q: &cu, max_seqlen_q: seq_len, position_ids: &pos,
            seq_lens: &seq_lens, paged_kv: None, deltanet_pool: None, deltanet_slots: None,
        };
        let logits = model.forward(&input, &mut ctx).unwrap();
        model.clear_kv_cache();
        logits
    }

    #[test]
    fn test_kv_cache_prefill_matches_standard() {
        let cfg = tiny_config();
        let mut model = build_model(&cfg);
        let device = Device::Cpu;
        let tokens = vec![1u32, 5, 10, 20, 3];
        let seq_len = tokens.len();
        let std_logits = forward_standard(&mut model, &tokens, &device);
        let input = Tensor::from_vec(tokens, (seq_len,), &device).unwrap();
        let cached_logits = model.forward_with_cache(crate::ops::select_ops(&device), &input, 0).unwrap();
        model.clear_kv_cache();
        let std_flat: Vec<f32> = std_logits.flatten_all().unwrap().to_vec1().unwrap();
        let cached_last: Vec<f32> = cached_logits.get(seq_len - 1).unwrap().to_vec1().unwrap();
        assert_eq!(std_flat.len(), cached_last.len(), "vocab size mismatch");
        let max_diff = std_flat.iter().zip(cached_last.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "prefill logits max diff = {max_diff} (threshold 1e-4)");
    }

    #[test]
    fn test_kv_cache_decode_matches_full_forward() {
        let cfg = tiny_config();
        let mut model = build_model(&cfg);
        let device = Device::Cpu;
        let prompt = vec![1u32, 5, 10];
        let full_seq = vec![1u32, 5, 10, 20];
        let ref_logits = forward_standard(&mut model, &full_seq, &device);
        let ref_flat: Vec<f32> = ref_logits.flatten_all().unwrap().to_vec1().unwrap();
        let prompt_input = Tensor::from_vec(prompt.clone(), (prompt.len(),), &device).unwrap();
        let _prefill = model.forward_with_cache(crate::ops::select_ops(&device), &prompt_input, 0).unwrap();
        let decode_input = Tensor::from_vec(vec![20u32], (1,), &device).unwrap();
        let decode_logits = model.forward_with_cache(crate::ops::select_ops(&device), &decode_input, prompt.len()).unwrap();
        model.clear_kv_cache();
        let decode_flat: Vec<f32> = decode_logits.get(0).unwrap().to_vec1().unwrap();
        assert_eq!(ref_flat.len(), decode_flat.len(), "vocab size mismatch");
        let max_diff = ref_flat.iter().zip(decode_flat.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "decode logits max diff = {max_diff} (threshold 1e-4)");
    }

    #[test]
    fn test_kv_cache_clear_determinism() {
        let cfg = tiny_config();
        let mut model = build_model(&cfg);
        let device = Device::Cpu;
        let tokens = vec![1u32, 5, 10];
        let input = Tensor::from_vec(tokens, (3,), &device).unwrap();
        let logits1 = model.forward_with_cache(crate::ops::select_ops(&device), &input, 0).unwrap();
        model.clear_kv_cache();
        let logits2 = model.forward_with_cache(crate::ops::select_ops(&device), &input, 0).unwrap();
        model.clear_kv_cache();
        let v1: Vec<f32> = logits1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = logits2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1.len(), v2.len());
        let max_diff = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "cache clear determinism failed: max diff = {max_diff}");
    }

    #[test]
    fn test_kv_cache_multi_step_decode() {
        let cfg = tiny_config();
        let mut model = build_model(&cfg);
        let device = Device::Cpu;
        let prompt = vec![1u32, 5, 10];
        let prompt_input = Tensor::from_vec(prompt.clone(), (prompt.len(),), &device).unwrap();
        let prefill_logits = model.forward_with_cache(crate::ops::select_ops(&device), &prompt_input, 0).unwrap();
        let mut generated = vec![
            prefill_logits.get(prompt.len() - 1).unwrap().argmax(0).unwrap().to_vec0::<u32>().unwrap(),
        ];
        for step in 0..2 {
            let offset = prompt.len() + step;
            let tok = *generated.last().unwrap();
            let input = Tensor::from_vec(vec![tok], (1,), &device).unwrap();
            let logits = model.forward_with_cache(crate::ops::select_ops(&device), &input, offset).unwrap();
            generated.push(logits.get(0).unwrap().argmax(0).unwrap().to_vec0::<u32>().unwrap());
        }
        model.clear_kv_cache();
        let mut ref_generated = Vec::new();
        let mut full_seq = prompt.clone();
        for _step in 0..3 {
            let ref_logits = forward_standard(&mut model, &full_seq, &device);
            let next = ref_logits.flatten_all().unwrap().argmax(0).unwrap().to_vec0::<u32>().unwrap();
            ref_generated.push(next);
            full_seq.push(next);
        }
        assert_eq!(generated, ref_generated,
            "multi-step decode mismatch: cached={generated:?} vs reforward={ref_generated:?}");
    }
}

// ── ArchSpec (model registry metadata) ─────────────────────────────

mod meta {
    use crate::loading::var_builder::VarBuilder;
    use crate::models::config::Qwen3Config;

    use super::{
        Qwen3ClassifierConfig, Qwen3ForEmbedding, Qwen3ForSequenceClassification, Qwen3ModelForCausalLM,
    };
    use crate::engine::EngineError;
    use crate::engine::{CommonModelConfig, RuntimeCaps, TaskKind, WeightsBackend};
    use crate::models::registry::{
        ArchSpec, ParsedModelConfig, candle_model_err, inject_num_labels_if_missing, parse_json,
        parse_value,
    };

    const ARCHITECTURE_ALIASES: &[&str] = &["Qwen3", "Qwen3Model"];
    const MODEL_TYPE_ALIASES: &[&str] = &["qwen3"];
    const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate, TaskKind::Classify, TaskKind::Embed];

    enum Qwen3ArchConfig {
        Dense(Qwen3Config),
        Classifier(Qwen3ClassifierConfig),
        Embedding(Qwen3Config),
    }

    fn common_from_qwen3(cfg: &Qwen3Config) -> CommonModelConfig {
        CommonModelConfig {
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            max_position_embeddings: cfg.max_position_embeddings,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        }
    }

    pub(crate) struct Qwen3ArchSpec;
    pub(crate) static QWEN3_ARCH_SPEC: Qwen3ArchSpec = Qwen3ArchSpec;
    inventory::submit!(crate::models::registry::ArchSpecEntry::new(&QWEN3_ARCH_SPEC));

    impl ArchSpec for Qwen3ArchSpec {
        fn name(&self) -> &'static str { "qwen3" }
        fn architecture_aliases(&self) -> &'static [&'static str] { ARCHITECTURE_ALIASES }
        fn model_type_aliases(&self) -> &'static [&'static str] { MODEL_TYPE_ALIASES }
        fn supported_tasks(&self) -> &'static [TaskKind] { SUPPORTED_TASKS }

        fn parse_config(&self, task: TaskKind, raw: &serde_json::Value, content: &str) -> Result<ParsedModelConfig, EngineError> {
            match task {
                TaskKind::Generate => {
                    let cfg = parse_json::<Qwen3Config>(content, "Qwen3 config")?;
                    let common = common_from_qwen3(&cfg);
                    Ok(ParsedModelConfig { common, deltanet: None, arch_config: Box::new(Qwen3ArchConfig::Dense(cfg)) })
                }
                TaskKind::Classify => {
                    let json = inject_num_labels_if_missing(raw);
                    let cfg = parse_value::<Qwen3ClassifierConfig>(json, "Qwen3 classifier config")?;
                    let common = common_from_qwen3(&cfg.base);
                    Ok(ParsedModelConfig { common, deltanet: None, arch_config: Box::new(Qwen3ArchConfig::Classifier(cfg)) })
                }
                TaskKind::Embed => {
                    let cfg = parse_json::<Qwen3Config>(content, "Qwen3 embedding config")?;
                    let common = common_from_qwen3(&cfg);
                    Ok(ParsedModelConfig { common, deltanet: None, arch_config: Box::new(Qwen3ArchConfig::Embedding(cfg)) })
                }
            }
        }

        fn build_model(&self, arch_config: &dyn std::any::Any, vb: VarBuilder<'_>) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
            let cfg = arch_config.downcast_ref::<Qwen3ArchConfig>()
                .ok_or_else(|| EngineError::Internal("unexpected arch config type for Qwen3".into()))?;
            match cfg {
                Qwen3ArchConfig::Dense(c) => Ok(Box::new(Qwen3ModelForCausalLM::new(c, vb).map_err(candle_model_err)?)),
                Qwen3ArchConfig::Classifier(c) => Ok(Box::new(Qwen3ForSequenceClassification::new(c, vb).map_err(candle_model_err)?)),
                Qwen3ArchConfig::Embedding(c) => Ok(Box::new(Qwen3ForEmbedding::new(c, vb).map_err(candle_model_err)?)),
            }
        }

        fn runtime_caps(&self, task: TaskKind, backend: WeightsBackend, device: &crate::tensor::Device) -> RuntimeCaps {
            let is_safetensors = backend == WeightsBackend::Safetensors;
            let is_cuda = device.is_cuda();
            let supports_cuda_varlen = is_cuda && is_safetensors;
            RuntimeCaps {
                supports_kv_cache: is_safetensors && task == TaskKind::Generate,
                supports_prefix_cache: is_safetensors && is_cuda,
                supports_paged_attn: is_cuda && is_safetensors,
                supports_varlen: supports_cuda_varlen,
                supports_deltanet: false,
                supports_cuda_graph: supports_cuda_varlen && task == TaskKind::Generate,
            }
        }

        fn gguf_aliases(&self) -> &'static [&'static str] {
            &["qwen3"]
        }

        #[cfg(feature = "candle-baseline")]
        fn load_gguf(
            &self,
            ct: crate::tensor::quantized::gguf_file::Content,
            reader: &mut std::fs::File,
            device: &crate::tensor::Device,
        ) -> Result<crate::models::registry::GgufLoadResult, EngineError> {
            let (model, cfg) = super::gguf::Qwen3GgufModel::from_gguf(ct, reader, device)
                .map_err(candle_model_err)?;
            let common = CommonModelConfig {
                vocab_size: cfg.vocab_size,
                num_hidden_layers: cfg.num_hidden_layers,
                max_position_embeddings: cfg.max_position_embeddings,
                num_attention_heads: cfg.num_attention_heads,
                num_key_value_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
            };
            Ok(crate::models::registry::GgufLoadResult {
                model: Box::new(model),
                common,
                deltanet: None,
                eos_token_ids: cfg.eos_token_ids,
            })
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GGUF support (from qwen3/gguf.rs)
// ═══════════════════════════════════════════════════════════════════════

#[cfg(feature = "candle-baseline")]
pub mod gguf {
    //! Quantized Qwen3 model loaded from GGUF format.
    //!
    //! Thin wrapper around `candle_transformers::models::quantized_qwen3::ModelWeights`
    //! to integrate with the `ModelForward` trait dispatch system.
    
    use crate::tensor::quantized::gguf_file;
    use crate::tensor::{Device, Result, Tensor};
    use candle_transformers::models::quantized_qwen3::ModelWeights;
    use std::io::{Read, Seek};
    
    use crate::constants::{GGUF_DEFAULT_VOCAB_SIZE, GGUF_INTERMEDIATE_SIZE_MULTIPLIER};
    
    /// Configuration extracted from GGUF metadata.
    #[derive(Debug, Clone)]
    pub struct Qwen3GgufConfig {
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
    }
    
    /// Quantized Qwen3 model wrapper.
    pub struct Qwen3GgufModel {
        inner: ModelWeights,
    }
    
    impl Qwen3GgufModel {
        /// Load a quantized Qwen3 model from parsed GGUF content.
        pub fn from_gguf<R: Read + Seek>(
            ct: gguf_file::Content,
            reader: &mut R,
            device: &Device,
        ) -> Result<(Self, Qwen3GgufConfig)> {
            let config = parse_gguf_config(&ct)?;
            let inner = ModelWeights::from_gguf(ct, reader, device)?;
            Ok((Self { inner }, config))
        }
    
        pub fn clear_kv_cache(&mut self) {
            self.inner.clear_kv_cache();
        }
    
        /// Stub for `dispatch_model!` compatibility — GGUF doesn't support varlen.
        pub fn forward(
            &mut self,
            _packed_input: &Tensor,
            _ctx: &mut crate::models::commons::BatchAttnContext,
        ) -> Result<Tensor> {
            crate::tensor::bail!("GGUF model does not support varlen forward")
        }
    }
    
    impl crate::models::ModelForward for Qwen3GgufModel {
        fn forward(
            &mut self,
            _packed_input: &Tensor,
            _ctx: &mut crate::models::commons::BatchAttnContext,
        ) -> crate::tensor::Result<Tensor> {
            crate::tensor::bail!("GGUF model does not support varlen forward")
        }
    
        fn clear_kv_cache(&mut self) {
            self.clear_kv_cache();
        }
    }
    
    /// Extract model config from GGUF metadata keys.
    // Cleaned -- Reviewed by Minzhou
    fn parse_gguf_config(ct: &gguf_file::Content) -> Result<Qwen3GgufConfig> {
        let md = &ct.metadata;
    
        let get_u32 = |key: &str| -> Result<usize> {
            md.get(key)
                .ok_or_else(|| crate::tensor::Error::Msg(format!("missing GGUF metadata: {key}")))?
                .to_u32()
                .map(|v| v as usize)
        };
    
        let get_f32 = |key: &str| -> Result<f64> {
            md.get(key)
                .ok_or_else(|| crate::tensor::Error::Msg(format!("missing GGUF metadata: {key}")))?
                .to_f32()
                .map(|v| v as f64)
        };
    
        // Detect architecture prefix (usually "qwen3")
        let default_arch = "qwen3".to_string();
        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .unwrap_or_else(|| {
                tracing::warn!("Qwen3 GGUF: 'general.architecture' not found, using default: qwen3");
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
        let intermediate_size = md
            .get(&format!("{arch}.feed_forward_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                tracing::warn!("Qwen3 GGUF: '{arch}.feed_forward_length' not found, using default: {default_intermediate} (= hidden_size * {GGUF_INTERMEDIATE_SIZE_MULTIPLIER})");
                default_intermediate
            });
    
        let vocab_size = ct
            .tensor_infos
            .get("token_embd.weight")
            .map(|t| t.shape.dims()[0])
            .unwrap_or_else(|| {
                tracing::warn!("Qwen3 GGUF: 'token_embd.weight' tensor not found, using default vocab_size: {GGUF_DEFAULT_VOCAB_SIZE}");
                GGUF_DEFAULT_VOCAB_SIZE
            });
        let eos_token_ids = md
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .map(|id| vec![id])
            .unwrap_or_else(|| {
                tracing::warn!("Qwen3 GGUF: 'tokenizer.ggml.eos_token_id' not found, using empty list");
                vec![]
            });
    
        Ok(Qwen3GgufConfig {
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
        })
    }
}
