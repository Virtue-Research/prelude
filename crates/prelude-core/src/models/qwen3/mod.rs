#[cfg(feature = "candle-baseline")]
pub mod gguf;

use std::collections::HashMap;
use std::sync::Arc;

use crate::tensor::{DType, Module, Result, Tensor};
use crate::loading::var_builder::VarBuilder;
use crate::nn_ops::{Embedding, Qwen3Config};
use serde::Deserialize;

use crate::modules::varlen_attention;
use crate::modules::{BatchAttnContext, LayerAttnContext};
use crate::modules::{
    GatedMlp, Linear, RmsNorm, RotaryEmbedding, TransformerBlock,
    fast_add, fast_rms_norm, fused_add_rmsnorm, last_token_select, qknorm_rope_varlen,
};
use crate::modules::debug_disable_fused_qknorm_rope;
use crate::profiling::{nvtx_push, nvtx_pop};

// Re-export public debug setters so existing callers (`use qwen3::set_debug_*`) still compile.
pub use crate::modules::{
    set_debug_disable_fast_rmsnorm, set_debug_disable_flash_attn_path,
    set_debug_disable_fused_add_rmsnorm, set_debug_disable_fused_qknorm_rope,
    set_debug_disable_fused_silu_mul, set_debug_disable_vectorized_add,
};

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
            let qw = q_proj.weight();
            if qw.device().is_cpu() && qw.dtype() == DType::BF16 {
                let merged_w =
                    Tensor::cat(&[qw, k_proj.weight(), v_proj.weight()], 0)?;
                match Linear::from_weight(merged_w, None) {
                    Ok(l) => Some(l),
                    Err(e) => {
                        tracing::warn!("Failed to create fused qkv_proj: {e}");
                        None
                    }
                }
            } else {
                None
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
    fn fused_qkv_projection(&self, x: &Tensor, total: usize) -> Result<(Tensor, Tensor, Tensor)> {
        crate::modules::attn_utils::fused_qkv_projection(
            x,
            &self.q_proj, &self.k_proj, &self.v_proj,
            self.qkv_proj.as_ref(),
            total, self.num_heads, self.num_kv_heads, self.head_dim,
        )
    }

    /// QK-norm + RoPE for varlen layout `[total, H, D]`.
    fn norm_rope_varlen(
        &self,
        ops: &crate::ops::Ops,
        q: &Tensor,
        k: &Tensor,
        total: usize,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        qknorm_rope_varlen(
            ops, q, k,
            &self.q_norm_weight, &self.k_norm_weight,
            &self.q_norm, &self.k_norm,
            &self.rotary_emb, position_ids,
            total, self.num_heads, self.num_kv_heads, self.head_dim,
            self.rms_norm_eps,
        )
    }

    // ── Varlen forward (unified: GPU flash-attn / CPU cpu_ops) ──────────

    pub(crate) fn forward(&self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
        {
            let total_q = x.dim(0)?;

            // Raw CPU fast paths (brgemm, raw_f32) are in prelude-cpu.
            // The generic path below handles both GPU and CPU via Ops traits.

            let (q, k, v) = self.fused_qkv_projection(x, total_q)?;

            // Fused path: try fused Q+K norm+rope via Ops trait
            if !debug_disable_fused_qknorm_rope() {
                if let Some(result) = ctx.ops.fused.fused_qknorm_rope(
                    &q, &k, &self.q_norm_weight, &self.k_norm_weight,
                    &self.rotary_emb.cos, &self.rotary_emb.sin,
                    ctx.position_ids, self.rms_norm_eps as f32,
                ) {
                    let (q, k) = result?;
                    if let Some(kv) = ctx.paged_kv {
                        // Try fused K-norm + RoPE + KV cache write
                        let used_fused_kv_write = if let Some(fused_result) =
                            ctx.ops.fused.fused_knorm_rope_cache_write(
                                &k, &v, &self.k_norm_weight,
                                &self.rotary_emb.cos, &self.rotary_emb.sin,
                                ctx.position_ids,
                                kv.key_cache, kv.value_cache, kv.slot_mapping,
                                self.rms_norm_eps as f32,
                            )
                        {
                            fused_result?;
                            true
                        } else {
                            false
                        };
                        if !used_fused_kv_write {
                            crate::modules::reshape_and_cache(
                                ctx.ops, &k, &v,
                                kv.key_cache, kv.value_cache, kv.slot_mapping,
                            )?;
                        }
                        let attn = crate::modules::varlen_attention_paged(
                            ctx.ops, &q,
                            kv.key_cache, kv.value_cache, kv.block_tables,
                            ctx.cu_seqlens_q, kv.cu_seqlens_k,
                            ctx.max_seqlen_q, kv.max_seqlen_k,
                            self.softmax_scale,
                        )?;
                        return attn
                            .reshape((total_q, self.hidden_size))?
                            .apply(&self.o_proj);
                    }
                    return varlen_attention(
                        ctx.ops, &q, &k, &v,
                        ctx.cu_seqlens_q, ctx.cu_seqlens_q,
                        ctx.max_seqlen_q, ctx.max_seqlen_q,
                        self.softmax_scale, None,
                    )?
                    .reshape((total_q, self.hidden_size))?
                    .apply(&self.o_proj);
                }
            }

            // Non-fused path: norm + rope then varlen attention (GPU flash-attn or CPU matmul)
            let (q, k) = self.norm_rope_varlen(ctx.ops, &q, &k, total_q, ctx.position_ids)?;

            let (cu_seqlens_k, max_seqlen_k) = match ctx.paged_kv {
                Some(kv) => (kv.cu_seqlens_k, kv.max_seqlen_k),
                None => (ctx.cu_seqlens_q, ctx.max_seqlen_q),
            };
            let attn_out = varlen_attention(
                ctx.ops,
                &q, &k, &v,
                ctx.cu_seqlens_q, cu_seqlens_k,
                ctx.max_seqlen_q, max_seqlen_k,
                self.softmax_scale,
                ctx.paged_kv,
            )?;

            attn_out
                .reshape((total_q, self.hidden_size))?
                .apply(&self.o_proj)
        }
    }

    /// Cached forward for CPU decode: handles both prefill (L=prompt_len) and
    /// decode (L=1). KV cache accumulates across calls; call `reset_kv_cache`
    /// between requests.
    ///
    /// KV cache is stored in varlen format `[total, H_kv, D]`.
    /// Uses matmul-based SDPA as a universal fallback.
    fn forward_with_cache(&mut self, x: &Tensor, position_offset: usize) -> Result<Tensor> {
        let seq_len = x.dim(0)?;

        // ── Tensor-based path ──
        let (q, k, v) = self.fused_qkv_projection(x, seq_len)?;
        let position_ids: Vec<u32> =
            (0..seq_len).map(|i| (position_offset + i) as u32).collect();
        let position_ids = Tensor::from_vec(position_ids, (seq_len,), x.device())?;
        let ops = crate::ops::select_ops(x.device());
        let (q, k) = self.norm_rope_varlen(ops, &q, &k, seq_len, &position_ids)?;

        self.k_cache.append(&k)?;
        self.v_cache.append(&v)?;
        let k_full = self.k_cache.view()?;
        let v_full = self.v_cache.view()?;

        let attn_out = self.matmul_cross_attention(
            &q, &k_full, &v_full, seq_len, position_offset,
        )?;

        attn_out
            .reshape((seq_len, self.hidden_size))?
            .apply(&self.o_proj)
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
        let attn_weights = crate::nn_ops::ops::softmax_last_dim(&scores)?;

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
// Decoder Layer
// ============================================================================

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: GatedMlp,
    block: TransformerBlock,
}

impl DecoderLayer {
    fn new(
        cfg: &Qwen3Config,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        let self_attn = Qwen3Attention::new(
            cfg,
            rotary,
            vb.pp("self_attn"),
        )?;
        let mlp = GatedMlp::new(cfg, vb.pp("mlp"))?;
        let ln1 = RmsNorm::load(vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let ln1_weight = ln1.weight().clone();
        let ln2 = RmsNorm::load(vb.pp("post_attention_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let ln2_weight = ln2.weight().clone();
        Ok(Self {
            self_attn,
            mlp,
            block: TransformerBlock::new(ln1, ln1_weight, ln2, ln2_weight, cfg.rms_norm_eps, layer_idx),
        })
    }

    #[inline]
    fn residual_mlp(&self, ops: &crate::ops::Ops, x_res: &Tensor, h2: &Tensor) -> Result<Tensor> {
        fast_add(ops, x_res, &self.mlp.forward(ops, h2)?)
    }

    fn forward_with_cache(&mut self, ops: &crate::ops::Ops, x: &Tensor, position_offset: usize) -> Result<Tensor> {
        let h = fast_rms_norm(ops, x, &self.block.ln1, &self.block.ln1_weight, self.block.rms_norm_eps)?;
        let h = self.self_attn.forward_with_cache(&h, position_offset)?;
        let (x_res, h2) = fused_add_rmsnorm(ops, x, &h, &self.block.ln2, &self.block.ln2_weight, self.block.rms_norm_eps)?;
        self.residual_mlp(ops, &x_res, &h2)
    }

    fn reset_kv_cache(&mut self) {
        self.self_attn.reset_kv_cache();
    }

    fn forward(&self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
        self.block.forward(ctx.ops, x,
            |h| self.self_attn.forward(h, ctx),
            |x_res, h2| self.residual_mlp(ctx.ops, x_res, h2),
        )
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
    norm_weight: Tensor,
    rms_norm_eps: f64,
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
        let norm_weight = norm.weight().clone();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            norm_weight,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.reset_kv_cache();
        }
    }

    fn forward_with_cache(
        &mut self,
        ops: &crate::ops::Ops,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            nvtx_push!("layer[{}]", i);
            h = layer.forward_with_cache(ops, &h, position_offset)?;
            nvtx_pop!();
        }
        fast_rms_norm(ops, &h, &self.norm, &self.norm_weight, self.rms_norm_eps)
    }

    fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let mut h = self.embed_tokens.forward(packed_input)?;
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
            h = layer.forward(&h, &layer_ctx)?;
            nvtx_pop!();
        }
        fast_rms_norm(ctx.ops, &h, &self.norm, &self.norm_weight, self.rms_norm_eps)
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
        last_token_select(hidden, seq_lens)?
            .unsqueeze(1)?
            .apply(&self.lm_head)
    }

    pub fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let hidden = self.base.forward(packed_input, ctx)?;
        self.last_token_logits(&hidden, ctx.seq_lens)
    }

    /// Cached forward: returns logits `[L, vocab_size]` for all input tokens.
    pub fn forward_with_cache(
        &mut self,
        ops: &crate::ops::Ops,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
        let hidden = self.base.forward_with_cache(ops, input_ids, position_offset)?;
        hidden.apply(&self.lm_head)
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
        last_token_select(&hidden_states, ctx.seq_lens)?.apply(&self.score)
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
        hidden.apply(&self.lm_head)
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
    use crate::modules::BatchAttnContext;

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
    use crate::nn_ops::Qwen3Config;

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
            let supports_cuda_varlen = (cfg!(feature = "cuda") || cfg!(feature = "flash-attn-v4") || cfg!(feature = "flashinfer"))
                && device.is_cuda() && is_safetensors;
            RuntimeCaps {
                supports_kv_cache: is_safetensors && task == TaskKind::Generate,
                supports_prefix_cache: is_safetensors && cfg!(feature = "cuda") && device.is_cuda(),
                supports_paged_attn: cfg!(feature = "cuda") && device.is_cuda() && is_safetensors,
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
