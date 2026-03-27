#[cfg(feature = "candle-baseline")]
pub mod gguf;
pub(crate) mod meta;

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Module, Result, Tensor};
use crate::loading::var_builder::VarBuilder;
use crate::nn_ops::{Embedding, Qwen3Config};
use serde::Deserialize;

use crate::models::common::varlen_attention;
use crate::models::common::{BatchAttnContext, LayerAttnContext};
use crate::models::common::{
    GatedMlp, Linear, RmsNorm, RotaryEmbedding, TransformerBlock,
    fast_add, fast_rms_norm, fused_add_rmsnorm, last_token_select, qknorm_rope_varlen,
};
#[cfg(feature = "cuda")]
use crate::models::common::debug_disable_fused_qknorm_rope;
use crate::profiling::{nvtx_push, nvtx_pop};

// Re-export public debug setters so existing callers (`use qwen3::set_debug_*`) still compile.
pub use crate::models::common::{
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
            candle_core::bail!("sliding window is not supported yet");
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
        crate::models::common::attention::fused_qkv_projection(
            x,
            &self.q_proj, &self.k_proj, &self.v_proj,
            self.qkv_proj.as_ref(),
            total, self.num_heads, self.num_kv_heads, self.head_dim,
        )
    }

    /// QK-norm + RoPE for varlen layout `[total, H, D]`.
    fn norm_rope_varlen(
        &self,
        q: &Tensor,
        k: &Tensor,
        total: usize,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        qknorm_rope_varlen(
            q, k,
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

            // ── Raw CPU BF16 fast path: bypass Tensor intermediates ──────
            if !self.is_cuda && x.dtype() == candle_core::DType::BF16 && ctx.paged_kv.is_none() {
                if let Some(ref qkv_proj) = self.qkv_proj {
                    if let (Some(qkv_brg), Some(oproj_brg)) =
                        (qkv_proj.brgemm_weight(), self.o_proj.brgemm_weight())
                    {
                        if let Some(ref cos_sin_cache) = self.rotary_emb.cos_sin_cache {
                            let result = self.forward_raw_cpu_bf16(
                                x, ctx,
                                qkv_brg, oproj_brg, cos_sin_cache,
                            )?;
                            return Ok(result);
                        }
                    }
                }
            }

            // ── Raw CPU F32 fast path: bypass Tensor intermediates ──────
            if !self.is_cuda && x.dtype() == candle_core::DType::F32 && ctx.paged_kv.is_none() {
                if let Some(ref qkv_proj) = self.qkv_proj {
                    if let (Some(qkv_pw), Some(oproj_pw)) =
                        (qkv_proj.f32_packed_weight(), self.o_proj.f32_packed_weight())
                    {
                        if let Some(ref cos_sin_cache) = self.rotary_emb.cos_sin_cache {
                            let result = self.forward_raw_cpu_f32(
                                x, ctx,
                                qkv_pw, oproj_pw, cos_sin_cache,
                            )?;
                            return Ok(result);
                        }
                    }
                }
            }

            let (q, k, v) = self.fused_qkv_projection(x, total_q)?;

            // Fused CUDA path: norm + rope + optional fused KV cache write
            #[cfg(feature = "cuda")]
            if self.is_cuda && !debug_disable_fused_qknorm_rope() {
                let q = crate::ops::gpu::fused_qknorm_rope_varlen(
                    &q,
                    &self.q_norm_weight,
                    &self.rotary_emb.cos,
                    &self.rotary_emb.sin,
                    ctx.position_ids,
                    self.rms_norm_eps,
                )?;
                if let Some(kv) = ctx.paged_kv {
                    #[cfg(any(feature = "flash-attn-v3", feature = "flashinfer"))]
                    let used_fused_kv_write = if crate::ops::gpu::fused_kv_cache_write_enabled() {
                        let bs = kv.key_cache.shape().dims()[1];
                        crate::ops::gpu::fused_knorm_rope_kv_cache_write_varlen(
                            &k,
                            &v,
                            &self.k_norm_weight,
                            &self.rotary_emb.cos,
                            &self.rotary_emb.sin,
                            ctx.position_ids,
                            kv.key_cache,
                            kv.value_cache,
                            kv.slot_mapping,
                            self.num_kv_heads,
                            self.head_dim,
                            bs,
                            self.rms_norm_eps,
                        )?;
                        true
                    } else {
                        false
                    };
                    #[cfg(not(any(feature = "flash-attn-v3", feature = "flashinfer")))]
                    let used_fused_kv_write = false;

                    if !used_fused_kv_write {
                        let k = crate::ops::gpu::fused_qknorm_rope_varlen(
                            &k,
                            &self.k_norm_weight,
                            &self.rotary_emb.cos,
                            &self.rotary_emb.sin,
                            ctx.position_ids,
                            self.rms_norm_eps,
                        )?;
                        crate::models::common::reshape_and_cache(
                            &k,
                            &v,
                            kv.key_cache,
                            kv.value_cache,
                            kv.slot_mapping,
                        )?;
                    }
                    let attn = crate::models::common::varlen_attention_paged(
                        &q,
                        kv.key_cache,
                        kv.value_cache,
                        kv.block_tables,
                        ctx.cu_seqlens_q,
                        kv.cu_seqlens_k,
                        ctx.max_seqlen_q,
                        kv.max_seqlen_k,
                        self.softmax_scale,
                    )?;
                    return attn
                        .reshape((total_q, self.hidden_size))?
                        .apply(&self.o_proj);
                }
                let k = crate::ops::gpu::fused_qknorm_rope_varlen(
                    &k,
                    &self.k_norm_weight,
                    &self.rotary_emb.cos,
                    &self.rotary_emb.sin,
                    ctx.position_ids,
                    self.rms_norm_eps,
                )?;
                return varlen_attention(
                    &q,
                    &k,
                    &v,
                    ctx.cu_seqlens_q,
                    ctx.cu_seqlens_q,
                    ctx.max_seqlen_q,
                    ctx.max_seqlen_q,
                    self.softmax_scale,
                    None,
                )?
                .reshape((total_q, self.hidden_size))?
                .apply(&self.o_proj);
            }

            // Non-fused path: norm + rope then varlen attention (GPU flash-attn or CPU matmul)
            let (q, k) = self.norm_rope_varlen(&q, &k, total_q, ctx.position_ids)?;

            let (cu_seqlens_k, max_seqlen_k) = match ctx.paged_kv {
                Some(kv) => (kv.cu_seqlens_k, kv.max_seqlen_k),
                None => (ctx.cu_seqlens_q, ctx.max_seqlen_q),
            };
            let attn_out = varlen_attention(
                &q,
                &k,
                &v,
                ctx.cu_seqlens_q,
                cu_seqlens_k,
                ctx.max_seqlen_q,
                max_seqlen_k,
                self.softmax_scale,
                ctx.paged_kv,
            )?;

            attn_out
                .reshape((total_q, self.hidden_size))?
                .apply(&self.o_proj)
        }
    }

    /// Raw CPU BF16 attention forward: all intermediate operations use raw &[u16]
    /// slices instead of Tensor, eliminating ~8 Tensor alloc/drop per layer.
    ///
    /// Flow: input Tensor → CpuTensor → QKV GEMM → split → QK norm → RoPE
    ///       → attention → O_proj GEMM → CpuTensor → output Tensor
    fn forward_raw_cpu_bf16(
        &self,
        x: &Tensor,
        ctx: &LayerAttnContext,
        qkv_brg: &crate::ops::onednn::BrgemmPackedWeight,
        oproj_brg: &crate::ops::onednn::BrgemmPackedWeight,
        cos_sin_cache: &Tensor,
    ) -> Result<Tensor> {
        use crate::models::common::raw_cpu;
        use crate::ops::cpu::buf_tensor::CpuTensor;

        let device = x.device();
        let x_buf = CpuTensor::from_candle(x)?;
        let q_norm_w = CpuTensor::from_candle(&self.q_norm_weight)?;
        let k_norm_w = CpuTensor::from_candle(&self.k_norm_weight)?;
        let cache_buf = CpuTensor::from_candle(cos_sin_cache)?;
        let positions = raw_cpu::extract_positions(ctx.position_ids)?;
        let seq_lens = raw_cpu::extract_seq_lens(ctx.cu_seqlens_q)?;

        // Full raw attention pipeline via shared infrastructure
        raw_cpu::with_scratch(|scratch| {
            let out = unsafe {
                raw_cpu::raw_attention_forward(
                    scratch,
                    &x_buf,
                    qkv_brg,
                    oproj_brg,
                    &q_norm_w,
                    &k_norm_w,
                    &cache_buf,
                    &positions,
                    &seq_lens,
                    self.num_heads,
                    self.num_kv_heads,
                    self.rms_norm_eps as f32,
                    self.softmax_scale,
                )
            };
            out.to_candle(device)
        })
    }

    /// Raw CPU F32 attention forward: mirrors forward_raw_cpu_bf16 but for F32.
    /// All intermediate operations use raw &[f32] slices via OnednnF32PackedWeight.
    fn forward_raw_cpu_f32(
        &self,
        x: &Tensor,
        ctx: &LayerAttnContext,
        qkv_pw: &crate::ops::onednn::OnednnF32PackedWeight,
        oproj_pw: &crate::ops::onednn::OnednnF32PackedWeight,
        cos_sin_cache: &Tensor,
    ) -> Result<Tensor> {
        use crate::models::common::raw_cpu;
        use crate::ops::cpu::buf_tensor::CpuTensorF32;

        let device = x.device();
        let x_buf = CpuTensorF32::from_candle(x)?;
        let q_norm_w = crate::ops::cpu::tensor_as_f32_slice(&self.q_norm_weight)?;
        let k_norm_w = crate::ops::cpu::tensor_as_f32_slice(&self.k_norm_weight)?;
        let cache_slice = crate::ops::cpu::tensor_as_f32_slice(cos_sin_cache)?;
        let rotary_dim = cos_sin_cache.dims()[1];
        let positions = raw_cpu::extract_positions(ctx.position_ids)?;
        let seq_lens = raw_cpu::extract_seq_lens(ctx.cu_seqlens_q)?;

        raw_cpu::with_scratch_f32(|scratch| {
            let out = unsafe {
                raw_cpu::raw_attention_forward_f32(
                    scratch,
                    &x_buf,
                    qkv_pw,
                    oproj_pw,
                    q_norm_w,
                    k_norm_w,
                    cache_slice,
                    rotary_dim,
                    &positions,
                    &seq_lens,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.rms_norm_eps as f32,
                    self.softmax_scale,
                )
            };
            raw_cpu::wrap_output_f32(out.as_slice(), out.dims(), device)
        })
    }

    /// Cached forward for CPU decode: handles both prefill (L=prompt_len) and
    /// decode (L=1). KV cache accumulates across calls; call `reset_kv_cache`
    /// between requests.
    ///
    /// KV cache is stored in varlen format `[total, H_kv, D]`.
    /// - Prefill (offset=0, BF16+oneDNN): raw BF16 path (zero Tensor allocs) + cache K,V
    /// - Prefill (offset=0, BF16 no-oneDNN): Tensor path + cpu_prefill_attention
    /// - Decode (seq_len=1, BF16): decode_attention_bf16 kernel
    /// - F32: matmul SDPA fallback
    fn forward_with_cache(&mut self, x: &Tensor, position_offset: usize) -> Result<Tensor> {
        let seq_len = x.dim(0)?;

        // ── BF16 + oneDNN raw prefill: fused QKV+norm+RoPE+attn+O_proj, save K,V ──
        if position_offset == 0 && !self.is_cuda && x.dtype() == DType::BF16 {
            // Clone Arc refs to avoid borrowing self immutably while calling &mut self
            let qkv_proj = self.qkv_proj.clone();
            let o_proj_ref = self.o_proj.brgemm_weight().map(|w| w as *const _);
            let cos_sin = self.rotary_emb.cos_sin_cache.clone();
            if let Some(ref qkv) = qkv_proj {
                if let (Some(qkv_brg), Some(oproj_ptr)) =
                    (qkv.brgemm_weight(), o_proj_ref)
                {
                    if let Some(ref cos_sin_cache) = cos_sin {
                        // Safety: oproj_brg points into self.o_proj which lives for &mut self
                        let oproj_brg = unsafe { &*oproj_ptr };
                        return self.forward_raw_prefill_and_cache(
                            x, seq_len, qkv_brg, oproj_brg, cos_sin_cache,
                        );
                    }
                }
            }
        }

        // ── Tensor-based path (BF16 without oneDNN, F32, or decode) ──
        let (q, k, v) = self.fused_qkv_projection(x, seq_len)?;
        let position_ids: Vec<u32> =
            (0..seq_len).map(|i| (position_offset + i) as u32).collect();
        let position_ids = Tensor::from_vec(position_ids, (seq_len,), x.device())?;
        let (q, k) = self.norm_rope_varlen(&q, &k, seq_len, &position_ids)?;

        self.k_cache.append(&k)?;
        self.v_cache.append(&v)?;
        let k_full = self.k_cache.view()?;
        let v_full = self.v_cache.view()?;
        let total_kv_len = k_full.dim(0)?;

        let attn_out = if x.dtype() == DType::BF16 {
            if position_offset == 0 {
                crate::ops::cpu::cpu_prefill_attention(
                    &q, &k, &v,
                    &[seq_len], self.num_heads, self.num_kv_heads,
                    self.head_dim, self.softmax_scale as f64,
                )?
            } else {
                crate::ops::cpu::cpu_decode_attention(
                    &q, &k_full, &v_full,
                    total_kv_len, self.num_heads, self.num_kv_heads,
                    self.head_dim, self.softmax_scale,
                )?
            }
        } else if position_offset > 0 {
            crate::ops::cpu::cpu_decode_attention_f32(
                &q, &k_full, &v_full,
                total_kv_len, self.num_heads, self.num_kv_heads,
                self.head_dim, self.softmax_scale,
            )?
        } else {
            self.matmul_cross_attention(
                &q, &k_full, &v_full, seq_len, position_offset,
            )?
        };

        attn_out
            .reshape((seq_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    /// Raw BF16 prefill: runs the fully fused raw path (zero Tensor allocs)
    /// and saves K,V to KvBuf as a side effect.
    fn forward_raw_prefill_and_cache(
        &mut self,
        x: &Tensor,
        seq_len: usize,
        qkv_brg: &crate::ops::onednn::BrgemmPackedWeight,
        oproj_brg: &crate::ops::onednn::BrgemmPackedWeight,
        cos_sin_cache: &Tensor,
    ) -> Result<Tensor> {
        use crate::models::common::raw_cpu;
        use crate::ops::cpu::buf_tensor::CpuTensor;

        let device = x.device();
        let kv_size = self.num_kv_heads * self.head_dim;
        let x_buf = CpuTensor::from_candle(x)?;
        let q_norm_w = CpuTensor::from_candle(&self.q_norm_weight)?;
        let k_norm_w = CpuTensor::from_candle(&self.k_norm_weight)?;
        let cache_buf = CpuTensor::from_candle(cos_sin_cache)?;
        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let seq_lens = vec![seq_len];

        raw_cpu::with_scratch(|scratch| {
            // raw_attention_forward borrows scratch mutably; the returned CpuTensor
            // borrows scratch.proj_out. Convert to owned Tensor first, then read K,V.
            let out = unsafe {
                raw_cpu::raw_attention_forward(
                    scratch, &x_buf, qkv_brg, oproj_brg,
                    &q_norm_w, &k_norm_w, &cache_buf,
                    &positions, &seq_lens,
                    self.num_heads, self.num_kv_heads,
                    self.rms_norm_eps as f32, self.softmax_scale,
                )
            };
            let attn_result = out.to_candle(device)?;

            // Save K,V from scratch to KvBuf (small memcpy: seq_len * kv_size * 2 bytes)
            let k_tensor = crate::ops::cpu::u16_vec_to_bf16_tensor(
                scratch.k_normed[..seq_len * kv_size].to_vec(),
                &[seq_len, self.num_kv_heads, self.head_dim],
                device,
            )?;
            let v_tensor = crate::ops::cpu::u16_vec_to_bf16_tensor(
                scratch.v[..seq_len * kv_size].to_vec(),
                &[seq_len, self.num_kv_heads, self.head_dim],
                device,
            )?;
            self.k_cache.append(&k_tensor)?;
            self.v_cache.append(&v_tensor)?;

            Ok(attn_result)
        })
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
        let causal = seq_len > 1;
        let q_f32 = q.to_dtype(DType::F32)?;
        let k_f32 = k_full.to_dtype(DType::F32)?;
        let v_f32 = v_full.to_dtype(DType::F32)?;

        {
            let out = crate::models::common::attn::cpu::cross_attention_f32_onednn(
                &q_f32, &k_f32, &v_f32,
                seq_len, total_kv_len,
                self.num_heads, self.num_kv_heads, self.head_dim,
                self.softmax_scale, causal, position_offset,
            )?;
            return out.to_dtype(v_full.dtype());
        }

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
    fn residual_mlp(&self, x_res: &Tensor, h2: &Tensor) -> Result<Tensor> {
        if h2.device().is_cpu() && h2.dtype() == candle_core::DType::BF16 {
            if self.mlp.gate_up_brgemm_weight().is_some() {
                return self.residual_mlp_raw(x_res, h2);
            }
        }
        if h2.device().is_cpu() && h2.dtype() == candle_core::DType::F32 {
            if self.mlp.gate_up_f32_packed_weight().is_some() {
                return self.residual_mlp_raw_f32(x_res, h2);
            }
        }
        fast_add(x_res, &self.mlp.forward(h2)?)
    }

    fn residual_mlp_raw(&self, x_res: &Tensor, h2: &Tensor) -> Result<Tensor> {
        use crate::models::common::raw_cpu;
        use crate::ops::cpu::buf_tensor::CpuTensor;
        let h2_buf = CpuTensor::from_candle(h2)?;
        let needed = h2_buf.len();
        raw_cpu::with_scratch(|scratch| {
            raw_cpu::ensure_len(&mut scratch.mlp_out, needed);
            let mlp_out_ptr = scratch.mlp_out.as_mut_ptr();
            unsafe { self.mlp.forward_raw(scratch, &h2_buf, mlp_out_ptr) }
            crate::ops::cpu::inplace_add_bf16(x_res, &scratch.mlp_out[..needed]).unwrap();
        });
        Ok(x_res.clone())
    }

    fn residual_mlp_raw_f32(&self, x_res: &Tensor, h2: &Tensor) -> Result<Tensor> {
        use crate::models::common::raw_cpu;
        let h2_slice = crate::ops::cpu::tensor_as_f32_slice(h2)?;
        let (total, hidden_size) = (h2.dim(0)?, h2.dim(1)?);
        let needed = total * hidden_size;
        raw_cpu::with_scratch_f32(|scratch| {
            raw_cpu::ensure_len_f32(&mut scratch.mlp_out, needed);
            let mlp_out_ptr = scratch.mlp_out.as_mut_ptr();
            unsafe { self.mlp.forward_raw_f32(scratch, h2_slice.as_ptr(), total, hidden_size, mlp_out_ptr) }
            crate::ops::cpu::inplace_add_f32(x_res, &scratch.mlp_out[..needed]).unwrap();
        });
        Ok(x_res.clone())
    }

    fn forward_with_cache(&mut self, x: &Tensor, position_offset: usize) -> Result<Tensor> {
        let h = fast_rms_norm(x, &self.block.ln1, &self.block.ln1_weight, self.block.rms_norm_eps)?;
        let h = self.self_attn.forward_with_cache(&h, position_offset)?;
        let (x_res, h2) = fused_add_rmsnorm(x, &h, &self.block.ln2, &self.block.ln2_weight, self.block.rms_norm_eps)?;
        self.residual_mlp(&x_res, &h2)
    }

    fn reset_kv_cache(&mut self) {
        self.self_attn.reset_kv_cache();
    }

    fn forward(&self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
        self.block.forward(x,
            |h| self.self_attn.forward(h, ctx),
            |x_res, h2| self.residual_mlp(x_res, h2),
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
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            nvtx_push!("layer[{}]", i);
            h = layer.forward_with_cache(&h, position_offset)?;
            nvtx_pop!();
        }
        fast_rms_norm(&h, &self.norm, &self.norm_weight, self.rms_norm_eps)
    }

    fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let mut h = self.embed_tokens.forward(packed_input)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            nvtx_push!("layer[{}]", i);
            let layer_kv = ctx.paged_kv.map(|kv| kv.layer(i));
            let layer_ctx = LayerAttnContext {
                cu_seqlens_q: ctx.cu_seqlens_q,
                max_seqlen_q: ctx.max_seqlen_q,
                position_ids: ctx.position_ids,
                paged_kv: layer_kv.as_ref(),
            };
            h = layer.forward(&h, &layer_ctx)?;
            nvtx_pop!();
        }
        fast_rms_norm(&h, &self.norm, &self.norm_weight, self.rms_norm_eps)
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
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
        let hidden = self.base.forward_with_cache(input_ids, position_offset)?;
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
    ) -> candle_core::Result<Tensor> {
        self.base.forward(packed_input, ctx)
    }

    fn compute_logits(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        hidden.apply(&self.lm_head)
    }
}

impl KvCacheModel for Qwen3ModelForCausalLM {
    fn forward_with_cache(
        &mut self,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> candle_core::Result<Tensor> {
        Qwen3ModelForCausalLM::forward_with_cache(self, input_ids, position_offset)
    }
}

impl ModelForward for Qwen3ModelForCausalLM {
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> candle_core::Result<Tensor> {
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
    ) -> candle_core::Result<Tensor> {
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
    ) -> candle_core::Result<Tensor> {
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
    use candle_core::Device;
    use crate::loading::var_builder::VarBuilder;
    use crate::models::common::BatchAttnContext;

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

    /// Helper: run standard varlen forward on a single sequence.
    fn forward_standard(
        model: &mut Qwen3ModelForCausalLM,
        tokens: &[u32],
        device: &Device,
    ) -> Tensor {
        let seq_len = tokens.len();
        let input = Tensor::from_vec(tokens.to_vec(), (seq_len,), device).unwrap();
        let cu = Tensor::from_vec(vec![0u32, seq_len as u32], (2,), device).unwrap();
        let pos = Tensor::from_vec(
            (0..seq_len as u32).collect::<Vec<_>>(),
            (seq_len,),
            device,
        )
        .unwrap();
        let seq_lens = vec![seq_len];
        let mut ctx = BatchAttnContext {
            cu_seqlens_q: &cu,
            max_seqlen_q: seq_len,
            position_ids: &pos,
            seq_lens: &seq_lens,
            paged_kv: None,
            deltanet_pool: None,
            deltanet_slots: None,
        };
        let logits = model.forward(&input, &mut ctx).unwrap();
        model.clear_kv_cache();
        logits
    }

    /// Prefill via `forward_with_cache` should produce the same last-token logits
    /// as the standard varlen forward path.
    #[test]
    fn test_kv_cache_prefill_matches_standard() {
        let cfg = tiny_config();
        let mut model = build_model(&cfg);
        let device = Device::Cpu;
        let tokens = vec![1u32, 5, 10, 20, 3];
        let seq_len = tokens.len();

        // Standard forward → [1, 1, vocab]
        let std_logits = forward_standard(&mut model, &tokens, &device);

        // Cached forward → [seq_len, vocab]
        let input = Tensor::from_vec(tokens, (seq_len,), &device).unwrap();
        let cached_logits = model.forward_with_cache(&input, 0).unwrap();
        model.clear_kv_cache();

        let std_flat: Vec<f32> = std_logits.flatten_all().unwrap().to_vec1().unwrap();
        let cached_last: Vec<f32> = cached_logits
            .get(seq_len - 1)
            .unwrap()
            .to_vec1()
            .unwrap();

        assert_eq!(std_flat.len(), cached_last.len(), "vocab size mismatch");
        let max_diff = std_flat
            .iter()
            .zip(cached_last.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "prefill logits max diff = {max_diff} (threshold 1e-4)"
        );
    }

    /// Cached prefill + decode should produce the same last-token logits
    /// as forwarding the full sequence in one shot.
    #[test]
    fn test_kv_cache_decode_matches_full_forward() {
        let cfg = tiny_config();
        let mut model = build_model(&cfg);
        let device = Device::Cpu;
        let prompt = vec![1u32, 5, 10];
        let full_seq = vec![1u32, 5, 10, 20];

        // Reference: standard varlen forward on full sequence → last-token logits
        let ref_logits = forward_standard(&mut model, &full_seq, &device);
        let ref_flat: Vec<f32> = ref_logits.flatten_all().unwrap().to_vec1().unwrap();

        // Cached: prefill prompt, then decode one new token
        let prompt_input =
            Tensor::from_vec(prompt.clone(), (prompt.len(),), &device).unwrap();
        let _prefill = model.forward_with_cache(&prompt_input, 0).unwrap();

        let decode_input = Tensor::from_vec(vec![20u32], (1,), &device).unwrap();
        let decode_logits = model.forward_with_cache(&decode_input, prompt.len()).unwrap();
        model.clear_kv_cache();

        let decode_flat: Vec<f32> = decode_logits.get(0).unwrap().to_vec1().unwrap();

        assert_eq!(ref_flat.len(), decode_flat.len(), "vocab size mismatch");
        let max_diff = ref_flat
            .iter()
            .zip(decode_flat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "decode logits max diff = {max_diff} (threshold 1e-4)"
        );
    }

    /// After `clear_kv_cache`, re-running the same input should produce
    /// identical output (cache doesn't leak between requests).
    #[test]
    fn test_kv_cache_clear_determinism() {
        let cfg = tiny_config();
        let mut model = build_model(&cfg);
        let device = Device::Cpu;
        let tokens = vec![1u32, 5, 10];
        let input = Tensor::from_vec(tokens, (3,), &device).unwrap();

        let logits1 = model.forward_with_cache(&input, 0).unwrap();
        model.clear_kv_cache();

        let logits2 = model.forward_with_cache(&input, 0).unwrap();
        model.clear_kv_cache();

        let v1: Vec<f32> = logits1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = logits2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1.len(), v2.len());
        let max_diff = v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "cache clear determinism failed: max diff = {max_diff}"
        );
    }

    /// Multi-step decode: generate 3 tokens with cache, compare against
    /// re-forwarding the full sequence at each step.
    #[test]
    fn test_kv_cache_multi_step_decode() {
        let cfg = tiny_config();
        let mut model = build_model(&cfg);
        let device = Device::Cpu;
        let prompt = vec![1u32, 5, 10];

        // Cached decode: prefill → 3 decode steps
        let prompt_input =
            Tensor::from_vec(prompt.clone(), (prompt.len(),), &device).unwrap();
        let prefill_logits = model.forward_with_cache(&prompt_input, 0).unwrap();
        let mut generated = vec![
            prefill_logits
                .get(prompt.len() - 1)
                .unwrap()
                .argmax(0)
                .unwrap()
                .to_vec0::<u32>()
                .unwrap(),
        ];

        for step in 0..2 {
            let offset = prompt.len() + step;
            let tok = *generated.last().unwrap();
            let input = Tensor::from_vec(vec![tok], (1,), &device).unwrap();
            let logits = model.forward_with_cache(&input, offset).unwrap();
            let next = logits
                .get(0)
                .unwrap()
                .argmax(0)
                .unwrap()
                .to_vec0::<u32>()
                .unwrap();
            generated.push(next);
        }
        model.clear_kv_cache();

        // Reference: re-forward full sequence at each step
        let mut ref_generated = Vec::new();
        let mut full_seq = prompt.clone();
        for _step in 0..3 {
            let ref_logits = forward_standard(&mut model, &full_seq, &device);
            // forward_standard returns [1, 1, vocab] — flatten and use Tensor argmax
            let next = ref_logits
                .flatten_all().unwrap()
                .argmax(0).unwrap()
                .to_vec0::<u32>().unwrap();
            ref_generated.push(next);
            full_seq.push(next);
        }

        assert_eq!(
            generated, ref_generated,
            "multi-step decode mismatch: cached={generated:?} vs reforward={ref_generated:?}"
        );
    }

    /// Microbenchmark: measure each stage of a single decode step for Qwen3-0.6B-sized tensors.
    /// cargo test -p prelude-core --lib --release --features onednn -- qwen3::tests::bench_decode_stages --nocapture --ignored
    #[test]
    #[ignore]
    fn bench_decode_stages() {
        use std::time::Instant;
        use candle_core::Module;

        let hidden = 1024usize;
        let intermediate = 3072usize;
        let num_heads = 16usize;
        let num_kv_heads = 8usize;
        let head_dim = 128usize;
        let context_len = 40usize;
        let device = Device::Cpu;

        let warmup = 5;
        let iters = 20;

        // 1. RMSNorm
        let rmsnorm_us = {
            let x = Tensor::randn(0.0f32, 1.0, (1, hidden), &device).unwrap();
            let norm = crate::nn_ops::CandleRmsNorm::new(
                Tensor::ones((hidden,), DType::F32, &device).unwrap(), 1e-6,
            );
            for _ in 0..warmup { let _ = norm.forward(&x); }
            let t = Instant::now();
            for _ in 0..iters { let _ = norm.forward(&x); }
            t.elapsed().as_micros() as f64 / iters as f64
        };
        eprintln!("[1] RMSNorm          (1×{hidden}):             {rmsnorm_us:.1} µs");

        // 2. Linear QKV via OnednnLinear
        let qkv_us = {
            let qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim;
            let x = Tensor::randn(0.0f32, 1.0, (1, hidden), &device).unwrap();
            let w = Tensor::randn(0.0f32, 1.0, (qkv_dim, hidden), &device).unwrap();
            let linear = crate::ops::onednn::OnednnLinear::new(
                crate::nn_ops::CandleLinear::new(w, None),
            ).unwrap();
            for _ in 0..warmup { let _ = linear.forward(&x); }
            let t = Instant::now();
            for _ in 0..iters { let _ = linear.forward(&x); }
            t.elapsed().as_micros() as f64 / iters as f64
        };
        let qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim;
        eprintln!("[2] Linear QKV       (1×{hidden} → 1×{qkv_dim}):  {qkv_us:.1} µs");

        // 3. Attention candle: reshape+transpose+matmul+softmax+matmul
        let attn_candle_us = {
            let q = Tensor::randn(0.0f32, 1.0, (1, 1, num_heads, head_dim), &device).unwrap()
                .transpose(1, 2).unwrap().contiguous().unwrap();
            let k = Tensor::randn(0.0f32, 1.0, (1, context_len, num_heads, head_dim), &device).unwrap()
                .transpose(1, 2).unwrap().contiguous().unwrap();
            let v = k.clone();
            for _ in 0..warmup {
                let s = q.matmul(&k.transpose(2, 3).unwrap()).unwrap();
                let s = crate::nn_ops::ops::softmax_last_dim(&s).unwrap();
                let _ = s.matmul(&v).unwrap();
            }
            let t = Instant::now();
            for _ in 0..iters {
                let s = q.matmul(&k.transpose(2, 3).unwrap()).unwrap();
                let s = crate::nn_ops::ops::softmax_last_dim(&s).unwrap();
                let _ = s.matmul(&v).unwrap();
            }
            t.elapsed().as_micros() as f64 / iters as f64
        };
        eprintln!("[3] Attn candle      (1q×{context_len}kv, {num_heads}H):     {attn_candle_us:.1} µs");

        // 4. Attention oneDNN cross_attention
        let attn_onednn_us = {
            let q = Tensor::randn(0.0f32, 1.0, (1, num_heads, head_dim), &device).unwrap();
            let k = Tensor::randn(0.0f32, 1.0, (context_len, num_kv_heads, head_dim), &device).unwrap();
            let v = Tensor::randn(0.0f32, 1.0, (context_len, num_kv_heads, head_dim), &device).unwrap();
            for _ in 0..warmup {
                let _ = crate::models::common::attn::cpu::cross_attention_f32_onednn(
                    &q, &k, &v, 1, context_len,
                    num_heads, num_kv_heads, head_dim, 0.088, false, 0,
                );
            }
            let t = Instant::now();
            for _ in 0..iters {
                let _ = crate::models::common::attn::cpu::cross_attention_f32_onednn(
                    &q, &k, &v, 1, context_len,
                    num_heads, num_kv_heads, head_dim, 0.088, false, 0,
                );
            }
            t.elapsed().as_micros() as f64 / iters as f64
        };
        eprintln!("[4] Attn oneDNN      (1q×{context_len}kv, {num_heads}H):     {attn_onednn_us:.1} µs");

        // 4b. Attention native F32 decode kernel
        let attn_native_f32_us = {
            let q = Tensor::randn(0.0f32, 1.0, (1, num_heads, head_dim), &device).unwrap();
            let k = Tensor::randn(0.0f32, 1.0, (context_len, num_kv_heads, head_dim), &device).unwrap();
            let v = Tensor::randn(0.0f32, 1.0, (context_len, num_kv_heads, head_dim), &device).unwrap();
            for _ in 0..warmup {
                let _ = crate::ops::cpu::cpu_decode_attention_f32(
                    &q, &k, &v, context_len,
                    num_heads, num_kv_heads, head_dim, 0.088,
                );
            }
            let t = Instant::now();
            for _ in 0..iters {
                let _ = crate::ops::cpu::cpu_decode_attention_f32(
                    &q, &k, &v, context_len,
                    num_heads, num_kv_heads, head_dim, 0.088,
                );
            }
            t.elapsed().as_micros() as f64 / iters as f64
        };
        eprintln!("[4b] Attn native F32 (1q×{context_len}kv, {num_heads}H):    {attn_native_f32_us:.1} µs");

        // 5. Linear O_proj
        let oproj_us = {
            let proj_in = num_heads * head_dim;
            let x = Tensor::randn(0.0f32, 1.0, (1, proj_in), &device).unwrap();
            let w = Tensor::randn(0.0f32, 1.0, (hidden, proj_in), &device).unwrap();
            let linear = crate::ops::onednn::OnednnLinear::new(
                crate::nn_ops::CandleLinear::new(w, None),
            ).unwrap();
            for _ in 0..warmup { let _ = linear.forward(&x); }
            let t = Instant::now();
            for _ in 0..iters { let _ = linear.forward(&x); }
            t.elapsed().as_micros() as f64 / iters as f64
        };
        eprintln!("[5] Linear O_proj    (1×{} → 1×{hidden}):  {oproj_us:.1} µs", num_heads*head_dim);

        // 6. MLP: gate_up + silu_mul + down
        let mlp_us = {
            let x = Tensor::randn(0.0f32, 1.0, (1, hidden), &device).unwrap();
            let w_gu = Tensor::randn(0.0f32, 1.0, (2 * intermediate, hidden), &device).unwrap();
            let w_down = Tensor::randn(0.0f32, 1.0, (hidden, intermediate), &device).unwrap();
            let lin_gu = crate::ops::onednn::OnednnLinear::new(
                crate::nn_ops::CandleLinear::new(w_gu, None),
            ).unwrap();
            let lin_down = crate::ops::onednn::OnednnLinear::new(
                crate::nn_ops::CandleLinear::new(w_down, None),
            ).unwrap();
            for _ in 0..warmup {
                let h = lin_gu.forward(&x).unwrap();
                let chunks: Vec<_> = h.chunk(2, 1).unwrap();
                let _ = lin_down.forward(
                    &(crate::nn_ops::ops::silu(&chunks[0]).unwrap() * &chunks[1]).unwrap()
                );
            }
            let t = Instant::now();
            for _ in 0..iters {
                let h = lin_gu.forward(&x).unwrap();
                let chunks: Vec<_> = h.chunk(2, 1).unwrap();
                let _ = lin_down.forward(
                    &(crate::nn_ops::ops::silu(&chunks[0]).unwrap() * &chunks[1]).unwrap()
                );
            }
            t.elapsed().as_micros() as f64 / iters as f64
        };
        eprintln!("[6] MLP gate+silu+dn (1×{hidden}→1×{}→1×{hidden}): {mlp_us:.1} µs", 2*intermediate);

        // 7. Residual add
        let add_us = {
            let a = Tensor::randn(0.0f32, 1.0, (1, hidden), &device).unwrap();
            let b = Tensor::randn(0.0f32, 1.0, (1, hidden), &device).unwrap();
            for _ in 0..warmup { let _ = (&a + &b).unwrap(); }
            let t = Instant::now();
            for _ in 0..iters { let _ = (&a + &b).unwrap(); }
            t.elapsed().as_micros() as f64 / iters as f64
        };
        eprintln!("[7] Residual add     (1×{hidden}):             {add_us:.1} µs");

        // Summary
        let linear_total = qkv_us + oproj_us + mlp_us;
        let non_linear = rmsnorm_us * 2.0 + attn_candle_us + add_us * 2.0;
        let layer_total = linear_total + non_linear;
        eprintln!("\n════════════════════════════════════════════════");
        eprintln!("Per-layer breakdown (context_len={context_len}):");
        eprintln!("  Linear (QKV+O+MLP):  {linear_total:.0} µs  ({:.1}%)", linear_total/layer_total*100.0);
        eprintln!("  Non-linear:          {non_linear:.0} µs  ({:.1}%)", non_linear/layer_total*100.0);
        eprintln!("    RMSNorm ×2:        {:.0} µs", rmsnorm_us*2.0);
        eprintln!("    Attn (candle):     {attn_candle_us:.0} µs");
        eprintln!("    Residual add ×2:   {:.0} µs", add_us*2.0);
        eprintln!("  Per-layer total:     {layer_total:.0} µs");
        eprintln!("  28 layers total:     {:.1} ms", layer_total * 28.0 / 1000.0);
        eprintln!("════════════════════════════════════════════════");
    }
}
