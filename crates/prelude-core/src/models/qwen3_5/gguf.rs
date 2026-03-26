//! Quantized Qwen3.5 model loaded from GGUF format.
//!
//! Uses GGML quantized matmul kernels (AVX-512/AMX) when compiled with `ggml-quants` feature,
//! falls back to candle's `QMatMul` otherwise.
//!
//! Reference: llama.cpp `src/models/qwen35.cpp` + `src/models/delta-net-base.cpp`.

use candle_core::quantized::gguf_file;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use std::io::{Read, Seek};
use std::sync::Arc;

use crate::constants::GGUF_INTERMEDIATE_SIZE_MULTIPLIER;

// ── Quantized Linear Layer ──────────────────────────────────────────────

/// Quantized linear layer using candle QMatMul.
/// When `ggml-quants` feature is enabled, the entire model forward is handled
/// by llama.cpp FFI (see LlamaGgufModel below), so this is only used as fallback.
type QuantLinear = QMatMul;

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

// ── Helpers ─────────────────────────────────────────────────────────────

fn l2_normalize_last_dim(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let norm = x_f32.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let norm = (norm + 1e-12)?;
    x_f32.broadcast_div(&norm)?.to_dtype(x.dtype())
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let exp_x = x.exp()?;
    (exp_x + 1.0)?.log()
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(
        &[
            (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?,
            (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?,
        ],
        D::Minus1,
    )
}

// ── RMS Norm (residual: weight = 1 + raw_weight) ───────────────────────

struct ResidualRmsNorm {
    weight: Tensor, // already (1 + raw_weight)
    eps: f64,
}

impl ResidualRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        normed.to_dtype(x.dtype())?.broadcast_mul(&self.weight)
    }
}

// ── RmsNormGated (per-head norm + SiLU gate) ────────────────────────────

struct RmsNormGated {
    weight: Tensor,
    eps: f64,
    num_heads: usize,
    head_dim: usize,
}

impl RmsNormGated {
    fn forward(&self, x: &Tensor, z: &Tensor) -> Result<Tensor> {
        let orig_shape = x.shape().clone();
        let leading: Vec<usize> = orig_shape.dims()[..orig_shape.dims().len() - 1].to_vec();
        let mut new_shape = leading.clone();
        new_shape.push(self.num_heads);
        new_shape.push(self.head_dim);

        let x = x.reshape(new_shape.as_slice())?;
        let z = z.reshape(new_shape.as_slice())?;

        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let normed = normed.to_dtype(x.dtype())?.broadcast_mul(&self.weight)?;
        let gate = crate::nn_ops::Activation::Silu.forward(&z)?;
        let result = normed.broadcast_mul(&gate)?;
        result.reshape(orig_shape)
    }
}

// ── Partial RoPE ────────────────────────────────────────────────────────

struct PartialRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl PartialRotaryEmbedding {
    fn new(cfg: &Qwen3_5GgufConfig, dtype: DType, device: &Device) -> Result<Self> {
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

    fn apply(&self, q: &Tensor, k: &Tensor, pos: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, pos, 1)?; // [1, rotary_dim/2]
        let sin = self.sin.narrow(0, pos, 1)?;

        let apply_one = |x: &Tensor| -> Result<Tensor> {
            let last = x.dim(D::Minus1)?;
            if self.rotary_dim >= last {
                apply_rotary_emb(x, &cos, &sin)
            } else {
                let x_rot = x.narrow(D::Minus1, 0, self.rotary_dim)?;
                let x_pass = x.narrow(D::Minus1, self.rotary_dim, last - self.rotary_dim)?;
                let x_rot = apply_rotary_emb(&x_rot, &cos, &sin)?;
                Tensor::cat(&[x_rot, x_pass], D::Minus1)
            }
        };
        Ok((apply_one(q)?, apply_one(k)?))
    }
}

// ── Gated DeltaNet Layer ────────────────────────────────────────────────

struct GatedDeltaNetLayer {
    in_proj_qkv: QuantLinear,
    in_proj_z: QuantLinear,
    in_proj_b: QuantLinear,
    in_proj_a: QuantLinear,
    conv_weight: Tensor, // [conv_dim, kernel_size]
    dt_bias: Tensor,
    a_log: Tensor,
    norm: RmsNormGated,
    out_proj: QuantLinear,
    // State
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
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

impl GatedDeltaNetLayer {
    fn clear_state(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(0)?;

        // Input projections (dequantize for matmul)
        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;
        let b_param = self.in_proj_b.forward(x)?;
        let a_param = self.in_proj_a.forward(x)?;

        // Concat Q|K|V for conv1d
        let q_cat = qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k_cat = qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v_cat = qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;
        let qkv_for_conv = Tensor::cat(&[&q_cat, &k_cat, &v_cat], D::Minus1)?;

        // Conv1d
        let qkv_conv = if seq_len == 1 {
            self.conv1d_decode(&qkv_for_conv.squeeze(0)?)?
                .unsqueeze(0)?
        } else {
            self.conv1d_prefill(&qkv_for_conv)?
        };

        // SiLU activation
        let qkv_conv = crate::nn_ops::Activation::Silu.forward(&qkv_conv)?;

        // Split
        let q = qkv_conv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = qkv_conv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = qkv_conv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        // Process each timestep with delta rule
        let device = x.device();
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let q_t = q.get(t)?.contiguous()?;
            let k_t = k.get(t)?.contiguous()?;
            let v_t = v.get(t)?.contiguous()?;
            let b_t = b_param.get(t)?.contiguous()?;
            let a_t = a_param.get(t)?.contiguous()?;
            let out_t = self.delta_rule_step(&q_t, &k_t, &v_t, &b_t, &a_t, device)?;
            outputs.push(out_t);
        }

        let output = Tensor::stack(&outputs, 0)?; // [L, value_dim]
        let z = z.contiguous()?;
        let normed = self.norm.forward(&output, &z)?;
        self.out_proj.forward(&normed)
    }

    fn delta_rule_step(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        b: &Tensor,
        a: &Tensor,
        device: &Device,
    ) -> Result<Tensor> {
        let kv_ratio = self.num_v_heads / self.num_k_heads;

        // L2-normalize q and k per head
        let q = q.reshape((self.num_k_heads, self.head_k_dim))?;
        let q = l2_normalize_last_dim(&q)?;
        let k = k.reshape((self.num_k_heads, self.head_k_dim))?;
        let k = l2_normalize_last_dim(&k)?;
        let v = v.reshape((self.num_v_heads, self.head_v_dim))?;

        // Gating (f32 for numerical stability)
        // ssm_a in GGUF is already -exp(A_log), pre-computed by convert_hf_to_gguf.py
        let dt_bias = self.dt_bias.to_dtype(DType::F32)?;
        let ssm_a = self.a_log.to_dtype(DType::F32)?; // already -exp(A_log)
        let a_f32 = a.to_dtype(DType::F32)?;
        let b_f32 = b.to_dtype(DType::F32)?;

        // gate = softplus(alpha + dt_bias) * ssm_a  (ssm_a is negative → gate is negative → exp(gate) < 1)
        let a_plus_dt = (a_f32 + dt_bias)?;
        let softplus_val = softplus(&a_plus_dt)?;
        let g = (softplus_val * ssm_a)?;
        let beta = crate::nn_ops::ops::sigmoid(&b_f32)?;
        let decay = g.exp()?;

        // Initialize state
        if self.recurrent_state.is_none() {
            self.recurrent_state = Some(Tensor::zeros(
                (self.num_v_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                device,
            )?);
        }
        let state = self.recurrent_state.as_ref().unwrap();

        // state = state * decay + outer(k, beta * (v - state^T @ k))
        let decay_3d = decay.reshape((self.num_v_heads, 1, 1))?;
        let state_decayed = state.broadcast_mul(&decay_3d)?;

        let k_f32 = k.to_dtype(DType::F32)?;
        let k_expanded = if kv_ratio > 1 {
            k_f32
                .unsqueeze(1)?
                .expand((self.num_k_heads, kv_ratio, self.head_k_dim))?
                .reshape((self.num_v_heads, self.head_k_dim))?
        } else {
            k_f32
        };
        let k_col = k_expanded.unsqueeze(D::Minus1)?;

        let state_k = state_decayed
            .transpose(1, 2)?
            .matmul(&k_col)?
            .squeeze(D::Minus1)?;
        let v_f32 = v.to_dtype(DType::F32)?;
        let v_error = (v_f32 - state_k)?;
        let beta_2d = beta.reshape((self.num_v_heads, 1))?;
        let v_prime = v_error.broadcast_mul(&beta_2d)?;

        let v_row = v_prime.unsqueeze(1)?;
        let outer = k_col.matmul(&v_row)?;
        let state = (state_decayed + outer)?;

        // output = state^T @ (q / sqrt(head_k_dim))
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
        let q_col = q_expanded.unsqueeze(D::Minus1)?;
        let out = state.transpose(1, 2)?.matmul(&q_col)?;
        let out = out.squeeze(D::Minus1)?;
        let out = out.to_dtype(v.dtype())?;
        let out = out.reshape((self.value_dim,))?;

        self.recurrent_state = Some(state);
        Ok(out)
    }

    fn conv1d_decode(&mut self, x: &Tensor) -> Result<Tensor> {
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

        let x_col = x.unsqueeze(D::Minus1)?;
        let full_window = Tensor::cat(&[state, &x_col], 1)?;
        let out = (full_window * &self.conv_weight)?.sum(D::Minus1)?;

        let new_state = if self.conv_kernel > 2 {
            let kept = state.narrow(1, 1, self.conv_kernel - 2)?;
            Tensor::cat(&[kept, x_col], 1)?
        } else {
            x_col
        };
        self.conv_state = Some(new_state);
        Ok(out)
    }

    fn conv1d_prefill(&mut self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(0)?;
        let device = x.device();
        let dtype = x.dtype();

        // Transpose to [conv_dim, L]
        let x_t = x.t()?;
        let pad_len = self.conv_kernel - 1;
        let prefix = if let Some(ref state) = self.conv_state {
            state.clone()
        } else {
            Tensor::zeros((self.conv_dim, pad_len), dtype, device)?
        };
        let padded = Tensor::cat(&[prefix, x_t.clone()], 1)?; // [conv_dim, pad+L]

        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let window = padded.narrow(1, t, self.conv_kernel)?;
            let out = (window * &self.conv_weight)?.sum(D::Minus1)?;
            outputs.push(out);
        }
        let result = Tensor::stack(&outputs, 0)?; // [L, conv_dim]

        // Save last kernel-1 as state
        if seq_len >= pad_len {
            self.conv_state = Some(x_t.narrow(1, seq_len - pad_len, pad_len)?);
        } else {
            let old = if let Some(ref state) = self.conv_state {
                state.narrow(1, seq_len, pad_len - seq_len)?
            } else {
                Tensor::zeros((self.conv_dim, pad_len - seq_len), dtype, device)?
            };
            self.conv_state = Some(Tensor::cat(&[old, x_t], 1)?);
        }

        Ok(result)
    }
}

// ── Gated Full Attention Layer ──────────────────────────────────────────

struct GatedAttentionLayer {
    q_proj: QuantLinear,
    k_proj: QuantLinear,
    v_proj: QuantLinear,
    o_proj: QuantLinear,
    q_norm: ResidualRmsNorm,
    k_norm: ResidualRmsNorm,
    rope: PartialRotaryEmbedding,
    kv_cache: Option<(Tensor, Tensor)>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_output_gate: bool,
}

impl GatedAttentionLayer {
    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }

    fn forward(&mut self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(0)?;
        let q_raw = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Split Q + gate
        let (q, gate) = if self.attn_output_gate {
            let q_and_gate = q_raw.reshape((seq_len, self.num_heads, self.head_dim * 2))?;
            let q = q_and_gate.narrow(D::Minus1, 0, self.head_dim)?;
            let gate = q_and_gate
                .narrow(D::Minus1, self.head_dim, self.head_dim)?
                .reshape((seq_len, self.num_heads * self.head_dim))?;
            (q, Some(gate))
        } else {
            let q = q_raw.reshape((seq_len, self.num_heads, self.head_dim))?;
            (q, None)
        };

        let k = k.reshape((seq_len, self.num_kv_heads, self.head_dim))?;
        let mut v = v.reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        // Per-head Q/K normalization
        let q = self.q_norm.forward(
            &q.reshape((seq_len * self.num_heads, self.head_dim))?,
        )?.reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = self.k_norm.forward(
            &k.reshape((seq_len * self.num_kv_heads, self.head_dim))?,
        )?.reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        // Partial RoPE (applied per-token via pos offset)
        let (q, mut k) = if seq_len == 1 {
            // Decode: single position
            let q2d = q.reshape((self.num_heads, self.head_dim))?;
            let k2d = k.reshape((self.num_kv_heads, self.head_dim))?;
            let (q_r, k_r) = self.rope.apply(&q2d, &k2d, pos)?;
            (
                q_r.reshape((1, self.num_heads, self.head_dim))?,
                k_r.reshape((1, self.num_kv_heads, self.head_dim))?,
            )
        } else {
            // Prefill: apply per-token
            let mut q_out = Vec::with_capacity(seq_len);
            let mut k_out = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let q_t = q.get(t)?; // [num_heads, head_dim]
                let k_t = k.get(t)?;
                let (q_r, k_r) = self.rope.apply(&q_t, &k_t, pos + t)?;
                q_out.push(q_r);
                k_out.push(k_r);
            }
            (Tensor::stack(&q_out, 0)?, Tensor::stack(&k_out, 0)?)
        };

        // GQA repeat
        let kv_ratio = self.num_heads / self.num_kv_heads;
        if kv_ratio > 1 {
            k = k
                .unsqueeze(2)?
                .expand((seq_len, self.num_kv_heads, kv_ratio, self.head_dim))?
                .reshape((seq_len, self.num_heads, self.head_dim))?;
            v = v
                .unsqueeze(2)?
                .expand((seq_len, self.num_kv_heads, kv_ratio, self.head_dim))?
                .reshape((seq_len, self.num_heads, self.head_dim))?;
        }

        // KV cache
        let (k, v) = if let Some((ref cached_k, ref cached_v)) = self.kv_cache {
            let k = Tensor::cat(&[cached_k, &k], 0)?;
            let v = Tensor::cat(&[cached_v, &v], 0)?;
            (k, v)
        } else {
            (k, v)
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let kv_len = k.dim(0)?;

        // Attention: Q [L, H, D] @ K^T [kv_len, H, D] → [H, L, kv_len]
        let q = q.transpose(0, 1)?; // [H, L, D]
        let k = k.transpose(0, 1)?; // [H, kv_len, D]
        let v = v.transpose(0, 1)?; // [H, kv_len, D]

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn_weights = (q.matmul(&k.transpose(1, 2)?)? * scale)?;

        // Causal mask
        let attn_weights = if seq_len > 1 {
            let offset = kv_len - seq_len; // how many past KV tokens before current Q
            let mut mask_data = vec![0.0f32; seq_len * kv_len];
            for i in 0..seq_len {
                for j in (offset + i + 1)..kv_len {
                    mask_data[i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
            let causal_mask =
                Tensor::from_vec(mask_data, (1, seq_len, kv_len), x.device())?;
            attn_weights.to_dtype(DType::F32)?.broadcast_add(&causal_mask)?
        } else {
            attn_weights.to_dtype(DType::F32)?
        };

        let attn_weights = crate::nn_ops::ops::softmax_last_dim(&attn_weights)?;
        let attn_weights = attn_weights.to_dtype(v.dtype())?;
        let attn_output = attn_weights.matmul(&v)?; // [H, L, D]
        let attn_output = attn_output
            .transpose(0, 1)? // [L, H, D]
            .reshape((seq_len, self.num_heads * self.head_dim))?;

        // Gate
        let gated = if let Some(gate) = gate {
            (attn_output * crate::nn_ops::ops::sigmoid(&gate)?)?
        } else {
            attn_output
        };
        self.o_proj.forward(&gated)
    }
}

// ── MLP ─────────────────────────────────────────────────────────────────

struct Mlp {
    gate_proj: QuantLinear,
    up_proj: QuantLinear,
    down_proj: QuantLinear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = crate::nn_ops::Activation::Silu.forward(&gate)?;
        let hidden = (act * up)?;
        self.down_proj.forward(&hidden)
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────

enum TokenMixer {
    Linear(GatedDeltaNetLayer),
    FullAttn(GatedAttentionLayer),
}

struct DecoderLayer {
    input_layernorm: ResidualRmsNorm,
    post_attention_layernorm: ResidualRmsNorm,
    token_mixer: TokenMixer,
    mlp: Mlp,
}

impl DecoderLayer {
    fn forward(&mut self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let h = match &mut self.token_mixer {
            TokenMixer::Linear(gdn) => gdn.forward(&h)?,
            TokenMixer::FullAttn(attn) => attn.forward(&h, pos)?,
        };
        let x = (x + h)?;
        let h2 = self.post_attention_layernorm.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        (x + h2).map(|t| t)
    }

    fn clear_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::Linear(gdn) => gdn.clear_state(),
            TokenMixer::FullAttn(attn) => attn.clear_cache(),
        }
    }
}

// ── Model ───────────────────────────────────────────────────────────────

pub struct Qwen3_5GgufModel {
    embed_tokens: crate::nn_ops::Embedding,
    layers: Vec<DecoderLayer>,
    norm: ResidualRmsNorm,
    lm_head: QuantLinear,
}

impl Qwen3_5GgufModel {
    fn load_qmatmul<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<QuantLinear> {
        let qtensor = ct.tensor(reader, name, device)?;
        QMatMul::from_arc(Arc::new(qtensor))
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

        // Embedding
        let embed_weight = Self::load_tensor(&ct, reader, "token_embd.weight", device)?;
        let embed_tokens =
            crate::nn_ops::Embedding::new(embed_weight, config.hidden_size);

        // Build layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let dtype = DType::F32;

        for i in 0..config.num_hidden_layers {
            let prefix = format!("blk.{i}");

            // Layer norms (residual: 1 + weight)
            // GGUF conversion already applies +1 to norm weights (residual RMSNorm)
            let input_ln_w = Self::load_tensor(&ct, reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let post_ln_w =
                Self::load_tensor(&ct, reader, &format!("{prefix}.post_attention_norm.weight"), device)?;
            let input_layernorm = ResidualRmsNorm {
                weight: input_ln_w,
                eps: config.rms_norm_eps,
            };
            let post_attention_layernorm = ResidualRmsNorm {
                weight: post_ln_w,
                eps: config.rms_norm_eps,
            };

            // Token mixer
            let token_mixer = if config.is_recurrent(i) {
                // DeltaNet layer — tensor names follow GGUF convention
                let in_proj_qkv = Self::load_qmatmul(&ct, reader, &format!("{prefix}.attn_qkv.weight"), device)?;
                let in_proj_z = Self::load_qmatmul(&ct, reader, &format!("{prefix}.attn_gate.weight"), device)?;
                let in_proj_b = Self::load_qmatmul(&ct, reader, &format!("{prefix}.ssm_beta.weight"), device)?;
                let in_proj_a = Self::load_qmatmul(&ct, reader, &format!("{prefix}.ssm_alpha.weight"), device)?;

                // Conv1d weight: need [conv_channels, d_conv] for element-wise mul.
                // GGUF stores [d_conv, conv_channels], so transpose if needed.
                let conv_weight_raw =
                    Self::load_tensor(&ct, reader, &format!("{prefix}.ssm_conv1d.weight"), device)?;
                let conv_weight = if conv_weight_raw.dim(0)? == config.linear_conv_kernel_dim {
                    conv_weight_raw.t()?.contiguous()?
                } else {
                    conv_weight_raw
                };

                let dt_bias = Self::load_tensor(&ct, reader, &format!("{prefix}.ssm_dt.bias"), device)?;
                let a_log = Self::load_tensor(&ct, reader, &format!("{prefix}.ssm_a"), device)?;

                let norm_weight =
                    Self::load_tensor(&ct, reader, &format!("{prefix}.ssm_norm.weight"), device)?;
                let norm = RmsNormGated {
                    weight: norm_weight,
                    eps: config.rms_norm_eps,
                    num_heads: config.linear_num_value_heads,
                    head_dim: config.linear_value_head_dim,
                };

                let out_proj = Self::load_qmatmul(&ct, reader, &format!("{prefix}.ssm_out.weight"), device)?;

                TokenMixer::Linear(GatedDeltaNetLayer {
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
                    key_dim: config.key_dim(),
                    value_dim: config.value_dim(),
                    conv_dim: config.conv_dim(),
                    conv_kernel: config.linear_conv_kernel_dim,
                })
            } else {
                // Full attention layer
                let q_proj = Self::load_qmatmul(&ct, reader, &format!("{prefix}.attn_q.weight"), device)?;
                let k_proj = Self::load_qmatmul(&ct, reader, &format!("{prefix}.attn_k.weight"), device)?;
                let v_proj = Self::load_qmatmul(&ct, reader, &format!("{prefix}.attn_v.weight"), device)?;
                let o_proj = Self::load_qmatmul(&ct, reader, &format!("{prefix}.attn_output.weight"), device)?;

                let q_norm_w =
                    Self::load_tensor(&ct, reader, &format!("{prefix}.attn_q_norm.weight"), device)?;
                let k_norm_w =
                    Self::load_tensor(&ct, reader, &format!("{prefix}.attn_k_norm.weight"), device)?;

                let q_norm = ResidualRmsNorm {
                    weight: q_norm_w,
                    eps: config.rms_norm_eps,
                };
                let k_norm = ResidualRmsNorm {
                    weight: k_norm_w,
                    eps: config.rms_norm_eps,
                };

                let rope = PartialRotaryEmbedding::new(&config, dtype, device)?;

                TokenMixer::FullAttn(GatedAttentionLayer {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    rope,
                    kv_cache: None,
                    num_heads: config.num_attention_heads,
                    num_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    attn_output_gate: true,
                })
            };

            // MLP (same for both layer types)
            let mlp = Mlp {
                gate_proj: Self::load_qmatmul(&ct, reader, &format!("{prefix}.ffn_gate.weight"), device)?,
                up_proj: Self::load_qmatmul(&ct, reader, &format!("{prefix}.ffn_up.weight"), device)?,
                down_proj: Self::load_qmatmul(&ct, reader, &format!("{prefix}.ffn_down.weight"), device)?,
            };

            layers.push(DecoderLayer {
                input_layernorm,
                post_attention_layernorm,
                token_mixer,
                mlp,
            });
        }

        // Final norm
        // GGUF conversion already applies +1 to norm weights (residual RMSNorm)
        let final_norm_w = Self::load_tensor(&ct, reader, "output_norm.weight", device)?;
        let norm = ResidualRmsNorm {
            weight: final_norm_w,
            eps: config.rms_norm_eps,
        };

        // LM head
        let lm_head = if ct.tensor_infos.get("output.weight").is_some() {
            Self::load_qmatmul(&ct, reader, "output.weight", device)?
        } else {
            // Tied embeddings: use token_embd.weight
            Self::load_qmatmul(&ct, reader, "token_embd.weight", device)?
        };

        Ok((
            Self {
                embed_tokens,
                layers,
                norm,
                lm_head,
            },
            config,
        ))
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }

    /// Sequential forward pass (no varlen, no paged KV).
    /// `position_offset` is the starting position for RoPE (0 for prefill, prompt_len for decode).
    pub fn forward_with_cache(&mut self, input_ids: &Tensor, position_offset: usize) -> Result<Tensor> {
        let mut h = self.embed_tokens.forward(input_ids)?;

        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, position_offset)?;
        }

        h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }
}

impl crate::models::ModelForward for Qwen3_5GgufModel {
    fn forward(
        &mut self,
        _packed_input: &Tensor,
        _ctx: &mut crate::models::common::BatchAttnContext,
    ) -> Result<Tensor> {
        candle_core::bail!("GGUF model does not support varlen forward")
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
        Qwen3_5GgufModel::forward_with_cache(self, input_ids, position_offset)
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache();
    }
}

// ── GGUF Config Parsing ─────────────────────────────────────────────────

fn parse_gguf_config(ct: &gguf_file::Content) -> Result<Qwen3_5GgufConfig> {
    let md = &ct.metadata;

    let get_u32 = |key: &str| -> Result<usize> {
        md.get(key)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing GGUF metadata: {key}")))?
            .to_u32()
            .map(|v| v as usize)
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
        md.get(key)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing GGUF metadata: {key}")))?
            .to_f32()
            .map(|v| v as f64)
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

// ── llama.cpp FFI Model ─────────────────────────────────────────────────
// When `ggml-quants` feature is enabled, this model delegates the entire
// forward pass to llama.cpp via FFI — all quantized matmul, DeltaNet,
// attention, conv1d handled by llama.cpp's optimized kernels.

#[cfg(feature = "ggml-quants")]
pub struct LlamaGgufModel {
    model: prelude_ggml_quants::LlamaModel,
    ctx: prelude_ggml_quants::LlamaContext,
    n_vocab: usize,
}

#[cfg(feature = "ggml-quants")]
impl LlamaGgufModel {
    pub fn load(
        gguf_path: &std::path::Path,
        n_gpu_layers: i32,
        n_ctx: u32,
    ) -> std::result::Result<Self, String> {
        let model = prelude_ggml_quants::LlamaModel::load(gguf_path, n_gpu_layers)?;
        let n_vocab = model.n_vocab();
        let ctx = prelude_ggml_quants::LlamaContext::new(&model, n_ctx, n_ctx)?;
        Ok(Self { model, ctx, n_vocab })
    }

    pub fn config(&self) -> LlamaGgufConfig {
        LlamaGgufConfig {
            vocab_size: self.model.n_vocab(),
            num_hidden_layers: self.model.n_layer(),
            max_position_embeddings: self.model.n_ctx_train(),
            num_attention_heads: self.model.n_head(),
            num_key_value_heads: self.model.n_head_kv(),
            head_dim: self.model.n_embd() / self.model.n_head(),
            eos_token: self.model.eos_token(),
        }
    }

    pub fn chat_template(&self) -> Option<String> {
        self.model.chat_template()
    }

    /// Generate tokens via llama.cpp's C-side decode loop (zero Rust↔C overhead per token).
    /// Returns (generated_token_ids, last_logits).
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new: usize,
    ) -> std::result::Result<(Vec<u32>, Vec<f32>), String> {
        let prompt: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
        let (tokens, logits) = self.ctx.generate(self.model.vocab(), &prompt, max_new)?;
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        Ok((tokens_u32, logits))
    }
}

#[cfg(feature = "ggml-quants")]
pub struct LlamaGgufConfig {
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub eos_token: i32,
}

#[cfg(feature = "ggml-quants")]
impl crate::models::ModelForward for LlamaGgufModel {
    fn forward(
        &mut self,
        _packed_input: &Tensor,
        _ctx: &mut crate::models::common::BatchAttnContext,
    ) -> Result<Tensor> {
        candle_core::bail!("llama.cpp GGUF model does not support varlen forward")
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Tensor,
        _position_offset: usize,
    ) -> Result<Tensor> {
        let tokens: Vec<i32> = input_ids.to_vec1::<u32>()?
            .iter()
            .map(|&t| t as i32)
            .collect();
        let logits = self.ctx.decode_tokens(&tokens)
            .map_err(|e| candle_core::Error::Msg(e))?;
        // Return [L, vocab_size] where L = number of input tokens.
        // llama.cpp only returns logits for the last token, so we
        // place them at position L-1 (callers use .get(L-1) or .get(0) for L=1).
        let seq_len = tokens.len();
        if seq_len == 1 {
            // Decode step: return [1, vocab_size] directly
            Tensor::from_vec(logits, (1, self.n_vocab), &candle_core::Device::Cpu)
        } else {
            // Prefill: return [L, vocab_size] with logits only at last position
            let mut data = vec![0.0f32; seq_len * self.n_vocab];
            let offset = (seq_len - 1) * self.n_vocab;
            data[offset..offset + self.n_vocab].copy_from_slice(&logits);
            Tensor::from_vec(data, (seq_len, self.n_vocab), &candle_core::Device::Cpu)
        }
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn generate_direct(
        &mut self,
        prompt_tokens: &[u32],
        max_new: usize,
    ) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        let (tokens, logits) = self.generate(prompt_tokens, max_new)
            .map_err(|e| candle_core::Error::Msg(e))?;
        Ok(Some((tokens, logits)))
    }

    fn clear_kv_cache(&mut self) {
        self.ctx.clear_kv_cache();
    }
}
