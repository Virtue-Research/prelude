//! Qwen3-Next: Hybrid attention (Gated DeltaNet + Gated Attention) with extreme-sparsity MoE.
//!
//! Architecture:
//! - 48 layers with alternating token mixers:
//!   - Layers where (i+1) % full_attention_interval == 0: standard gated softmax attention
//!   - All other layers: Gated DeltaNet (linear attention with delta rule recurrence)
//! - Every layer has MoE (512 experts, top-10 routing + 1 shared expert)
//!
//! Portions of this implementation are derived from:
//! - SGLang: <https://github.com/sgl-project/sglang/blob/78ddf05a/python/sglang/srt/models/qwen3_next.py>
//! - HuggingFace `modeling_qwen3_next.py`
//! SGLang is licensed under the Apache License, Version 2.0.

use crate::loading::var_builder::VarBuilder;
use crate::models::commons::embedding::Embedding;
use crate::models::commons::linear::DenseLinear;
use crate::tensor::{DType, Device, Module, Result, Tensor, D};

use crate::models::commons::{
    last_token_select, BatchAttnContext, BatchState, LayerAttnContext, Linear,
};
use crate::models::model_config;
use crate::ops::{MaskType, PagedParams, VarlenParams};
use std::collections::BTreeMap;

// ── Config ──────────────────────────────────────────────────────────────

model_config! {
    pub struct Qwen3NextConfig("Qwen3Next") {
        required {
            vocab_size: usize,
            hidden_size: usize,
            intermediate_size: usize,
            num_hidden_layers: usize,
            num_attention_heads: usize,
            num_key_value_heads: usize,
            head_dim: usize,
            max_position_embeddings: usize,
        }
        serde_default {
            norm_topk_prob: bool,
            tie_word_embeddings: bool,
        }
        warn_default {
            rms_norm_eps: f64 = 1e-6,
            rope_theta: f64 = 10_000_000.0,
            partial_rotary_factor: f64 = 0.25,
            full_attention_interval: usize = 4,
            // DeltaNet
            linear_num_key_heads: usize = 16,
            linear_num_value_heads: usize = 32,
            linear_key_head_dim: usize = 128,
            linear_value_head_dim: usize = 128,
            linear_conv_kernel_dim: usize = 4,
            // MoE
            num_experts: usize = 512,
            num_experts_per_tok: usize = 10,
            moe_intermediate_size: usize = 512,
            shared_expert_intermediate_size: usize = 512,
            decoder_sparse_step: usize = 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerType {
    LinearAttention,
    FullAttention,
}

impl Qwen3NextConfig {
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
}

// ── RoPE with partial rotary factor ─────────────────────────────────────

struct PartialRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl PartialRotaryEmbedding {
    fn new(cfg: &Qwen3NextConfig, dtype: DType, device: &Device) -> Result<Self> {
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

    /// Apply partial RoPE with per-token position_ids for varlen paths.
    /// q, k shape: [total_tokens, num_heads, head_dim]
    fn apply_varlen(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.index_select(position_ids, 0)?.unsqueeze(1)?;
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

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(
        &[
            (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?,
            (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?,
        ],
        D::Minus1,
    )?;
    Ok(rotated)
}

// ── RMSNormGated ────────────────────────────────────────────────────────

struct RmsNormGated {
    weight: Tensor,
    eps: f64,
    num_heads: usize,
    head_dim: usize,
}

impl RmsNormGated {
    fn new(head_dim: usize, num_heads: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(head_dim, "weight")?;
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

        // RMS norm on last dimension (head_dim)
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let normed = normed.to_dtype(x.dtype())?.broadcast_mul(&self.weight)?;
        let gate = ops.silu(&z)?;
        let result = normed.broadcast_mul(&gate)?;

        // Reshape back to [..., num_heads * head_dim]
        result.reshape(orig_shape)
    }
}

// ── Gated DeltaNet (Linear Attention) ────────────────────────────────────

struct Qwen3NextGatedDeltaNet {
    in_proj_qkvz: Linear,
    in_proj_ba: Linear,
    conv_weight: Tensor, // [conv_dim, kernel_size] reshaped for dot product
    dt_bias: Tensor,     // [num_v_heads]
    a_log: Tensor,       // [num_v_heads]
    norm: RmsNormGated,
    out_proj: Linear,
    // State
    conv_state: Option<Tensor>,      // [conv_dim, kernel-1]
    recurrent_state: Option<Tensor>, // [num_v_heads, v_dim, k_dim] in f32
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

impl Qwen3NextGatedDeltaNet {
    fn new(cfg: &Qwen3NextConfig, vb: VarBuilder) -> Result<Self> {
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let proj_dim = key_dim * 2 + value_dim * 2; // Q + K + V + Z

        let in_proj_qkvz = Linear::load(vb.pp("in_proj_qkvz"), cfg.hidden_size, proj_dim, false)?;
        let in_proj_ba = Linear::load(
            vb.pp("in_proj_ba"),
            cfg.hidden_size,
            cfg.linear_num_value_heads * 2,
            false,
        )?;

        // Conv1d weight: stored as [conv_dim, 1, kernel_size], reshape to [conv_dim, kernel_size]
        let conv_weight_raw = vb.get((conv_dim, 1, cfg.linear_conv_kernel_dim), "conv1d.weight")?;
        let conv_weight = conv_weight_raw.squeeze(1)?;

        let dt_bias = vb.get(cfg.linear_num_value_heads, "dt_bias")?;
        let a_log = vb.get(cfg.linear_num_value_heads, "A_log")?;

        let norm = RmsNormGated::new(
            cfg.linear_value_head_dim,
            cfg.linear_num_value_heads,
            cfg.rms_norm_eps,
            vb.pp("norm"),
        )?;
        let out_proj = Linear::load(vb.pp("out_proj"), value_dim, cfg.hidden_size, false)?;

        Ok(Self {
            in_proj_qkvz,
            in_proj_ba,
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

    fn project_qkvz_ba(
        &self,
        x: &Tensor,
        bs: &BatchState,
        ops: &dyn crate::ops::Ops,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (b, seq_len, _) = x.dims3()?;
        let kv_ratio = self.num_v_heads / self.num_k_heads;

        let full_proj = self.in_proj_qkvz.forward(x, bs, ops)?;
        let full_proj = full_proj.reshape((
            b,
            seq_len,
            self.num_k_heads,
            self.head_k_dim * 2 + kv_ratio * self.head_v_dim * 2,
        ))?;
        let q =
            full_proj
                .narrow(D::Minus1, 0, self.head_k_dim)?
                .reshape((b, seq_len, self.key_dim))?;
        let k = full_proj
            .narrow(D::Minus1, self.head_k_dim, self.head_k_dim)?
            .reshape((b, seq_len, self.key_dim))?;
        let v_group_dim = kv_ratio * self.head_v_dim;
        let v = full_proj
            .narrow(D::Minus1, self.head_k_dim * 2, v_group_dim)?
            .reshape((b, seq_len, self.value_dim))?;
        let z = full_proj
            .narrow(D::Minus1, self.head_k_dim * 2 + v_group_dim, v_group_dim)?
            .reshape((b, seq_len, self.value_dim))?;
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1)?;

        let ba = self.in_proj_ba.forward(x, bs, ops)?;
        let ba = ba.reshape((b, seq_len, self.num_k_heads, kv_ratio * 2))?;
        let b_param = ba
            .narrow(D::Minus1, 0, kv_ratio)?
            .reshape((b, seq_len, self.num_v_heads))?;
        let a_param =
            ba.narrow(D::Minus1, kv_ratio, kv_ratio)?
                .reshape((b, seq_len, self.num_v_heads))?;

        Ok((qkv, z, b_param, a_param))
    }

    /// Forward pass for a single token (decode) or a sequence (prefill).
    fn forward(&mut self, x: &Tensor, _offset: usize, ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        assert_eq!(b, 1, "Qwen3-Next DeltaNet only supports batch_size=1");
        let bs = BatchState::no_lora();
        let (qkv, z, b_param, a_param) = self.project_qkvz_ba(x, &bs, ops)?;
        let device = x.device();

        // Apply causal conv1d
        let qkv_conv = if seq_len == 1 {
            self.conv1d_decode(&qkv.squeeze(0)?.squeeze(0)?, ops)?
                .unsqueeze(0)?
                .unsqueeze(0)?
        } else {
            self.conv1d_prefill(&qkv, ops)?
        };

        // Apply SiLU activation after conv
        let qkv_conv = ops.silu(&qkv_conv)?;

        if seq_len > 1 && self.head_k_dim == self.head_v_dim {
            let qkv_2d = qkv_conv.squeeze(0)?.contiguous()?;
            let b_2d = b_param.squeeze(0)?.contiguous()?;
            let a_2d = a_param.squeeze(0)?.contiguous()?;
            if let Some(delta_out) =
                self.delta_rule_prefill_gdn(&qkv_2d, &b_2d, &a_2d, device, ops)?
            {
                let output = delta_out.unsqueeze(0)?;
                let z = z.contiguous()?;
                let normed = self.norm.forward(&output, &z, ops)?;
                return self.out_proj.forward(&normed, &bs, ops);
            }
        }

        // Split into q, k, v
        let q = qkv_conv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = qkv_conv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = qkv_conv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        // Process each timestep
        let mut outputs = Vec::with_capacity(seq_len);

        // Squeeze batch dim: [1, L, dim] -> [L, dim]
        let q = q.get(0)?;
        let k = k.get(0)?;
        let v = v.get(0)?;
        let b_param = b_param.get(0)?;
        let a_param = a_param.get(0)?;

        for t in 0..seq_len {
            let q_t = q.get(t)?.contiguous()?; // [key_dim]
            let k_t = k.get(t)?.contiguous()?; // [key_dim]
            let v_t = v.get(t)?.contiguous()?; // [value_dim]
            let b_t = b_param.get(t)?.contiguous()?; // [num_v_heads]
            let a_t = a_param.get(t)?.contiguous()?; // [num_v_heads]

            let out_t = self.delta_rule_step(&q_t, &k_t, &v_t, &b_t, &a_t, device, ops)?;
            outputs.push(out_t);
        }

        // Stack outputs: [1, L, value_dim]
        let output = Tensor::stack(&outputs, 0)?.unsqueeze(0)?;

        // Reshape z for gated norm: [1, L, value_dim]
        let z = z.contiguous()?;

        // Gated RMSNorm + output projection
        let normed = self.norm.forward(&output, &z, ops)?;
        self.out_proj.forward(&normed, &bs, ops)
    }

    /// Single-step delta rule update (batched across all v_heads).
    fn delta_rule_step(
        &mut self,
        q: &Tensor, // [key_dim]
        k: &Tensor, // [key_dim]
        v: &Tensor, // [value_dim]
        b: &Tensor, // [num_v_heads]
        a: &Tensor, // [num_v_heads]
        device: &Device,
        ops: &dyn crate::ops::Ops,
    ) -> Result<Tensor> {
        let kv_ratio = self.num_v_heads / self.num_k_heads;

        // Reshape to heads
        let q_heads = q.reshape((self.num_k_heads, self.head_k_dim))?; // [K_heads, k_dim]
        let k_heads = k.reshape((self.num_k_heads, self.head_k_dim))?; // [K_heads, k_dim]
        let v_heads = v.reshape((self.num_v_heads, self.head_v_dim))?; // [V_heads, v_dim]

        // L2 normalize q and k per head
        let q_heads = l2_normalize(&q_heads)?;
        let k_heads = l2_normalize(&k_heads)?;

        // Expand k from [K_heads, k_dim] to [V_heads, k_dim] (repeat for grouped heads)
        let k_expanded = if kv_ratio > 1 {
            // Each k_head is shared by kv_ratio v_heads
            let k_rep: Vec<Tensor> = (0..self.num_k_heads)
                .flat_map(|kh| std::iter::repeat_n(kh, kv_ratio))
                .map(|kh| k_heads.get(kh))
                .collect::<Result<Vec<_>>>()?;
            Tensor::stack(&k_rep, 0)?
        } else {
            k_heads.clone()
        };
        // Same for q
        let q_expanded = if kv_ratio > 1 {
            let q_rep: Vec<Tensor> = (0..self.num_k_heads)
                .flat_map(|kh| std::iter::repeat_n(kh, kv_ratio))
                .map(|kh| q_heads.get(kh))
                .collect::<Result<Vec<_>>>()?;
            Tensor::stack(&q_rep, 0)?
        } else {
            q_heads.clone()
        };

        // Compute gating
        // g = -exp(A_log) * softplus(a + dt_bias)
        let a_plus_dt = (a + &self.dt_bias)?;
        let g = (self.a_log.exp()?.neg()? * softplus(&a_plus_dt)?)?;
        // beta = sigmoid(b)
        let beta = ops.sigmoid(b)?;

        // Initialize recurrent state if needed. Shape (V, K) matches
        // `DeltaNetPool`'s `[slots, HV, V, K]` layout, so the pooled varlen
        // writer in this file can scatter `recurrent_state` into a pool row
        // without transposing.
        if self.recurrent_state.is_none() {
            self.recurrent_state = Some(Tensor::zeros(
                (self.num_v_heads, self.head_v_dim, self.head_k_dim),
                DType::F32,
                device,
            )?);
        }
        let state = self.recurrent_state.as_ref().unwrap();

        // Cast to f32 for numerical stability
        let k_f32 = k_expanded.to_dtype(DType::F32)?; // [V_heads, k_dim]
        let v_f32 = v_heads.to_dtype(DType::F32)?; // [V_heads, v_dim]
        let q_f32 = q_expanded.to_dtype(DType::F32)?; // [V_heads, k_dim]
        let g_f32 = g.to_dtype(DType::F32)?; // [V_heads]
        let beta_f32 = beta.to_dtype(DType::F32)?; // [V_heads]

        // Scale q by 1/sqrt(head_k_dim) after L2 norm
        let scale = (self.head_k_dim as f64).powf(-0.5);
        let q_f32 = (q_f32 * scale)?;

        // 1. Decay the state: state *= exp(g). state is [V_heads, v_dim, k_dim].
        let decay = g_f32.exp()?.reshape((self.num_v_heads, 1, 1))?;
        let state_decayed = state.broadcast_mul(&decay)?;

        // 2. Delta correction: v -= state @ k (prediction error)
        // prediction[h, v] = Σ_k state[h, v, k] * k[h, k]
        // state: [V_heads, v_dim, k_dim], k.unsqueeze(2): [V_heads, k_dim, 1]
        let prediction = state_decayed
            .matmul(&k_f32.unsqueeze(2)?)? // [V_heads, v_dim, 1]
            .squeeze(2)?; // [V_heads, v_dim]
        let v_residual = (v_f32 - prediction)?;

        // 3. Gate the residual: v_gated = v_residual * beta
        let v_gated = v_residual.broadcast_mul(&beta_f32.unsqueeze(1)?)?;

        // 4. Update state: state += outer(v_gated, k)
        // outer[h, v, k] = v_gated[h, v] * k[h, k]
        let outer = v_gated.unsqueeze(2)?.matmul(&k_f32.unsqueeze(1)?)?;
        let new_state = (state_decayed + outer)?;

        // 5. Output[h, v] = Σ_k new_state[h, v, k] * q[h, k]
        let output = new_state
            .matmul(&q_f32.unsqueeze(2)?)? // [V_heads, v_dim, 1]
            .squeeze(2)?; // [V_heads, v_dim]

        // Store updated state
        self.recurrent_state = Some(new_state);

        // Cast back and flatten
        output.to_dtype(v.dtype())?.reshape((self.value_dim,))
    }

    fn delta_rule_prefill_gdn(
        &mut self,
        mixed_qkv: &Tensor,
        b_in: &Tensor,
        a_in: &Tensor,
        device: &Device,
        ops: &dyn crate::ops::Ops,
    ) -> Result<Option<Tensor>> {
        let t = mixed_qkv.dim(0)?;
        let hk = self.num_k_heads;
        let hv = self.num_v_heads;
        let kdim = self.head_k_dim;
        if self.head_k_dim != self.head_v_dim || hv < hk {
            return Ok(None);
        }

        let dt_bias_f32 = self.dt_bias.to_dtype(DType::F32)?;
        let a_log_f32 = if self.a_log.dtype() == DType::F32 {
            self.a_log.clone()
        } else {
            self.a_log.to_dtype(DType::F32)?
        };
        let Some(prep) = ops.gdn_post_conv(
            mixed_qkv,
            a_in,
            b_in,
            &a_log_f32,
            &dt_bias_f32,
            hk,
            hv,
            kdim,
        ) else {
            return Ok(None);
        };
        let (q_bf16, k_bf16, v_bf16, alpha, beta) = prep?;

        let cu_seqlens = Tensor::from_vec(vec![0i64, t as i64], (2,), device)?;
        let initial_state = self.recurrent_state.take();
        let initial_state_4d = match initial_state.as_ref() {
            Some(s) => Some(s.unsqueeze(0)?.contiguous()?),
            None => None,
        };
        let scale = (kdim as f32).powf(-0.5);

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
            self.recurrent_state = initial_state;
            return Ok(None);
        };
        let (out, final_state) = result?;
        self.recurrent_state = Some(final_state.squeeze(0)?.contiguous()?);
        Ok(Some(out.reshape((t, self.value_dim))?))
    }

    /// Decode conv1d: single token update with state.
    fn conv1d_decode(&mut self, x: &Tensor, ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        let device = x.device();
        let dtype = x.dtype();
        let pad_len = self.conv_kernel - 1;

        // Initialize conv state if needed
        if self.conv_state.is_none() {
            self.conv_state = Some(Tensor::zeros((self.conv_dim, pad_len), dtype, device)?);
        }

        let x_bd = x.unsqueeze(0)?;
        let state_flat = self.conv_state.as_ref().unwrap();
        let state_fresh = (state_flat + 0.0f64)?.contiguous()?;
        let state_bd = state_fresh.unsqueeze(0)?.contiguous()?;
        if let Some(res) = ops.causal_conv1d_update(
            &x_bd,
            &state_bd,
            &self.conv_weight,
            None,
            /*silu_activation=*/ false,
            None,
        ) {
            let out_bd = res?;
            self.conv_state = Some(state_bd.squeeze(0)?.contiguous()?);
            return out_bd.squeeze(0);
        }
        let state = self.conv_state.as_ref().unwrap();

        // Build full kernel window: [state | x] = [conv_dim, kernel_size]
        let x_col = x.unsqueeze(1)?; // [conv_dim, 1]
        let full_window = Tensor::cat(&[state, &x_col], 1)?; // [conv_dim, kernel_size]

        // Dot product with conv weights (no bias)
        let output = (full_window * &self.conv_weight)?.sum(1)?;

        // Update state: shift left, append x
        let new_state = if self.conv_kernel > 2 {
            Tensor::cat(&[state.narrow(1, 1, self.conv_kernel - 2)?, x_col], 1)?
        } else {
            x_col // kernel=2, state is just [conv_dim, 1]
        };
        self.conv_state = Some(new_state);

        Ok(output) // [conv_dim]
    }

    /// Prefill conv1d: causal convolution over full sequence.
    fn conv1d_prefill(&mut self, qkv: &Tensor, ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        let (b, seq_len, _) = qkv.dims3()?;
        let device = qkv.device();
        let dtype = qkv.dtype();

        // Transpose to [B, conv_dim, L] for conv1d
        let qkv_t = qkv.transpose(1, 2)?;

        // Left-pad with conv_kernel-1 zeros (or conv state if available)
        let pad = if let Some(state) = &self.conv_state {
            state
                .unsqueeze(0)?
                .expand((b, self.conv_dim, self.conv_kernel - 1))?
        } else {
            Tensor::zeros((b, self.conv_dim, self.conv_kernel - 1), dtype, device)?
        };
        let padded = Tensor::cat(&[pad, qkv_t.contiguous()?], 2)?;

        if let Some(res) = ops.causal_conv1d_fn(
            &padded,
            &self.conv_weight,
            None,
            None,
            /*silu_activation=*/ false,
        ) {
            let output_t = res?;
            let last_tokens = qkv_t.narrow(
                2,
                seq_len.saturating_sub(self.conv_kernel - 1),
                (self.conv_kernel - 1).min(seq_len),
            )?;
            self.conv_state = Some(last_tokens.squeeze(0)?.contiguous()?);
            return output_t
                .narrow(2, self.conv_kernel - 1, seq_len)?
                .transpose(1, 2);
        }

        // Depthwise conv1d: weight [conv_dim, 1, kernel], groups=conv_dim
        // Cast to F32 for CPU conv1d (candle CPU doesn't support BF16 matmul in conv1d)
        let conv_weight_3d = self.conv_weight.unsqueeze(1)?; // [conv_dim, 1, kernel]
        let output_t = if padded.device().is_cpu() && padded.dtype() == DType::BF16 {
            let padded_f32 = padded.to_dtype(DType::F32)?;
            let conv_f32 = conv_weight_3d.to_dtype(DType::F32)?;
            padded_f32
                .conv1d(&conv_f32, 0, 1, 1, self.conv_dim)?
                .to_dtype(DType::BF16)?
        } else {
            padded.conv1d(&conv_weight_3d, 0, 1, 1, self.conv_dim)?
        };

        // Save last kernel-1 tokens as conv state for next call
        let last_tokens = qkv_t.narrow(
            2,
            seq_len.saturating_sub(self.conv_kernel - 1),
            (self.conv_kernel - 1).min(seq_len),
        )?;
        self.conv_state = Some(last_tokens.squeeze(0)?.contiguous()?);

        // Transpose back to [B, L, conv_dim]
        output_t.transpose(1, 2)
    }
}

/// L2 normalize along the last dimension: x / sqrt(sum(x^2) + eps)
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
    x.broadcast_div(&norm)
}

/// Softplus: log(1 + exp(x)) with numerical stability.
fn softplus(x: &Tensor) -> Result<Tensor> {
    // For large x: softplus(x) ≈ x. For small x: log(1 + exp(x)).
    // Use threshold of 20 to avoid overflow.
    let safe = x.clamp(f64::NEG_INFINITY, 20.0)?;
    let sp = (safe.exp()? + 1.0)?.log()?;
    // Where x >= 20, use x directly (sp would overflow); else use sp
    let mask = x.ge(20.0f64)?.to_dtype(x.dtype())?;
    let inv_mask = (mask.affine(-1.0, 1.0))?; // 1 - mask
    mask.broadcast_mul(x)?
        .broadcast_add(&inv_mask.broadcast_mul(&sp)?)
}

// ── Gated Attention (Standard Softmax) ──────────────────────────────────

struct Qwen3NextAttention {
    q_proj: Linear, // output is 2x width: Q + gate
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    rope: PartialRotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    softmax_scale: f64,
}

impl Qwen3NextAttention {
    fn new(cfg: &Qwen3NextConfig, rope: PartialRotaryEmbedding, vb: VarBuilder) -> Result<Self> {
        let q_proj_dim = cfg.num_attention_heads * cfg.head_dim * 2; // 2x for gate
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

        // Qwen3-Next uses residual RMSNorm: output = norm(x) * (1 + weight)
        // Safetensors stores residual weight (near 0), so we add 1.0 at load time.
        let q_norm_weight = (vb.pp("q_norm").get(cfg.head_dim, "weight")? + 1.0)?;

        let k_norm_weight = (vb.pp("k_norm").get(cfg.head_dim, "weight")? + 1.0)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm_weight,
            k_norm_weight,
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            softmax_scale: 1.0 / (cfg.head_dim as f64).sqrt(),
        })
    }

    /// Varlen forward for prefill (GPU flash-attn or CPU fallback via ops dispatch).
    fn forward(&mut self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
        let total_tokens = x.dim(0)?;
        let bs = BatchState::no_lora();

        let q_raw = self.q_proj.forward(x, &bs, ctx.ops)?;
        let k = self.k_proj.forward(x, &bs, ctx.ops)?;
        let v = self.v_proj.forward(x, &bs, ctx.ops)?;

        let q_and_gate = q_raw.reshape((total_tokens, self.num_heads, self.head_dim * 2))?;
        let q = q_and_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_and_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((total_tokens, self.num_heads * self.head_dim))?;

        let k = k.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;

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

        let (q, k) = self.rope.apply_varlen(&q, &k, ctx.position_ids)?;
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
        let gate = ctx.ops.sigmoid(&gate)?;
        self.o_proj.forward(&(attn_output * gate)?, &bs, ctx.ops)
    }
}

// ── Expert MLP ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct ExpertMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl ExpertMlp {
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

// ── Sparse MoE Block with Shared Expert ─────────────────────────────────

struct Qwen3NextSparseMoeBlock {
    gate: DenseLinear,
    experts: Option<Vec<ExpertMlp>>,
    shared_expert: ExpertMlp,
    shared_expert_gate: DenseLinear,
    // Stacked weights for fused MoE GEMM (GPU only). `experts_gate_up`
    // is `[E, 2*inter, hidden]` in `[up|gate]` order for FlashInfer CUTLASS.
    experts_gate_up: Option<Tensor>,
    experts_down: Option<Tensor>,
    moe_intermediate_size: usize,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    num_experts: usize,
}

impl Qwen3NextSparseMoeBlock {
    fn new(cfg: &Qwen3NextConfig, vb: VarBuilder) -> Result<Self> {
        let gate = {
            let gvb = vb.pp("gate");
            let w = gvb.get((cfg.num_experts, cfg.hidden_size), "weight")?;
            DenseLinear::new(w, None)
        };

        let mut experts = Vec::with_capacity(cfg.num_experts);
        let vb_e = vb.pp("experts");
        for idx in 0..cfg.num_experts {
            experts.push(ExpertMlp::new(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                vb_e.pp(idx),
            )?);
        }

        let shared_expert = ExpertMlp::new(
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            vb.pp("shared_expert"),
        )?;
        let shared_expert_gate = {
            let gvb = vb.pp("shared_expert_gate");
            let w = gvb.get((1, cfg.hidden_size), "weight")?;
            DenseLinear::new(w, None)
        };

        // Stack expert weights for fused GEMM (GPU only). Drop the per-expert
        // Linear fallback after packing; keeping both copies OOMs 80B-class
        // checkpoints on single-card deployments.
        let is_cuda = experts
            .first()
            .map_or(false, |e| e.gate_proj.weight().device().is_cuda());
        let (experts_gate_up, experts_down, experts) = if is_cuda {
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
            (
                Some(Tensor::cat(&[&up_w, &gate_w], 1)?.contiguous()?),
                Some(down_w),
                None,
            )
        } else {
            (None, None, Some(experts))
        };

        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            experts_gate_up,
            experts_down,
            moe_intermediate_size: cfg.moe_intermediate_size,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
            num_experts: cfg.num_experts,
        })
    }

    fn forward(&self, ops: &dyn crate::ops::Ops, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        let hidden_dim = *dims
            .last()
            .ok_or_else(|| crate::tensor::Error::Msg("Qwen3-Next MoE got scalar input".into()))?;
        let num_tokens = xs.elem_count() / hidden_dim;
        let xs_2d = xs.reshape((num_tokens, hidden_dim))?;

        // Shared expert (always active)
        let shared_out = self.shared_expert.forward(ops, &xs_2d)?;
        let shared_gate_logit = xs_2d.apply(&self.shared_expert_gate)?;
        let shared_gate = ops.sigmoid(&shared_gate_logit)?;
        let shared_contribution = shared_out.broadcast_mul(&shared_gate)?;

        // Router: topk expert selection
        let router_logits = xs_2d.apply(&self.gate)?;
        let last_dim = router_logits.rank() - 1;
        let routing_weights = ops.softmax(&router_logits, last_dim)?;

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

        // Routed expert computation
        if xs.device().is_cuda() && self.experts_gate_up.is_some() {
            let routed = self.forward_cutlass_or_grouped(
                ops,
                &xs_2d,
                &topk_weights,
                &experts_per_tok,
                num_tokens,
                hidden_dim,
            )?;
            let out = (routed + shared_contribution)?;
            return if dims.len() == 3 {
                out.reshape((dims[0], dims[1], hidden_dim))
            } else {
                Ok(out)
            };
        }

        let routed =
            self.forward_sequential(ops, &xs_2d, &topk_weights, &experts_per_tok, hidden_dim)?;

        let out = (routed + shared_contribution)?;
        if dims.len() == 3 {
            out.reshape((dims[0], dims[1], hidden_dim))
        } else {
            Ok(out)
        }
    }

    fn forward_cutlass_or_grouped(
        &self,
        ops: &dyn crate::ops::Ops,
        xs: &Tensor,
        topk_weights: &Tensor,
        experts_per_tok: &Tensor,
        num_tokens: usize,
        hidden_dim: usize,
    ) -> Result<Tensor> {
        let experts_gate_up = self.experts_gate_up.as_ref().unwrap();
        let experts_down = self.experts_down.as_ref().unwrap();

        if let Some(result) = ops.cutlass_fused_moe(
            xs,
            experts_per_tok,
            topk_weights,
            experts_gate_up,
            experts_down,
        ) {
            match result {
                Ok(out) => return Ok(out),
                Err(error) => {
                    tracing::debug!(
                        %error,
                        "Qwen3-Next CUTLASS fused MoE failed; falling back to grouped MoE"
                    );
                }
            }
        }

        let (sorted_expert_ids, sorted_token_ids) =
            sort_expert_assignments(experts_per_tok, xs.device(), self.num_experts)?;

        let is_prefill = num_tokens > 1;
        let counts_sentinel = &sorted_expert_ids;

        let gate_up = ops.grouped_gemm(
            xs,
            experts_gate_up,
            &sorted_token_ids,
            &sorted_expert_ids,
            counts_sentinel,
        )?;
        let inter = self.moe_intermediate_size;
        let down_input = {
            let up = gate_up.narrow(D::Minus1, 0, inter)?.contiguous()?;
            let gate = gate_up.narrow(D::Minus1, inter, inter)?.contiguous()?;
            ops.silu_mul(&gate, &up)?
        };

        // Last step: GEMM + topk weight application (fused when available)
        let ys = match ops.fused_moe_gemm(
            &down_input,
            experts_down,
            topk_weights,
            &sorted_token_ids,
            &sorted_expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        ) {
            Some(r) => r?,
            None => {
                // Fallback: grouped_gemm + manual weight apply
                let raw = ops.grouped_gemm(
                    &down_input,
                    experts_down,
                    &sorted_token_ids,
                    &sorted_expert_ids,
                    counts_sentinel,
                )?;
                let raw = raw.reshape((num_tokens, self.num_experts_per_tok, hidden_dim))?;
                let w = topk_weights.unsqueeze(D::Minus1)?;
                return (raw * w)?.sum(D::Minus2);
            }
        };

        ys.reshape((num_tokens, self.num_experts_per_tok, hidden_dim))?
            .sum(D::Minus2)
    }

    fn forward_sequential(
        &self,
        ops: &dyn crate::ops::Ops,
        xs: &Tensor,
        topk_weights: &Tensor,
        experts_per_tok: &Tensor,
        _hidden_dim: usize,
    ) -> Result<Tensor> {
        let routing_weights = topk_weights.to_vec2::<f32>()?;
        let experts_per_tok_cpu = experts_per_tok.to_vec2::<u32>()?;
        let experts = self.experts.as_ref().ok_or_else(|| {
            crate::tensor::Error::Msg(
                "Qwen3-Next sequential MoE fallback unavailable after CUDA fused weight packing"
                    .into(),
            )
        })?;

        let mut top_x: Vec<Vec<u32>> = vec![vec![]; self.num_experts];
        for (i, row) in experts_per_tok_cpu.iter().enumerate() {
            for &expert_id in row {
                top_x[expert_id as usize].push(i as u32);
            }
        }

        let mut ys = Tensor::zeros_like(xs)?;
        for (expert_id, token_ids) in top_x.iter().enumerate() {
            if token_ids.is_empty() {
                continue;
            }
            let idx = Tensor::from_vec(token_ids.clone(), (token_ids.len(),), xs.device())?;
            let x_subset = xs.index_select(&idx, 0)?;
            let expert_out = experts[expert_id].forward(ops, &x_subset)?;

            // Gather routing weights for this expert
            let weights: Vec<f32> = token_ids
                .iter()
                .map(|&tid| {
                    let row = &experts_per_tok_cpu[tid as usize];
                    let col = row.iter().position(|&e| e == expert_id as u32).unwrap();
                    routing_weights[tid as usize][col]
                })
                .collect();
            let w = Tensor::from_vec(weights, (token_ids.len(), 1), xs.device())?
                .to_dtype(xs.dtype())?;
            let weighted = expert_out.broadcast_mul(&w)?;
            ys = ys.index_add(&idx, &weighted, 0)?;
        }
        Ok(ys)
    }
}

fn sort_expert_assignments(
    experts_per_tok: &Tensor,
    device: &Device,
    _num_experts: usize,
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

// ── Decoder Layer ───────────────────────────────────────────────────────

enum TokenMixer {
    LinearAttention(Qwen3NextGatedDeltaNet),
    FullAttention(Qwen3NextAttention),
}

struct Qwen3NextDecoderLayer {
    token_mixer: TokenMixer,
    moe: Qwen3NextSparseMoeBlock,
    ln1_weight: Tensor,
    ln2_weight: Tensor,
    rms_norm_eps: f32,
}

impl Qwen3NextDecoderLayer {
    fn new(
        cfg: &Qwen3NextConfig,
        layer_idx: usize,
        rope: PartialRotaryEmbedding,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Qwen3-Next uses residual RMSNorm: output = norm(x) * (1 + weight)
        let ln1_weight = (vb.pp("input_layernorm").get(cfg.hidden_size, "weight")? + 1.0)?;
        let ln2_weight = (vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")?
            + 1.0)?;

        let token_mixer = match cfg.layer_type(layer_idx) {
            LayerType::LinearAttention => {
                TokenMixer::LinearAttention(Qwen3NextGatedDeltaNet::new(cfg, vb.pp("linear_attn"))?)
            }
            LayerType::FullAttention => {
                TokenMixer::FullAttention(Qwen3NextAttention::new(cfg, rope, vb.pp("self_attn"))?)
            }
        };

        let moe = Qwen3NextSparseMoeBlock::new(cfg, vb.pp("mlp"))?;

        Ok(Self {
            token_mixer,
            moe,
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
        ops.add_or_fused(&x_res, &self.moe.forward(ops, &h2)?)
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
        ops.add_or_fused(&x_res, &self.moe.forward(ops, &h2)?)
    }

    fn clear_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::LinearAttention(gdn) => gdn.clear_state(),
            TokenMixer::FullAttention(_) => {}
        }
    }
}

/// Free function to run DeltaNet on packed varlen input (avoids borrow conflicts).
fn deltanet_varlen(
    gdn: &mut Qwen3NextGatedDeltaNet,
    packed: &Tensor,
    seq_lens: &[usize],
    ops: &dyn crate::ops::Ops,
) -> Result<Tensor> {
    let mut outputs = Vec::new();
    let mut offset = 0usize;
    for &len in seq_lens {
        let seq = packed.narrow(0, offset, len)?.unsqueeze(0)?; // [1, L, D]
        let out = gdn.forward(&seq, 0, ops)?; // [1, L, D]
        outputs.push(out.squeeze(0)?); // [L, D]
        offset += len;
    }
    Tensor::cat(&outputs, 0) // [total_tokens, D]
}

fn deltanet_decode_batched_fused(
    gdn: &mut Qwen3NextGatedDeltaNet,
    packed: &Tensor,
    pool: &mut crate::deltanet_pool::DeltaNetPool,
    slot_ids: &[u32],
    slot_ids_gpu: Option<&Tensor>,
    dn_layer_idx: usize,
    ops: &dyn crate::ops::Ops,
) -> Result<Option<Tensor>> {
    if !packed.device().is_cuda() {
        return Ok(None);
    }
    let n = slot_ids.len();
    if n == 0 {
        return Ok(None);
    }

    let bs = BatchState::no_lora();
    let x = packed.unsqueeze(0)?;
    let (qkv, z, b_raw, a_raw) = gdn.project_qkvz_ba(&x, &bs, ops)?;
    let qkv_2d = qkv.squeeze(0)?.contiguous()?;

    let slot_ids_owned: Option<Tensor>;
    let slot_ids_gpu = match slot_ids_gpu {
        Some(t) => t,
        None => {
            slot_ids_owned = Some(Tensor::from_vec(slot_ids.to_vec(), (n,), packed.device())?);
            slot_ids_owned.as_ref().unwrap()
        }
    };

    let pool_conv = &pool.conv_states[dn_layer_idx];
    let qkv_conv = match ops.causal_conv1d_update(
        &qkv_2d,
        pool_conv,
        &gdn.conv_weight,
        None,
        /*silu_activation=*/ true,
        Some(slot_ids_gpu),
    ) {
        Some(r) => r?,
        None => return Ok(None),
    };

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

    let a_2d = a_raw.squeeze(0)?.contiguous()?;
    let b_2d = b_raw.squeeze(0)?.contiguous()?;
    let a_log_f32 = if gdn.a_log.dtype() == DType::F32 {
        gdn.a_log.clone()
    } else {
        gdn.a_log.to_dtype(DType::F32)?
    };
    let decode_out = match ops.kda_decode_batched(
        &q_nhk,
        &k_nhk,
        &v_nhv,
        &a_2d,
        &b_2d,
        &a_log_f32,
        &gdn.dt_bias,
        &pool.recurrent_states[dn_layer_idx],
        slot_ids_gpu,
    ) {
        Some(Ok(o)) => o,
        Some(Err(e)) => return Err(e),
        None => return Ok(None),
    };

    let out = decode_out.reshape((n, gdn.value_dim))?.unsqueeze(0)?;
    let normed = gdn.norm.forward(&out, &z.contiguous()?, ops)?;
    let projected = gdn.out_proj.forward(&normed, &bs, ops)?;
    Ok(Some(projected.squeeze(0)?))
}

/// Pooled varlen DeltaNet: process each sequence independently, scatter final state to pool.
fn deltanet_varlen_pooled(
    gdn: &mut Qwen3NextGatedDeltaNet,
    packed: &Tensor,
    seq_lens: &[usize],
    pool: &mut crate::deltanet_pool::DeltaNetPool,
    slot_ids: &[u32],
    slot_ids_gpu: Option<&Tensor>,
    dn_layer_idx: usize,
    ops: &dyn crate::ops::Ops,
) -> Result<Tensor> {
    let all_decode = !seq_lens.is_empty() && seq_lens.iter().all(|&l| l == 1);
    let fused_supported = matches!(
        (
            gdn.num_k_heads,
            gdn.num_v_heads,
            gdn.head_k_dim,
            gdn.head_v_dim,
        ),
        (16, 32, 128, 128)
    );
    if all_decode && packed.device().is_cuda() && fused_supported && ops.kda_decode_available() {
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
        if let Some(out) =
            deltanet_mixed_grouped_fused(gdn, packed, seq_lens, pool, slot_ids, dn_layer_idx, ops)?
        {
            return Ok(out);
        }
    }

    let mut outputs = Vec::new();
    let mut offset = 0usize;
    for (i, &len) in seq_lens.iter().enumerate() {
        let slot = slot_ids[i] as usize;
        gdn.recurrent_state = Some(
            pool.recurrent_states[dn_layer_idx]
                .get(slot)?
                .contiguous()?,
        );
        gdn.conv_state = Some(pool.conv_states[dn_layer_idx].get(slot)?.contiguous()?);

        let seq = packed.narrow(0, offset, len)?.unsqueeze(0)?; // [1, L, D]
        let out = gdn.forward(&seq, 0, ops)?; // [1, L, D]
        outputs.push(out.squeeze(0)?); // [L, D]

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
    gdn.clear_state();
    Tensor::cat(&outputs, 0) // [total_tokens, D]
}

fn deltanet_mixed_grouped_fused(
    gdn: &mut Qwen3NextGatedDeltaNet,
    packed: &Tensor,
    seq_lens: &[usize],
    pool: &mut crate::deltanet_pool::DeltaNetPool,
    slot_ids: &[u32],
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
    if !decode_indices.is_empty() && !ops.kda_decode_available() {
        return Ok(None);
    }

    let mut outputs: Vec<Option<Tensor>> = (0..seq_lens.len()).map(|_| None).collect();
    const MAX_GROUPED_PREFILL_BATCH: usize = 8;
    for (&seq_len, indices) in prefill_by_len.iter() {
        let mut start = 0usize;
        while start < indices.len() {
            let remaining = indices.len() - start;
            let take = if remaining == 1 {
                1
            } else if remaining <= MAX_GROUPED_PREFILL_BATCH {
                remaining
            } else if remaining - MAX_GROUPED_PREFILL_BATCH == 1 {
                MAX_GROUPED_PREFILL_BATCH - 1
            } else {
                MAX_GROUPED_PREFILL_BATCH
            };
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
            let out = deltanet_prefill_group_fused(
                gdn,
                packed,
                &offsets,
                seq_len,
                chunk,
                pool,
                slot_ids,
                dn_layer_idx,
                ops,
            )?;
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

#[allow(clippy::too_many_arguments)]
fn deltanet_single_pooled(
    gdn: &mut Qwen3NextGatedDeltaNet,
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

#[allow(clippy::too_many_arguments)]
fn deltanet_prefill_group_fused(
    gdn: &mut Qwen3NextGatedDeltaNet,
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

    let bs = BatchState::no_lora();
    let mut seqs = Vec::with_capacity(batch);
    for &req_idx in indices {
        seqs.push(packed.narrow(0, offsets[req_idx], seq_len)?);
    }
    let x = Tensor::stack(&seqs, 0)?.contiguous()?;
    let (qkv, z, b_raw, a_raw) = gdn.project_qkvz_ba(&x, &bs, ops)?;

    let pad_len = gdn.conv_kernel - 1;
    let x_t = qkv.transpose(1, 2)?.contiguous()?;
    let pool_conv = &pool.conv_states[dn_layer_idx];
    let mut prefixes = Vec::with_capacity(batch);
    for &req_idx in indices {
        prefixes.push(pool_conv.get(slot_ids[req_idx] as usize)?.contiguous()?);
    }
    let prefix = Tensor::stack(&prefixes, 0)?.contiguous()?;
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
    let qkv_conv = conv?
        .narrow(2, pad_len, seq_len)?
        .transpose(1, 2)?
        .contiguous()?;

    let total_tokens = batch * seq_len;
    let mixed = qkv_conv.reshape((total_tokens, gdn.conv_dim))?;
    let a_2d = a_raw.reshape((total_tokens, gdn.num_v_heads))?;
    let b_2d = b_raw.reshape((total_tokens, gdn.num_v_heads))?;
    let dt_bias_f32 = gdn.dt_bias.to_dtype(DType::F32)?;
    let a_log_f32 = if gdn.a_log.dtype() == DType::F32 {
        gdn.a_log.clone()
    } else {
        gdn.a_log.to_dtype(DType::F32)?
    };

    let Some(prep) = ops.gdn_post_conv(
        &mixed,
        &a_2d,
        &b_2d,
        &a_log_f32,
        &dt_bias_f32,
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
    let projected = gdn.out_proj.forward(&normed, &bs, ops)?;
    Ok(Some(projected))
}

// ── Model ───────────────────────────────────────────────────────────────

struct Qwen3NextModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen3NextDecoderLayer>,
    norm_weight: Tensor,
    rms_norm_eps: f64,
}

impl Qwen3NextModel {
    fn new(cfg: &Qwen3NextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = {
            let emb_vb = vb_m.pp("embed_tokens");
            let weight = emb_vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            Embedding::new(weight, cfg.hidden_size)
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for idx in 0..cfg.num_hidden_layers {
            let rope = PartialRotaryEmbedding::new(cfg, vb.dtype(), vb.device())?;
            layers.push(Qwen3NextDecoderLayer::new(cfg, idx, rope, vb_l.pp(idx))?);
        }

        // Qwen3-Next uses residual RMSNorm: output = norm(x) * (1 + weight)
        let norm_weight = (vb_m.pp("norm").get(cfg.hidden_size, "weight")? + 1.0)?;

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
        let mut attn_layer_idx = 0usize;
        let mut dn_layer_idx = 0usize;
        for layer in self.layers.iter_mut() {
            match &layer.token_mixer {
                TokenMixer::FullAttention(_) => {
                    let layer_kv = ctx.paged_kv.map(|kv| kv.layer(attn_layer_idx));
                    let layer_ctx = LayerAttnContext {
                        ops: ctx.ops,
                        cu_seqlens_q: ctx.cu_seqlens_q,
                        max_seqlen_q: ctx.max_seqlen_q,
                        position_ids: ctx.position_ids,
                        paged_kv: layer_kv.as_ref(),
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
                            ctx.cu_seqlens_q,
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_q,
                            ctx.position_ids,
                            seq_lens,
                            pool,
                            slots,
                            ctx.deltanet_slots_gpu,
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
        ctx.ops
            .rms_norm(&h, &self.norm_weight, self.rms_norm_eps as f32)
    }

    fn clear_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }
}

// ── ForCausalLM ─────────────────────────────────────────────────────────

pub struct Qwen3NextForCausalLM {
    model: Qwen3NextModel,
    lm_head: Linear,
}

impl Qwen3NextForCausalLM {
    pub fn new(cfg: &Qwen3NextConfig, vb: VarBuilder) -> Result<Self> {
        let model = Qwen3NextModel::new(cfg, vb.clone())?;
        let lm_head = Linear::load(vb.pp("lm_head"), cfg.hidden_size, cfg.vocab_size, false)?;
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
}

impl crate::models::LogitsSplitModel for Qwen3NextForCausalLM {
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

impl crate::models::ModelForward for Qwen3NextForCausalLM {
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

mod meta {
    use crate::loading::var_builder::VarBuilder;

    use super::{Qwen3NextConfig, Qwen3NextForCausalLM};
    use crate::cache::deltanet_pool::DeltaNetPoolConfig;
    use crate::engine::EngineError;
    use crate::engine::{CommonModelConfig, RuntimeCaps, TaskKind, WeightsBackend};
    use crate::models::registry::{candle_model_err, parse_json, ArchSpec, ParsedModelConfig};

    const ARCHITECTURE_ALIASES: &[&str] = &["Qwen3Next"];
    const MODEL_TYPE_ALIASES: &[&str] = &["qwen3_next"];
    const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate];

    fn deltanet_config_from(cfg: &Qwen3NextConfig) -> DeltaNetPoolConfig {
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

    pub(crate) struct Qwen3NextArchSpec;

    pub(crate) static QWEN3_NEXT_ARCH_SPEC: Qwen3NextArchSpec = Qwen3NextArchSpec;
    inventory::submit!(crate::models::registry::ArchSpecEntry::new(
        &QWEN3_NEXT_ARCH_SPEC
    ));

    impl ArchSpec for Qwen3NextArchSpec {
        fn name(&self) -> &'static str {
            "qwen3_next"
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
            let cfg = parse_json::<Qwen3NextConfig>(content, "Qwen3-Next config")?;
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
            let cfg = arch_config
                .downcast_ref::<Qwen3NextConfig>()
                .ok_or_else(|| {
                    EngineError::Internal("unexpected arch config type for Qwen3-Next".into())
                })?;
            Ok(Box::new(
                Qwen3NextForCausalLM::new(cfg, vb).map_err(candle_model_err)?,
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
    }
}
