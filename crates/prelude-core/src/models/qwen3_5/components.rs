use crate::loading::var_builder::VarBuilder;
use crate::tensor::{D, DType, Device, Result, Tensor};

use super::Qwen3_5Config;

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
    pub(super) fn apply_varlen(
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
    pub(super) fn forward(
        &self,
        x: &Tensor,
        z: &Tensor,
        ops: &dyn crate::ops::Ops,
    ) -> Result<Tensor> {
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
