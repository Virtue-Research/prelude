//! Drop-in replacements for the candle-nn and candle-transformers items used in prelude-core.
//!
//! This module allows us to remove the `candle-nn` and `candle-transformers` crate
//! dependencies while keeping identical public APIs for:
//!
//! - [`Embedding`] — wraps a weight tensor, forward does index_select on dim 0
//! - [`CandleLinear`] — reimplements `candle_nn::Linear` (weight + optional bias matmul)
//! - [`Activation`] — enum dispatching to common activation functions
//! - [`ops`] — `softmax`, `softmax_last_dim`, `sigmoid`, `silu`, `log_softmax`
//! - [`rotary_emb`] — `rope`, `rope_thd`
//! - [`moe`] — `moe_gemm` (CUDA-only, stub on CPU)
//! - [`generation`] — re-exports `LogitsProcessor`, `Sampling` from `engine::sampling`
//! - [`Qwen3Config`] — the Config struct from `candle_transformers::models::qwen3`
//! - [`RmsNorm`] — wraps a weight tensor + eps, same semantics as `candle_nn::RmsNorm`

use crate::tensor::{DType, Module, Result, Tensor, D};

// ═══════════════════════════════════════════════════════════════════════
// Embedding
// ═══════════════════════════════════════════════════════════════════════

/// Wraps a weight tensor of shape `(vocab_size, hidden_size)`.
/// `forward` performs `index_select` on dimension 0.
#[derive(Clone, Debug)]
pub struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl Module for Embedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

/// Load an embedding layer from a `VarBuilder` (reads `"weight"` tensor).
pub fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let weight = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(weight, hidden_size))
}

// ═══════════════════════════════════════════════════════════════════════
// Linear  (candle_nn::Linear replacement)
// ═══════════════════════════════════════════════════════════════════════

/// A linear layer: `y = x @ weight.T + bias`.
///
/// This replaces `candle_nn::Linear`. The project's own `crate::modules::Linear`
/// wraps *this* struct (or the oneDNN variant) and should be preferred in model code.
#[derive(Clone, Debug)]
pub struct CandleLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl CandleLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for CandleLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = match *x.dims() {
            [b1, b2, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((b1 * b2 * m, k))?
                        .matmul(&w)?
                        .reshape((b1, b2, m, ()))?
                } else {
                    let w = self.weight.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&w)?
                }
            }
            [bsize, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((bsize * m, k))?
                        .matmul(&w)?
                        .reshape((bsize, m, ()))?
                } else {
                    let w = self.weight.broadcast_left(bsize)?.t()?;
                    x.matmul(&w)?
                }
            }
            _ => {
                let w = self.weight.t()?;
                x.matmul(&w)?
            }
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

/// Create a linear layer **with** bias from a `VarBuilder`.
pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<CandleLinear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    let bs = vb.get(out_dim, "bias")?;
    Ok(CandleLinear::new(ws, Some(bs)))
}

/// Create a linear layer **without** bias from a `VarBuilder`.
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<CandleLinear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    Ok(CandleLinear::new(ws, None))
}

/// Create a linear layer, optionally with bias, from a `VarBuilder`.
pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
) -> Result<CandleLinear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Activation
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    Silu,
    Sigmoid,
    HardSigmoid,
    Swiglu,
    Swish,
    Mish,
    HardSwish,
    Elu(f64),
    LeakyRelu(f64),
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu_erf(),
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Relu2 => xs.relu()?.sqr(),
            Self::Relu6 => xs.clamp(0f32, 6f32),
            Self::Silu => xs.silu(),
            Self::Sigmoid => ops::sigmoid(xs),
            Self::HardSigmoid => ops::hard_sigmoid(xs),
            Self::Swiglu => ops::swiglu(xs),
            Self::Swish => xs * ops::sigmoid(xs)?,
            Self::HardSwish => xs * ops::hard_sigmoid(xs)?,
            Self::Mish => ops::mish(xs),
            &Self::Elu(alpha) => xs.elu(alpha),
            &Self::LeakyRelu(negative_slope) => ops::leaky_relu(xs, negative_slope),
            Self::GeluPytorchTanh => xs.gelu(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RmsNorm  (candle_nn::RmsNorm replacement)
// ═══════════════════════════════════════════════════════════════════════

/// RMS normalization layer.
///
/// Stores a weight vector and epsilon. Uses the `candle_nn::ops::rms_norm`-equivalent
/// custom op for contiguous inputs, and a fallback for non-contiguous.
#[derive(Clone, Debug)]
pub struct CandleRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl CandleRmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for CandleRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Slow but universally correct path:
        // x_norm = x * rsqrt(mean(x^2) + eps) * weight
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let eps_t =
            Tensor::new(&[self.eps as f32], xs.device())?.broadcast_as(variance.shape())?;
        let inv_rms = (variance + eps_t)?.sqrt()?.recip()?;
        let normed = xs_f32.broadcast_mul(&inv_rms)?;
        let weight = self.weight.to_dtype(DType::F32)?;
        let result = normed.broadcast_mul(&weight)?;
        result.to_dtype(xs.dtype())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Ops
// ═══════════════════════════════════════════════════════════════════════

pub mod ops {
    use crate::tensor::{Result, Tensor, D};

    /// Softmax along an arbitrary dimension.
    pub fn softmax<Dim: crate::tensor::shape::Dim>(xs: &Tensor, dim: Dim) -> Result<Tensor> {
        let dim = dim.to_index(xs.shape(), "softmax")?;
        let max = xs.max_keepdim(dim)?;
        let diff = xs.broadcast_sub(&max)?;
        let num = diff.exp()?;
        let den = num.sum_keepdim(dim)?;
        num.broadcast_div(&den)
    }

    /// Optimised softmax along the **last** dimension.
    ///
    /// Uses the same `SoftmaxLastDim` custom-op (CPU + CUDA + Metal kernels)
    /// that `candle_nn::ops::softmax_last_dim` dispatches to. When that op is
    /// unavailable (shouldn't happen with candle-core) we fall back to the
    /// generic `softmax` implementation.
    pub fn softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
        // candle_core exposes the custom-op through Tensor helpers, but the
        // actual SoftmaxLastDim struct is private to candle-nn.  We reimplement
        // with the portable path which is still correct (and reasonably fast on
        // CPU thanks to the max/exp/sum vectorisation inside candle-core).
        softmax(xs, D::Minus1)
    }

    /// Log-softmax along a given dimension.
    pub fn log_softmax<Dim: crate::tensor::shape::Dim>(xs: &Tensor, d: Dim) -> Result<Tensor> {
        let d = d.to_index(xs.shape(), "log-softmax")?;
        let max = xs.max_keepdim(d)?;
        let diff = xs.broadcast_sub(&max)?;
        let sum_exp = diff.exp()?.sum_keepdim(d)?;
        let log_sm = diff.broadcast_sub(&sum_exp.log()?)?;
        Ok(log_sm)
    }

    /// Sigmoid: `1 / (1 + exp(-x))`.
    ///
    /// The original candle-nn version registers a `CustomOp1` with specialised
    /// CUDA/Metal kernels.  Our replacement uses the portable formula which
    /// candle-core can still auto-dispatch to CUDA via its built-in unary ops.
    pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
        // 1 / (1 + exp(-x))
        let neg = xs.neg()?;
        let exp_neg = neg.exp()?;
        let one_plus = (exp_neg + 1.0)?;
        one_plus.recip()
    }

    /// SiLU (Sigmoid Linear Unit): `x * sigmoid(x)`.
    pub fn silu(xs: &Tensor) -> Result<Tensor> {
        xs.silu()
    }

    /// Hard-sigmoid: `clamp((x + 3) / 6, 0, 1)`.
    pub fn hard_sigmoid(xs: &Tensor) -> Result<Tensor> {
        ((xs + 3.0)? / 6.0)?.clamp(0f32, 1f32)
    }

    /// Mish: `x * tanh(softplus(x))` = `x * tanh(ln(1 + exp(x)))`.
    pub fn mish(xs: &Tensor) -> Result<Tensor> {
        xs * (1.0 + xs.exp()?)?.log()?.tanh()
    }

    /// Leaky ReLU.
    pub fn leaky_relu(xs: &Tensor, negative_slope: f64) -> Result<Tensor> {
        let zeros = xs.zeros_like()?;
        xs.maximum(&zeros)? + xs.minimum(&zeros)? * negative_slope
    }

    /// SwiGLU: split last dim in half, silu(first) * second.
    pub fn swiglu(xs: &Tensor) -> Result<Tensor> {
        let chunks = xs.chunk(2, D::Minus1)?;
        &chunks[0].silu()? * &chunks[1]
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Rotary Embeddings
// ═══════════════════════════════════════════════════════════════════════

pub mod rotary_emb {
    use crate::tensor::{Result, Tensor, D};

    /// Rotate-half helper: splits the last dim in two halves and returns
    /// `cat([-x2, x1], dim=-1)`.
    fn rotate_half(xs: &Tensor) -> Result<Tensor> {
        let last_dim = xs.dim(D::Minus1)?;
        let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
        let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
        Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
    }

    /// Standard RoPE for tensors shaped `(B, H, T, D)` with cos/sin shaped
    /// `(T, D/2)` or `(B, T, D/2)`.
    ///
    /// This is the "slow" (but portable) path.  The candle-nn version uses a
    /// custom op with hand-written CUDA/CPU kernels for speed; we fall back to
    /// the mathematical definition which candle-core can still dispatch.
    pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (_b_sz, _h, seq_len, _n_embd) = x.dims4()?;
        let cos = Tensor::cat(&[cos, cos], D::Minus1)?;
        let sin = Tensor::cat(&[sin, sin], D::Minus1)?;
        let cos = cos.narrow(0, 0, seq_len)?;
        let sin = sin.narrow(0, 0, seq_len)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
    }

    /// T-H-D contiguous variant of RoPE for tensors shaped `(B, T, H, D)`.
    ///
    /// cos/sin are `(T, D/2)` or `(B, T, D/2)`.
    pub fn rope_thd(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (_b_sz, seq_len, _n_head, _n_embd) = x.dims4()?;
        let cos = Tensor::cat(&[cos, cos], D::Minus1)?;
        let sin = Tensor::cat(&[sin, sin], D::Minus1)?;
        let cos = cos.narrow(0, 0, seq_len)?;
        let sin = sin.narrow(0, 0, seq_len)?;
        // cos/sin: (T, D) → (1, T, 1, D)
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;
        x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MoE GEMM (CUDA kernel wrapper)
// ═══════════════════════════════════════════════════════════════════════

pub mod moe {
    use crate::tensor::{Result, Tensor};

    #[cfg(feature = "cuda")]
    pub fn moe_gemm(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        use crate::tensor::backend::cuda_backend::kernels::ffi;
        use crate::tensor::backend::cuda_backend::cudarc::driver::DevicePtr;
        use crate::tensor::DType;
        use half::{bf16, f16};

        fn cuda_fwd<
            T: crate::tensor::backend::cuda_backend::CudaDType
                + crate::tensor::backend::cuda_backend::cudarc::driver::DeviceRepr,
        >(
            input: &Tensor,
            weights: &Tensor,
            topk_weights: &Option<Tensor>,
            sorted_token_ids: &Tensor,
            experts_ids: &Tensor,
            topk: usize,
            is_prefill: bool,
        ) -> Result<Tensor> {
            let (mut size_m, size_k1) = input.dims2()?;
            if topk_weights.is_none() {
                size_m *= topk;
            }
            let (num_experts, size_n, size_k) = weights.dims3()?;
            assert!(
                size_k == size_k1,
                "input {:?} and weight {:?} last dim mismatch!",
                size_k1,
                size_k
            );
            let dev = input.device().as_cuda_device()?;
            let data_type = match input.dtype() {
                DType::F16 => 0,
                DType::BF16 => 1,
                _ => {
                    crate::tensor::bail!("moe_gemm_wmma only accepts f16/bf16 inputs")
                }
            };

            let (input, _) = input.storage_and_layout();
            let input = match &*input {
                crate::tensor::backend::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => crate::tensor::bail!("input must be a cuda tensor"),
            };

            let (weights, _) = weights.storage_and_layout();
            let weights = match &*weights {
                crate::tensor::backend::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => crate::tensor::bail!("weight must be a cuda tensor"),
            };

            let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
            let sorted_token_ids = match &*sorted_token_ids {
                crate::tensor::backend::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => crate::tensor::bail!("sorted_token_ids must be a cuda tensor"),
            };

            let (experts_ids, _) = experts_ids.storage_and_layout();
            let experts_ids = match &*experts_ids {
                crate::tensor::backend::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => crate::tensor::bail!("experts_ids must be a cuda tensor"),
            };

            let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
                let (topk_weights, _) = topk_weights.storage_and_layout();
                let topk_weights = match &*topk_weights {
                    crate::tensor::backend::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => crate::tensor::bail!("topk_weights must be a cuda tensor"),
                };
                topk_weights.device_ptr(topk_weights.stream()).0 as *const f32
            } else {
                std::ptr::null()
            };

            let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;
            let expert_counts = unsafe { dev.alloc::<u32>(num_experts) }?;
            let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;

            let stream = dev.cuda_stream().cu_stream() as i64;
            use core::ffi::c_void;

            unsafe {
                ffi::moe_gemm_wmma(
                    input.device_ptr(input.stream()).0 as *const c_void,
                    weights.device_ptr(weights.stream()).0 as *const c_void,
                    sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
                    experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
                    topk_weights_ptr,
                    output.device_ptr(output.stream()).0 as *mut c_void,
                    expert_counts.device_ptr(expert_counts.stream()).0 as *mut i32,
                    expert_offsets.device_ptr(expert_offsets.stream()).0 as *mut i32,
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    data_type as i32,
                    is_prefill,
                    stream,
                );
            }

            use crate::tensor::backend::BackpropOp;
            let output = crate::tensor::backend::CudaStorage::wrap_cuda_slice(output, dev.clone());
            let output = Tensor::from_storage(
                crate::tensor::backend::Storage::Cuda(output),
                (size_m, size_n),
                BackpropOp::none(),
                false,
            );

            Ok(output)
        }

        match input.dtype() {
            DType::F16 => cuda_fwd::<f16>(
                input,
                weights,
                topk_weights,
                sorted_token_ids,
                experts_ids,
                topk,
                is_prefill,
            ),
            DType::BF16 => cuda_fwd::<bf16>(
                input,
                weights,
                topk_weights,
                sorted_token_ids,
                experts_ids,
                topk,
                is_prefill,
            ),
            _ => {
                crate::tensor::bail!("moe_gemm only accepts f16/bf16 inputs")
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn moe_gemm(
        _: &Tensor,
        _: &Tensor,
        _: &Option<Tensor>,
        _: &Tensor,
        _: &Tensor,
        _: usize,
        _: bool,
    ) -> Result<Tensor> {
        crate::tensor::bail!("moe_gemm is only implemented for the cuda backend")
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Generation — re-export from engine/sampling/ (canonical location)
// ═══════════════════════════════════════════════════════════════════════

pub mod generation {
    pub use crate::engine::sampling::{LogitsProcessor, Sampling};
}

// ═══════════════════════════════════════════════════════════════════════
// Qwen3Config  (from candle_transformers::models::qwen3::Config)
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

// ═══════════════════════════════════════════════════════════════════════
// VarBuilder  (re-export from candle_nn — cannot replace, it's a complex
//              trait hierarchy; kept as a thin re-export)
// ═══════════════════════════════════════════════════════════════════════

// VarBuilder: now a standalone, inference-only copy in crate::loading::var_builder.
// Re-export it here so call-sites can do `use crate::nn_ops::VarBuilder`.
pub use crate::loading::var_builder::VarBuilder;
