//! Quantized Qwen3.5 model loaded from GGUF format.
//!
//! Uses GGML quantized matmul kernels (AVX-512/AMX) when compiled with `ggml-quants` feature,
//! falls back to candle's `QMatMul` otherwise.
//!
//! Reference: llama.cpp `src/models/qwen35.cpp` + `src/models/delta-net-base.cpp`.

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result, Tensor};
use std::io::{Read, Seek};
use std::sync::Arc;

use crate::constants::GGUF_INTERMEDIATE_SIZE_MULTIPLIER;
use crate::models::common::{Linear, RmsNorm, TransformerBlock};
use crate::nn_ops::Embedding;

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

// ── Model ───────────────────────────────────────────────────────────────

pub struct Qwen3_5GgufModel {
    inner: super::Qwen3_5ForCausalLM,
}

impl Qwen3_5GgufModel {
    fn load_linear<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<Linear> {
        let qtensor = ct.tensor(reader, name, device)?;
        Linear::from_qtensor(Arc::new(qtensor))
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

        // Embedding (dequantize for lookup table)
        let embed_weight = Self::load_tensor(&ct, reader, "token_embd.weight", device)?;
        let embed_tokens = Embedding::new(embed_weight, config.hidden_size);

        // Build PartialRotaryEmbedding from GGUF config
        let build_rope = |dtype: DType| -> Result<super::PartialRotaryEmbedding> {
            let rotary_dim = config.rotary_dim();
            let inv_freq: Vec<f32> = (0..rotary_dim)
                .step_by(2)
                .map(|i| 1.0 / config.rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
                .collect();
            let inv_freq = Tensor::new(inv_freq, device)?;
            let positions =
                Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
                    .to_dtype(DType::F32)?;
            let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
            let cos = freqs.cos()?.to_dtype(dtype)?;
            let sin = freqs.sin()?.to_dtype(dtype)?;
            Ok(super::PartialRotaryEmbedding {
                cos,
                sin,
                rotary_dim,
            })
        };

        // Build layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let dtype = DType::F32;

        for i in 0..config.num_hidden_layers {
            let prefix = format!("blk.{i}");

            // Layer norms (GGUF already has +1 applied for residual RMSNorm)
            let ln1_weight =
                Self::load_tensor(&ct, reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ln1 = RmsNorm::from_weight(ln1_weight.clone(), config.rms_norm_eps);

            let ln2_weight = Self::load_tensor(
                &ct,
                reader,
                &format!("{prefix}.post_attention_norm.weight"),
                device,
            )?;
            let ln2 = RmsNorm::from_weight(ln2_weight.clone(), config.rms_norm_eps);

            let block =
                TransformerBlock::new(ln1, ln1_weight, ln2, ln2_weight, config.rms_norm_eps, i);

            // Token mixer
            let token_mixer = if config.is_recurrent(i) {
                // DeltaNet layer
                let in_proj_qkv = Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.attn_qkv.weight"),
                    device,
                )?;
                let in_proj_z = Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.attn_gate.weight"),
                    device,
                )?;
                let in_proj_b = Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.ssm_beta.weight"),
                    device,
                )?;
                let in_proj_a = Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.ssm_alpha.weight"),
                    device,
                )?;

                // Conv1d weight: GGUF stores [d_conv, conv_channels], transpose if needed
                let conv_weight_raw = Self::load_tensor(
                    &ct,
                    reader,
                    &format!("{prefix}.ssm_conv1d.weight"),
                    device,
                )?;
                let conv_weight =
                    if conv_weight_raw.dim(0)? == config.linear_conv_kernel_dim {
                        conv_weight_raw.t()?.contiguous()?
                    } else {
                        conv_weight_raw
                    };

                let dt_bias = Self::load_tensor(
                    &ct,
                    reader,
                    &format!("{prefix}.ssm_dt.bias"),
                    device,
                )?;

                // GGUF stores -exp(A_log); convert back: A_log = ln(-ssm_a)
                let ssm_a =
                    Self::load_tensor(&ct, reader, &format!("{prefix}.ssm_a"), device)?;
                let a_log = ssm_a.neg()?.log()?;

                // Gated RMSNorm (NO +1 for ssm_norm)
                let norm_weight = Self::load_tensor(
                    &ct,
                    reader,
                    &format!("{prefix}.ssm_norm.weight"),
                    device,
                )?;
                let norm = super::RmsNormGated {
                    weight: norm_weight,
                    eps: config.rms_norm_eps,
                    num_heads: config.linear_num_value_heads,
                    head_dim: config.linear_value_head_dim,
                };

                let out_proj = Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.ssm_out.weight"),
                    device,
                )?;

                let key_dim = config.key_dim();
                let value_dim = config.value_dim();
                let conv_dim = config.conv_dim();

                super::TokenMixer::LinearAttention(super::Qwen3_5GatedDeltaNet {
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
                    key_dim,
                    value_dim,
                    conv_dim,
                    conv_kernel: config.linear_conv_kernel_dim,
                })
            } else {
                // Full attention layer
                let q_proj = Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.attn_q.weight"),
                    device,
                )?;
                let k_proj = Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.attn_k.weight"),
                    device,
                )?;
                let v_proj = Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.attn_v.weight"),
                    device,
                )?;
                let o_proj = Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.attn_output.weight"),
                    device,
                )?;

                // QK norms (GGUF already has +1 applied)
                let q_norm_weight = Self::load_tensor(
                    &ct,
                    reader,
                    &format!("{prefix}.attn_q_norm.weight"),
                    device,
                )?;
                let k_norm_weight = Self::load_tensor(
                    &ct,
                    reader,
                    &format!("{prefix}.attn_k_norm.weight"),
                    device,
                )?;

                let q_norm =
                    RmsNorm::from_weight(q_norm_weight.clone(), config.rms_norm_eps);
                let k_norm =
                    RmsNorm::from_weight(k_norm_weight.clone(), config.rms_norm_eps);

                let rope = build_rope(dtype)?;

                super::TokenMixer::FullAttention(super::Qwen3_5Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    q_norm_weight,
                    k_norm_weight,
                    rope,
                    kv_cache: None,
                    k_cache: Vec::new(),
                    v_cache: Vec::new(),
                    num_heads: config.num_attention_heads,
                    num_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    rms_norm_eps: config.rms_norm_eps,
                    softmax_scale: 1.0 / (config.head_dim as f64).sqrt(),
                    attn_output_gate: true,
                })
            };

            // MLP (same for both layer types)
            let mlp = super::MlpVariant::Dense(super::Qwen3_5Mlp {
                gate_proj: Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.ffn_gate.weight"),
                    device,
                )?,
                up_proj: Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.ffn_up.weight"),
                    device,
                )?,
                down_proj: Self::load_linear(
                    &ct,
                    reader,
                    &format!("{prefix}.ffn_down.weight"),
                    device,
                )?,
            });

            layers.push(super::Qwen3_5DecoderLayer {
                token_mixer,
                mlp,
                block,
            });
        }

        // Final norm (GGUF already has +1 applied)
        let norm_weight = Self::load_tensor(&ct, reader, "output_norm.weight", device)?;
        let norm = RmsNorm::from_weight(norm_weight.clone(), config.rms_norm_eps);

        // LM head
        let lm_head = if ct.tensor_infos.get("output.weight").is_some() {
            Self::load_linear(&ct, reader, "output.weight", device)?
        } else {
            // Tied embeddings: use token_embd.weight
            Self::load_linear(&ct, reader, "token_embd.weight", device)?
        };

        let model = super::Qwen3_5Model {
            embed_tokens,
            layers,
            norm,
            norm_weight,
            rms_norm_eps: config.rms_norm_eps,
        };

        let inner = super::Qwen3_5ForCausalLM { model, lm_head };

        Ok((Self { inner }, config))
    }

    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
        self.inner.forward_with_cache(input_ids, position_offset)
    }

    pub fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut crate::models::common::BatchAttnContext,
    ) -> Result<Tensor> {
        self.inner.forward(packed_input, ctx)
    }
}

impl crate::models::KvCacheModel for Qwen3_5GgufModel {
    fn forward_with_cache(
        &mut self,
        input_ids: &Tensor,
        position_offset: usize,
    ) -> Result<Tensor> {
        Qwen3_5GgufModel::forward_with_cache(self, input_ids, position_offset)
    }
}

impl crate::models::ModelForward for Qwen3_5GgufModel {
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut crate::models::common::BatchAttnContext,
    ) -> Result<Tensor> {
        self.inner.forward(packed_input, ctx)
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache();
    }

    fn as_kv_cache_model(&mut self) -> Option<&mut dyn crate::models::KvCacheModel> {
        Some(self)
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

    /// Generate tokens via llama.cpp's C-side decode loop (zero Rust<>C overhead per token).
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
impl crate::models::KvCacheModel for LlamaGgufModel {
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

    fn as_kv_cache_model(&mut self) -> Option<&mut dyn crate::models::KvCacheModel> {
        Some(self)
    }
}
