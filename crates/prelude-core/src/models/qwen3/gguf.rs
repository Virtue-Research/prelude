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
        _ctx: &mut crate::models::common::BatchAttnContext,
    ) -> Result<Tensor> {
        crate::tensor::bail!("GGUF model does not support varlen forward")
    }
}

impl crate::models::ModelForward for Qwen3GgufModel {
    fn forward(
        &mut self,
        _packed_input: &Tensor,
        _ctx: &mut crate::models::common::BatchAttnContext,
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
