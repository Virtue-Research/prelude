# Prelude Development Guide

## Adding a New Model Architecture

This guide walks through adding support for a new model family (e.g., Llama, Mistral).

### Step 1: Create Architecture Module

Create a new directory and module file:

```
crates/prelude-core/src/models/architectures/
├── llama/
│   └── mod.rs    ← New file
└── mod.rs        ← Update this
```

### Step 2: Implement Model Structure

In `llama/mod.rs`:

```rust
use candle_core::{Device, DType, Tensor, Result};
use candle_nn::VarBuilder;

// Model configuration from config.json
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
}

// Attention layer
struct LlamaAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    // ... RoPE, KV cache
}

// Decoder layer
struct LlamaDecoderLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

// Full model
pub struct LlamaModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
    lm_head: candle_nn::Linear,
    config: LlamaConfig,
    device: Device,
    dtype: DType,
}
```

### Step 3: Implement Forward Pass

```rust
impl LlamaModel {
    pub fn new(config: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        // Load weights from VarBuilder
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens")
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new(config, vb.pp(format!("model.layers.{i}")))?);
        }

        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = candle_nn::linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self { embed_tokens, layers, norm, lm_head, config, device, dtype })
    }

    pub fn forward(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;
        let mut hidden = self.embed_tokens.forward(input_ids)?;

        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, index_pos)?;
        }

        let hidden = self.norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;

        Ok(logits)
    }

    // Varlen forward for batched inference
    pub fn forward_varlen(
        &mut self,
        packed_input: &Tensor,
        cu_seqlens: &Tensor,
        max_seqlen: usize,
        position_ids: &Tensor,
        seq_lens: &[usize],
    ) -> Result<Tensor> {
        // Similar to forward but uses flash_attn_varlen
        // ...
    }
}
```

### Step 4: Register in ModelVariant

Update `crates/prelude-core/src/candle_engine.rs`:

```rust
// Add to ModelVariant enum
pub enum ModelVariant {
    Qwen3(Qwen3Model),
    Qwen3MoE(Qwen3MoeModel),
    Gemma3(Gemma3Model),
    Llama(LlamaModel),  // ← Add new variant
}

// Add pattern matching in all trait methods
impl ModelVariant {
    pub fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        match self {
            Self::Qwen3(m) => m.forward(input, index_pos),
            Self::Qwen3MoE(m) => m.forward(input, index_pos),
            Self::Gemma3(m) => m.forward(input, index_pos),
            Self::Llama(m) => m.forward(input, index_pos),  // ← Add
        }
    }
    // ... repeat for other methods
}
```

### Step 5: Add Model Loading

Update `load_model()` in `candle_engine.rs`:

```rust
fn load_model(
    model_id: &str,
    config_content: &str,
    vb: VarBuilder,
    device: &Device,
    dtype: DType,
) -> Result<(ModelVariant, ModelConfig)> {
    let config: serde_json::Value = serde_json::from_str(config_content)?;

    let architectures = config["architectures"].as_array();

    if let Some(archs) = architectures {
        let arch_name = archs[0].as_str().unwrap_or("");

        match arch_name {
            "Qwen3ForCausalLM" => { /* existing */ },
            "LlamaForCausalLM" => {
                let llama_config: LlamaConfig = serde_json::from_str(config_content)?;
                let model = LlamaModel::new(&llama_config, vb)?;
                return Ok((ModelVariant::Llama(model), ModelConfig::Text));
            },
            _ => { /* fallback */ }
        }
    }
    // ...
}
```

### Step 6: Export in mod.rs

Update `crates/prelude-core/src/models/architectures/mod.rs`:

```rust
pub mod qwen3;
pub mod qwen3_moe;
pub mod gemma3;
pub mod llama;  // ← Add

pub use llama::LlamaModel;  // ← Export
```

---

## Adding a New Endpoint

### Step 1: Define Types

In `crates/prelude-core/src/types.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankerResult {
    pub model: String,
    pub results: Vec<RerankScore>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankScore {
    pub index: usize,
    pub score: f32,
}
```

### Step 2: Add Engine Method

In `crates/prelude-core/src/engine.rs`:

```rust
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    // ... existing methods

    async fn rerank(&self, request: RerankerRequest) -> Result<RerankerResult, EngineError>;
}
```

### Step 3: Implement in CandleEngine

In `crates/prelude-core/src/candle_engine.rs`:

```rust
impl CandleEngine {
    pub fn rerank_batch(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        // Tokenize query + each document
        // Run classifier forward
        // Extract scores
    }
}
```

### Step 4: Implement in ScheduledEngine

In `crates/prelude-core/src/scheduled_engine.rs`:

```rust
impl InferenceEngine for ScheduledEngine {
    async fn rerank(&self, request: RerankerRequest) -> Result<RerankerResult, EngineError> {
        // Queue request
        // Wait for batch
        // Return result
    }
}
```

### Step 5: Add HTTP Handler

In `crates/prelude-server/src/main.rs`:

```rust
async fn rerank(
    State(state): State<AppState>,
    Json(request): Json<RerankerRequest>,
) -> Result<Json<RerankerResponse>, ApiError> {
    let result = state.engine.rerank(request).await?;
    // Format response
    Ok(Json(response))
}

// Add route
.route("/v1/rerank", post(rerank))
```

---

## Adding a New CPU Kernel

New CPU kernels should be added to the pure Rust `cpu_ops/` module. For GEMM operations, use the oneDNN FFI layer.

### Option A: Pure Rust kernel (cpu_ops/)

Add your kernel implementation in `crates/prelude-core/src/cpu_ops/`:

```rust
// my_kernel.rs
use std::arch::x86_64::*;

/// AVX-512 optimized kernel with scalar fallback
pub fn my_custom_kernel_bf16(input: &[u16], output: &mut [u16]) {
    if is_x86_feature_detected!("avx512f") {
        unsafe { my_custom_kernel_avx512(input, output) }
    } else {
        my_custom_kernel_scalar(input, output)
    }
}
```

### Option B: oneDNN FFI kernel

For operations that benefit from oneDNN primitives, add to the FFI layer:

1. C++ wrapper in `crates/onednn-ffi/src/onednn_ffi.cpp`
2. Header declaration in `crates/onednn-ffi/include/onednn_ffi.h`
3. Rust FFI bindings in `crates/prelude-core/src/onednn_ffi.rs`
4. Safe wrapper in `crates/prelude-core/src/onednn_ops.rs`

---

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_forward() {
        let device = Device::Cpu;
        let config = LlamaConfig { /* ... */ };
        // ...
    }
}
```

### Integration Tests

```bash
# Start server
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server --model Qwen/Qwen3-4B

# Run benchmark
python benchmark/benchmark_simple.py --mode generate --requests 100
```

### Accuracy Tests

```python
# Compare outputs with reference implementation
import torch
from transformers import AutoModelForCausalLM

# Load reference
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")

# Compare outputs
# ...
```

---

## Debugging Tips

### Enable Debug Logging

```bash
RUST_LOG=debug ./target/release/prelude-server
```

### Profile GPU Usage

```bash
# Monitor GPU
watch -n 0.5 nvidia-smi

# Profile with nsys
nsys profile ./target/release/prelude-server
```

### Common Issues

1. **Shape Mismatch**
   - Check tensor dimensions at each layer
   - Verify config values match weights

2. **NaN/Inf in Output**
   - Check dtype conversions (bf16 ↔ f32)
   - Verify softmax/layernorm numerics

3. **Memory Issues**
   - Check KV cache allocation
   - Monitor with `nvidia-smi`

4. **Performance Issues**
   - Enable Flash Attention
   - Use varlen batching
   - Check batch sizes

---

## Code Style

- Use `rustfmt` for formatting
- Prefer explicit error handling over `.unwrap()`
- Add doc comments for public APIs
- Follow existing naming conventions
