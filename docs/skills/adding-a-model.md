# Adding a New Model Architecture

This guide walks through adding a new model architecture to Prelude.

## Before You Start

**Prerequisites:**
- Familiarity with Rust and the transformer architecture
- Access to the model's `config.json` (from HuggingFace or local)
- Understanding of the model's attention mechanism (standard, GQA, sliding window, linear recurrence)

**Recommended approach:** If your model is architecturally similar to an existing one (e.g., a LLaMA variant), start by copying `qwen3/` and modifying. Most transformer-based LLMs share 90%+ of the same structure.

**Reference implementations:**
- `qwen3/` -- dense model (generate + classify + embed)
- `gemma3/` -- model with sliding window + bidirectional attention
- `qwen3_moe/` -- Mixture-of-Experts with fused GEMM
- `qwen3_next/` -- hybrid (DeltaNet + attention + MoE)

## Overview

Adding a model requires **4 steps**:

1. Create `models/<name>/mod.rs` — config struct, model struct, `ModelForward` impl
2. Create `models/<name>/meta.rs` — `ArchSpec` impl with static instance
3. Add `pub mod <name>;` in `models/mod.rs`
4. Add one line in `ALL_ARCH_SPECS` in `models/registry.rs`

No enum variants, no match arms, no macro changes — just implement two traits and register.

## File Structure

```
crates/prelude-core/src/models/
  mymodel/
    mod.rs    # Config struct, model struct, forward logic, ModelForward impl
    meta.rs   # ArchSpec impl, static registration
```

## Step 1: Config Struct

Your config struct deserializes from the model's `config.json`:

```rust
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MyModelConfig {
    // --- needed for CommonModelConfig (extracted in meta.rs) ---
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    // --- architecture-specific ---
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    // ...
}

fn default_rms_norm_eps() -> f64 { 1e-6 }
```

The 5 fields (`vocab_size`, `num_hidden_layers`, `max_position_embeddings`, `num_key_value_heads`,
`head_dim`) are extracted into `CommonModelConfig` by your `parse_config()` implementation.

## Step 2: Model Struct + Forward

Build the model layers and implement the forward pass:

```rust
pub struct MyModelForCausalLM {
    embed_tokens: prelude_core::modules::embedding::Embedding,
    layers: Vec<MyDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    kv_cache: Vec<Option<(Tensor, Tensor)>>,
    // ...
}

impl MyModelForCausalLM {
    pub fn new(cfg: &MyModelConfig, vb: VarBuilder<'_>) -> prelude_core::tensor::Result<Self> {
        // load weights from vb
    }

    pub fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> prelude_core::tensor::Result<Tensor> {
        // embed -> layers -> norm -> lm_head
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.iter_mut().for_each(|c| *c = None);
    }
}
```

## Step 3: Implement `ModelForward`

Only 2 methods are required. Override optional methods as needed.

**Generation model (with KV cache):**

```rust
use crate::models::ModelForward;

impl ModelForward for MyModelForCausalLM {
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> prelude_core::tensor::Result<Tensor> {
        self.forward(packed_input, ctx)
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache();
    }

    // Override if your model supports KV cache operations:
    fn set_kv_cache_enabled(&mut self, enabled: bool) { /* ... */ }
    fn set_kv_cache_capacity(&mut self, target: usize) { /* ... */ }
    fn force_kv_cache_prealloc(&mut self, target: usize) { /* ... */ }
    fn inject_kv_cache(&mut self, layer_kvs: &[(Tensor, Tensor)]) -> prelude_core::tensor::Result<()> { /* ... */ }
    fn extract_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> { /* ... */ }
}
```

**Classifier model:**

```rust
impl ModelForward for MyModelForClassification {
    fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> prelude_core::tensor::Result<Tensor> {
        MyModelForClassification::forward(self, packed_input, ctx)
        // ^ use fully-qualified syntax to call inherent method, not trait method
    }

    fn clear_kv_cache(&mut self) {
        MyModelForClassification::clear_kv_cache(self);
    }

    fn is_classifier(&self) -> bool { true }

    fn classifier_info(&self) -> Option<(usize, Option<String>)> {
        Some((self.num_labels, self.get_label(0)))
    }

    fn get_label(&self, class_idx: usize) -> Option<String> {
        self.id2label.as_ref()?.get(&class_idx).cloned()
    }

    fn num_labels(&self) -> Option<usize> {
        Some(self.num_labels)
    }
}
```

**Embedding model:**

```rust
impl ModelForward for MyModelForEmbedding {
    fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> prelude_core::tensor::Result<Tensor> {
        self.forward(packed_input, ctx)
    }

    fn clear_kv_cache(&mut self) {}

    fn is_embedding(&self) -> bool { true }
    fn embedding_dim(&self) -> Option<usize> { Some(self.hidden_size) }
}
```

### ModelForward Method Reference

| Method | Required | Default | Override when |
|--------|----------|---------|---------------|
| `forward()` | Yes | — | Always |
| `clear_kv_cache()` | Yes | — | Always (no-op for embed/classify) |
| `set_kv_cache_enabled()` | No | no-op | Generation with KV cache |
| `set_kv_cache_capacity()` | No | no-op | Generation with KV cache |
| `force_kv_cache_prealloc()` | No | no-op | Generation with KV cache |
| `inject_kv_cache()` | No | Ok(()) | Generation with prefix cache |
| `extract_kv_cache()` | No | vec![] | Generation with prefix cache |
| `is_classifier()` | No | false | Classifier models |
| `is_embedding()` | No | false | Embedding models |
| `embedding_dim()` | No | None | Embedding models |
| `classifier_info()` | No | None | Classifier models |
| `get_label()` | No | None | Classifier models |
| `num_labels()` | No | None | Classifier models |

## Step 4: Implement `ArchSpec`

Create `meta.rs` to handle model discovery, config parsing, and construction:

```rust
use super::*;
use crate::engine::{CommonModelConfig, RuntimeCaps, TaskKind, WeightsBackend};
use crate::engine::EngineError;
use crate::models::registry::{
    parse_json, model_err, ArchSpec, ParsedModelConfig,
};

const ARCHITECTURE_ALIASES: &[&str] = &["MyModelForCausalLM"];
const MODEL_TYPE_ALIASES: &[&str] = &["mymodel"];
const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate];

pub(crate) struct MyModelArchSpec;
pub(crate) static MYMODEL_ARCH_SPEC: MyModelArchSpec = MyModelArchSpec;

impl ArchSpec for MyModelArchSpec {
    fn name(&self) -> &'static str { "mymodel" }

    fn architecture_aliases(&self) -> &'static [&'static str] { ARCHITECTURE_ALIASES }
    fn model_type_aliases(&self) -> &'static [&'static str] { MODEL_TYPE_ALIASES }
    fn supported_tasks(&self) -> &'static [TaskKind] { SUPPORTED_TASKS }

    fn parse_config(
        &self,
        _task: TaskKind,
        _raw: &serde_json::Value,
        content: &str,
    ) -> Result<ParsedModelConfig, EngineError> {
        let cfg: MyModelConfig = parse_json(content, "MyModel config")?;
        let common = CommonModelConfig {
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            max_position_embeddings: cfg.max_position_embeddings,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        };
        Ok(ParsedModelConfig {
            common,
            deltanet: None,
            arch_config: Box::new(cfg),
        })
    }

    fn build_model(
        &self,
        arch_config: &dyn std::any::Any,
        vb: VarBuilder<'_>,
    ) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
        let cfg = arch_config
            .downcast_ref::<MyModelConfig>()
            .ok_or_else(|| EngineError::Internal("unexpected config for MyModel".into()))?;
        Ok(Box::new(
            MyModelForCausalLM::new(cfg, vb).map_err(model_err)?,
        ))
    }

    fn runtime_caps(
        &self,
        task: TaskKind,
        backend: WeightsBackend,
        device: &prelude_core::tensor::Device,
    ) -> RuntimeCaps {
        let cuda_safetensors = device.is_cuda() && backend == WeightsBackend::Safetensors;
        RuntimeCaps {
            supports_kv_cache: task == TaskKind::Generate,
            supports_prefix_cache: false,
            supports_paged_attn: cfg!(feature = "paged-attn") && cuda_safetensors,
            supports_varlen: cfg!(feature = "flash-attn-v3") && cuda_safetensors,
            supports_deltanet: false,
            supports_cuda_graph: false,
            supports_varlen_cpu: false,
        }
    }
}
```

### Alias Matching

- `architecture_aliases` — matched against `config.json` → `"architectures": ["MyModelForCausalLM"]`
- `model_type_aliases` — matched against `config.json` → `"model_type": "mymodel"`

Either one matching is sufficient for auto-detection.

### Multi-Task Architectures

If one architecture supports multiple tasks (generate + classify + embed), list all tasks in
`SUPPORTED_TASKS` and use an internal enum for the arch config:

```rust
enum MyModelArchConfig {
    Dense(MyModelConfig),
    Classifier(MyModelClassifierConfig),
    Embedding(MyModelConfig),
}

fn parse_config(&self, task: TaskKind, raw: &serde_json::Value, content: &str)
    -> Result<ParsedModelConfig, EngineError>
{
    let cfg: MyModelConfig = parse_json(content, "MyModel config")?;
    let common = CommonModelConfig { /* ... extract from cfg ... */ };
    let arch_config: Box<dyn std::any::Any + Send> = match task {
        TaskKind::Classify => {
            let cls_cfg = parse_json(content, "MyModel classifier config")?;
            Box::new(MyModelArchConfig::Classifier(cls_cfg))
        }
        TaskKind::Embed => Box::new(MyModelArchConfig::Embedding(cfg)),
        _ => Box::new(MyModelArchConfig::Dense(cfg)),
    };
    Ok(ParsedModelConfig { common, deltanet: None, arch_config })
}

fn build_model(&self, arch_config: &dyn std::any::Any, vb: VarBuilder<'_>)
    -> Result<Box<dyn crate::models::ModelForward>, EngineError>
{
    let cfg = arch_config.downcast_ref::<MyModelArchConfig>()
        .ok_or_else(|| EngineError::Internal("unexpected config".into()))?;
    match cfg {
        MyModelArchConfig::Dense(c) => Ok(Box::new(MyModelForCausalLM::new(c, vb).map_err(model_err)?)),
        MyModelArchConfig::Classifier(c) => Ok(Box::new(MyModelForClassification::new(c, vb).map_err(model_err)?)),
        MyModelArchConfig::Embedding(c) => Ok(Box::new(MyModelForEmbedding::new(c, vb).map_err(model_err)?)),
    }
}
```

### DeltaNet Support

If your architecture uses DeltaNet layers, compute the `DeltaNetPoolConfig` in `parse_config()`:

```rust
use crate::cache::deltanet_pool::DeltaNetPoolConfig;

fn parse_config(&self, _task: TaskKind, _raw: &serde_json::Value, content: &str)
    -> Result<ParsedModelConfig, EngineError>
{
    let cfg: MyModelConfig = parse_json(content, "MyModel config")?;
    let common = CommonModelConfig { /* ... */ };
    let deltanet = Some(DeltaNetPoolConfig {
        num_deltanet_layers: /* ... */,
        num_v_heads: cfg.linear_num_value_heads,
        head_k_dim: cfg.linear_key_head_dim,
        head_v_dim: cfg.linear_value_head_dim,
        conv_dim: /* ... */,
        conv_kernel: cfg.linear_conv_kernel_dim,
    });
    Ok(ParsedModelConfig { common, deltanet, arch_config: Box::new(cfg) })
}
```

## Step 5: Wire into the Registry

Only **2 lines** needed in existing files:

### 5a. Export the module

In `crates/prelude-core/src/models/mod.rs`:

```rust
pub mod mymodel;   // <-- add
```

### 5b. Register in `all_arch_specs()`

In `crates/prelude-core/src/models/registry.rs`:

```rust
static ALL_ARCH_SPECS: &[&dyn ArchSpec] = &[
    &super::qwen3::meta::QWEN3_ARCH_SPEC,
    // ...existing...
    &super::mymodel::meta::MYMODEL_ARCH_SPEC,    // <-- add
];
```

No array size to bump — it's a static slice.

## Checklist

- [ ] Config struct with 5 common fields (`vocab_size`, `num_hidden_layers`, `max_position_embeddings`, `num_key_value_heads`, `head_dim`)
- [ ] Model struct with `new()` and `forward()`
- [ ] `impl ModelForward` with required + relevant optional methods
- [ ] `meta.rs` with `ArchSpec` impl and static instance
- [ ] Module exported in `models/mod.rs`
- [ ] Registered in `ALL_ARCH_SPECS` in `models/registry.rs`
- [ ] `cargo build` passes
- [ ] `cargo test` passes
- [ ] Server loads model and serves requests

## Testing Your Model

### 1. Build and smoke test

```bash
cargo build -p prelude-server --release --features flash-attn-v3
./target/release/prelude-server --model <your-model> --port 8001
curl http://localhost:8001/health
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<your-model>","prompt":"Hello","max_tokens":16}'
```

### 2. Run accuracy tests

Compare output against HuggingFace transformers reference:

```bash
python tests/accuracy/run_accuracy_test.py --variant cpu-f32 \
  --server prelude --binary target/release/prelude-server \
  --model <your-model>
```

This checks exact text match and logprob cosine similarity.

### 3. Test streaming

```bash
curl -sN http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<your-model>","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"stream":true}'
```

Verify tokens arrive incrementally and the stream ends with `data: [DONE]`.

### 4. Common failure modes

- **Weight name mismatch:** Model loads but outputs garbage. Check that your `VarBuilder` paths match the safetensor key names. Use `safetensors.torch.load_file()` in Python to inspect keys.
- **Missing config field:** `serde::Deserialize` fails. Add `#[serde(default = ...)]` for optional fields.
- **Wrong head_dim:** Attention produces NaN. Verify `head_dim = hidden_size / num_attention_heads` matches the model's config.
- **Shape mismatch in MLP:** Usually means `intermediate_size` is wrong. Check the model's config.json carefully.
- **Mock mode for fast iteration:** Use `--pseudo` to test API routing without loading model weights.
