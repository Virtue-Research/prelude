# Model Registry & Loading

## File Layout

```
prelude-core/src/
├── models/
│   ├── mod.rs                                 # trait Model, re-exports
│   ├── registry.rs                            # ModelRegistry: arch name → model constructor
│   ├── weight_loader.rs                       # WeightLoader: safetensors/GGUF → model struct
│   ├── config.rs                              # ModelConfig: parsed from HF config.json
│   ├── qwen3.rs                               # impl Model for Qwen3
│   ├── llama.rs                               # impl Model for Llama (also serves Phi3, InternLM3)
│   └── ...
```

## Model Registration

Auto-registration via `inventory` crate. Adding a new model = one file + one `inventory::submit!`.
No manual modification of registry.rs or mod.rs.

```rust
// prelude-core/src/models/registry.rs

/// Registration entry. Submitted once per model file via inventory::submit!
struct ModelRegistration {
    /// HF architecture names this model handles.
    /// Matched against config.json "architectures" field.
    architectures: &'static [&'static str],
    /// Constructor function.
    create: fn(&ModelConfig, &Ops) -> Result<Box<dyn Model>>,
}

inventory::collect!(ModelRegistration);

/// Lookup: HF arch name → constructor.
fn resolve_model(arch: &str) -> Option<&'static ModelRegistration> {
    inventory::iter::<ModelRegistration>()
        .find(|r| r.architectures.contains(&arch))
}
```

```rust
// prelude-core/src/models/llama.rs

struct LlamaModel { /* layers, embed, lm_head */ }

impl Model for LlamaModel {
    fn forward(&mut self, x: &Tensor, ctx: &ForwardCtx, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        // ...
    }
}

// One line: register this model for multiple HF architectures
inventory::submit! {
    ModelRegistration {
        architectures: &[
            "LlamaForCausalLM",
            "Phi3ForCausalLM",       // Phi3 = Llama architecture
            "InternLM3ForCausalLM",  // InternLM3 = Llama architecture
        ],
        create: |config, ops| {
            let model = LlamaModel::from_config(config, ops)?;
            Ok(Box::new(model))
        },
    }
}
```

**Design choice (learn from SGLang, avoid vLLM):**
- SGLang: `EntryClass` convention + `pkgutil.iter_modules()` auto-discovery. 133 lines of registry.
  Clean — adding a model doesn't touch the registry file.
- vLLM: 19K-line registry dict mapping 500+ architecture names. Manual, error-prone.
- **We use `inventory` crate** — Rust equivalent of SGLang's EntryClass.
  Each model file `submit!`s its own registration. Registry just iterates.
  Adding a model = add one file. Zero changes to existing code.

### Architecture Aliasing

Multiple HF architectures can map to the same model implementation:

```rust
// prelude-core/src/models/llama.rs — one file serves Llama, Phi3, InternLM3, Yi, etc.
// All are architecturally identical (RoPE + GQA + SiLU MLP).
// Config differences (num_layers, num_heads, vocab_size) handled by ModelConfig.

inventory::submit! {
    ModelRegistration {
        architectures: &[
            "LlamaForCausalLM",
            "Phi3ForCausalLM",
            "InternLM3ForCausalLM",
            "YiForCausalLM",
            "MistralForCausalLM",
        ],
        create: |config, ops| Ok(Box::new(LlamaModel::from_config(config, ops)?)),
    }
}
```

This is SGLang's pattern (Phi3 = LlamaForCausalLM via inheritance) adapted for Rust.
No inheritance needed — same struct, different config values.

## Model Config

```rust
// prelude-core/src/models/config.rs

/// Parsed from HF config.json. Architecture-agnostic fields.
struct ModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,

    // Quantization (if specified in config)
    pub quantization_config: Option<QuantizationConfig>,

    // MoE (optional)
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,

    // Model-specific extra fields (parsed as raw JSON)
    pub extra: serde_json::Value,
}

impl ModelConfig {
    /// Parse from HF config.json. Handles different HF naming conventions.
    fn from_hf_config(path: &Path) -> Result<Self> {
        let json: serde_json::Value = serde_json::from_reader(File::open(path)?)?;
        // Normalize: some models nest under "text_config" (Gemma3, multimodal)
        let config = if let Some(tc) = json.get("text_config") { tc } else { &json };
        // Extract fields with fallbacks for different naming conventions
        Ok(ModelConfig {
            hidden_size: config["hidden_size"].as_u64()? as usize,
            num_attention_heads: config["num_attention_heads"].as_u64()? as usize,
            num_key_value_heads: config.get("num_key_value_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(config["num_attention_heads"].as_u64()?) as usize,
            // ... etc ...
            extra: json.clone(),  // keep raw JSON for model-specific fields
        })
    }
}
```

**Design choice:**
- vLLM: `ModelArchitectureConfig` with per-model converter functions (`MODEL_ARCH_CONFIG_CONVERTORS`). Modular but verbose.
- SGLang: loads HF config directly, per-model `__init__` extracts fields. Simpler.
- **We use**: common `ModelConfig` struct with `extra: serde_json::Value` for model-specific fields.
  Common fields (hidden_size, num_layers, etc.) are normalized once. Models access
  `config.extra["sliding_window"]` for anything non-standard.

## Weight Loading

```rust
// prelude-core/src/models/weight_loader.rs

/// Load model weights from safetensors/GGUF files.
/// Maps HF tensor names to model struct fields.
trait WeightLoadable {
    /// Stacked parameter mappings: combine separate Q/K/V into fused QKV.
    fn stacked_params() -> &'static [StackedParam] {
        &[]  // default: no stacking
    }

    /// Load weights into self. Called once at model init.
    fn load_weights(&mut self, weights: &mut WeightIterator, ops: &Ops) -> Result<()>;
}

/// Stacked parameter mapping: combine multiple HF weights into one fused tensor.
struct StackedParam {
    /// Name in our model struct (e.g., "qkv_proj")
    target: &'static str,
    /// Names in HF safetensors (e.g., ["q_proj", "k_proj", "v_proj"])
    sources: &'static [&'static str],
    /// Shard IDs for each source (e.g., ["q", "k", "v"] or [0, 1])
    shard_ids: &'static [&'static str],
}

/// Iterator over (name, tensor) pairs from safetensors/GGUF files.
struct WeightIterator {
    // Handles format detection, shard discovery, lazy loading
}

impl WeightIterator {
    /// Open weight files at path. Auto-detects format.
    fn open(path: &Path) -> Result<Self> {
        if path.join("model.safetensors.index.json").exists() {
            // Sharded safetensors: read index → load relevant shards
            Self::open_safetensors_sharded(path)
        } else if path.extension() == Some("gguf") {
            Self::open_gguf(path)
        } else {
            // Single safetensors or .bin
            Self::open_single(path)
        }
    }
}
```

### Example: Llama Weight Loading

```rust
// prelude-core/src/models/llama.rs

impl WeightLoadable for LlamaModel {
    fn stacked_params() -> &'static [StackedParam] {
        &[
            StackedParam {
                target: "qkv_proj",
                sources: &["q_proj", "k_proj", "v_proj"],
                shard_ids: &["q", "k", "v"],
            },
            StackedParam {
                target: "gate_up_proj",
                sources: &["gate_proj", "up_proj"],
                shard_ids: &["0", "1"],
            },
        ]
    }

    fn load_weights(&mut self, weights: &mut WeightIterator, ops: &Ops) -> Result<()> {
        let stacked = Self::stacked_params();
        for (name, tensor) in weights {
            // 1. Try stacked params mapping
            if let Some(sp) = stacked.iter().find(|sp| sp.sources.iter().any(|s| name.contains(s))) {
                let target_name = name.replace(matched_source, sp.target);
                let param = self.get_param_mut(&target_name)?;
                param.load_shard(&tensor, shard_id)?;
                continue;
            }
            // 2. Direct load (name matches)
            if let Some(param) = self.get_param_mut(&name) {
                param.load(&tensor)?;
            }
        }
        Ok(())
    }
}
```

**Design choice (learn from SGLang):**
- SGLang: per-model `stacked_params_mapping` list. Explicit, clear, but duplicated across 10+ models.
- vLLM: recursive `AutoWeightsLoader` that traverses module tree. Generic but complex.
- **We use**: `stacked_params()` as a trait method with default empty impl.
  Models that need QKV fusion override it with a static list. Simple, no duplication
  if models share the same stacking pattern (they can call a shared constant).

## Weight Format Support

| Format | Detection | Loading |
|--------|-----------|---------|
| **Safetensors** (sharded) | `model.safetensors.index.json` exists | Read index → load relevant shards |
| **Safetensors** (single) | `model.safetensors` exists | Direct load |
| **GGUF** | `.gguf` extension | GGUF parser → tensor iterator |
| **.bin** | `.bin` extension (legacy PyTorch) | Not supported initially |

GGUF needs a name mapping table (GGUF uses different naming than HF safetensors).
This is per-model-family — kept in the model file alongside `stacked_params`.

## Engine Integration

```rust
// prelude-server/src/main.rs — model loading flow

fn load_model(model_path: &Path, ops: &Ops) -> Result<Box<dyn Model>> {
    // 1. Parse config
    let config = ModelConfig::from_hf_config(&model_path.join("config.json"))?;
    
    // 2. Resolve model class
    let arch = &config.architectures[0];
    let registration = resolve_model(arch)
        .ok_or_else(|| anyhow!("unsupported architecture: {arch}"))?;
    
    // 3. Construct model
    let mut model = (registration.create)(&config, ops)?;
    
    // 4. Load weights
    let mut weights = WeightIterator::open(model_path)?;
    model.load_weights(&mut weights, ops)?;
    
    Ok(model)
}
```

The engine calls `resolve_model()` once. No runtime registry modification.
`inventory` collects all registrations at link time — the registry is immutable.
