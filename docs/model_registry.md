# Model Registry & Loading

Reference for how models are wired into Prelude. For a step-by-step walkthrough of
adding a new model, see [`skills/adding-a-model.md`](skills/adding-a-model.md).

## File Layout

Models are **flat, one file per architecture** under `crates/prelude-core/src/models/`:

```
crates/prelude-core/src/models/
├── mod.rs                     # `pub mod qwen3; pub mod gemma3; ...`
├── registry.rs                # ArchSpec trait + inventory collection
├── forward.rs                 # ModelForward trait + helper traits
├── config.rs                  # Shared config types (rope, MoE configs, ...)
├── commons/                   # Shared layers: Linear, RmsNorm, Embedding, RotaryEmbedding
├── qwen3.rs                   # Qwen3 (dense GQA + QK-norm)
├── qwen3_5.rs                 # Qwen3.5 hybrid (DeltaNet + attention)
├── qwen3_moe.rs               # Qwen3-MoE
├── qwen3_next.rs              # Qwen3-Next hybrid MoE
├── gemma3.rs                  # Gemma3
└── gemma4.rs                  # Gemma4
```

Each model is its own `.rs` file. No subdirectories, no `meta.rs`, no `architectures/`
wrapper. The file contains config structs, model structs, `ModelForward` impls, and
one `ArchSpec` registration block.

## Auto-Registration via `inventory`

`registry.rs` defines an `ArchSpec` trait plus an `ArchSpecEntry` wrapper collected
by the `inventory` crate. Each model file submits one entry; no central list to edit.

```rust
// crates/prelude-core/src/models/registry.rs (shortened)
pub(crate) trait ArchSpec: Sync {
    fn name(&self) -> &'static str;
    fn architecture_aliases(&self) -> &'static [&'static str];   // HF "architectures"
    fn model_type_aliases(&self) -> &'static [&'static str];     // HF "model_type"
    fn gguf_aliases(&self) -> &'static [&'static str] { &[] }    // GGUF architecture
    fn supported_tasks(&self) -> &'static [TaskKind];            // Generate / Classify / Embed

    fn parse_config(
        &self, task: TaskKind, raw: &serde_json::Value, content: &str,
    ) -> Result<ParsedModelConfig, EngineError>;

    fn build_model(
        &self, arch_config: &dyn Any, vb: VarBuilder<'_>,
    ) -> Result<Box<dyn ModelForward>, EngineError>;

    fn runtime_caps(&self, task: TaskKind, backend: WeightsBackend, device: &Device)
        -> RuntimeCaps;

    // Optional — only for models that ship GGUF checkpoints.
    fn load_gguf(
        &self, ct: gguf_file::Content, reader: &mut File, device: &Device,
    ) -> Result<GgufLoadResult, EngineError> { /* default: error */ }
}

pub(crate) struct ArchSpecEntry { pub spec: &'static dyn ArchSpec }
inventory::collect!(ArchSpecEntry);
```

Registration happens inside the model file itself:

```rust
// crates/prelude-core/src/models/qwen3.rs (tail)
pub(crate) struct Qwen3ArchSpec;
pub(crate) static QWEN3_ARCH_SPEC: Qwen3ArchSpec = Qwen3ArchSpec;
inventory::submit!(crate::models::registry::ArchSpecEntry::new(&QWEN3_ARCH_SPEC));

impl ArchSpec for Qwen3ArchSpec {
    fn name(&self) -> &'static str { "qwen3" }
    fn architecture_aliases(&self) -> &'static [&'static str] { ARCHITECTURE_ALIASES }
    fn model_type_aliases(&self) -> &'static [&'static str] { MODEL_TYPE_ALIASES }
    fn supported_tasks(&self) -> &'static [TaskKind] { SUPPORTED_TASKS }
    // ... parse_config, build_model, runtime_caps ...
}
```

Adding a new architecture = new `.rs` file with its own `ArchSpec` impl + one
`inventory::submit!` line. `registry.rs` never needs edits.

## Resolving an Architecture

`ParsedModelConfig` comes back from `ArchSpec::parse_config()`; it bundles the
engine-facing `CommonModelConfig`, any DeltaNet pool config, and an opaque
`arch_config: Box<dyn Any>` that the same `ArchSpec::build_model()` later
down-casts to its own concrete config struct.

The resolver walks HF config fields in order:

1. **`architectures: ["Qwen3ForCausalLM", ...]`** — prefix match via
   `resolve_architecture_name` / `find_arch_spec_by_architecture_prefix`. The
   suffix (`ForCausalLM`, `ForSequenceClassification`, `ForEmbedding`, ...) is
   mapped to `TaskKind` by `task_from_architecture_suffix`.
2. **`model_type: "qwen3"`** — matched via `find_arch_spec_by_model_type` when
   the `architectures` list is missing or unhandled.
3. **GGUF `general.architecture` metadata** — `find_arch_spec_by_gguf_arch` for
   GGUF loads; uses `ArchSpec::gguf_aliases()`.

Multiple HF architectures map to the same `ArchSpec` by returning them all in
`architecture_aliases()` — e.g. Qwen3 covers both `Qwen3ForCausalLM` and
`Qwen3ForSequenceClassification` with one spec.

## Weight Loading

Models take a `VarBuilder` in `ArchSpec::build_model`:

```rust
use crate::loading::var_builder::VarBuilder;

fn build_model(
    &self, arch_config: &dyn Any, vb: VarBuilder<'_>,
) -> Result<Box<dyn ModelForward>, EngineError> {
    let cfg = arch_config.downcast_ref::<Qwen3Config>().unwrap();
    let model = Qwen3Model::load(vb, cfg)?;
    Ok(Box::new(model))
}
```

Inside `Qwen3Model::load`, the model walks the tensor tree with `vb.pp("model")`
and builds each layer from shared building blocks in `models::commons`:

- `Linear::load(vb, in_dim, out_dim, bias)` — picks oneDNN BRGeMM on CPU,
  the 3-tier GEMM dispatch on CUDA.
- `RmsNorm::load(vb, dim, eps)` — AVX-512 on CPU, a CUDA kernel on GPU.
- `Embedding::load`, `RotaryEmbedding`, etc.

There is no separate `WeightLoadable` trait or central parameter-stacking table
— fused QKV is handled per-model (see `Qwen3Attention::load`).

## GGUF Loading

GGUF follows a parallel path: `ArchSpec::load_gguf()` consumes a
`gguf_file::Content`, returns the built model plus `CommonModelConfig` and
`eos_token_ids`. Default impl returns `Unavailable`; only architectures that
ship GGUF checkpoints (currently Qwen3 / Qwen3.5 / Qwen3-MoE) override it.

`gguf_aliases()` reports the GGUF `general.architecture` strings this `ArchSpec`
handles (e.g. `qwen3` serves `qwen3` and `qwen35`).

## Engine Integration

```rust
// crates/prelude-core/src/engine/loading.rs
let (spec, task) = resolve_architecture_name(&config["architectures"][0])
    .or_else(|| find_arch_spec_by_model_type(&config["model_type"]))
    .ok_or(EngineError::UnknownArchitecture)?;

let parsed = spec.parse_config(task, &raw_json, &raw_content)?;
let vb     = VarBuilder::from_safetensors(shards, dtype, device)?;
let model  = spec.build_model(&*parsed.arch_config, vb)?;
let caps   = spec.runtime_caps(task, WeightsBackend::Safetensors, device);
```

The engine stores the returned `Box<dyn ModelForward>` plus `RuntimeCaps`
(which drive paged-attn, CUDA-graph, DeltaNet-pool, and prefix-cache
decisions) and does not look at the concrete architecture again.

## Adding a New Architecture

See [`skills/adding-a-model.md`](skills/adding-a-model.md) for the end-to-end
walkthrough.
