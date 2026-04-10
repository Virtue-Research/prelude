# Adding a Component

## Adding a New Model

> **Skill:** [`docs/skills/adding-a-model.md`](../../../skills/adding-a-model.md) contains a detailed reference with a checklist, testing instructions, and common failure modes. The guide below reflects the current code structure — prefer it over the skill for registration details (the skill references `ALL_ARCH_SPECS` and a subdirectory layout that are outdated).

All model code lives in `prelude-core/src/models/`. Adding a model touches exactly two files: the new model file and `mod.rs`.

### Step 1 — Create the model file

```
crates/prelude-core/src/models/mymodel.rs
```

### Step 2 — Define the config struct

Use the `model_config!` macro (defined in `models/mod.rs`). It generates a `Deserialize` impl that handles three field categories:

```rust
model_config! {
    pub struct MyModelConfig("MyModel") {
        required {
            // Must be present in config.json — error if missing
            hidden_size: usize,
            num_hidden_layers: usize,
            num_attention_heads: usize,
            num_key_value_heads: usize,
            head_dim: usize,
            intermediate_size: usize,
            vocab_size: usize,
        }
        serde_default {
            // Optional — uses serde default (0 / false / None) if absent
            attention_bias: bool,
            sliding_window: Option<usize>,
        }
        warn_default {
            // Has a fallback value — logs a warning if absent in config.json
            rms_norm_eps: f64 = 1e-6,
            rope_theta: f64 = 500_000.0,
            max_position_embeddings: usize = 32768,
        }
    }
}
```

### Step 3 — Implement the model structs

Build the architecture bottom-up: attention → MLP → decoder layer → backbone → causal LM head. Use `commons/` building blocks and call `ops.xxx()` for all compute.

```rust
use crate::models::commons::{BatchAttnContext, BatchState, Linear, RmsNorm, RotaryEmbedding};
use crate::ops::{MaskType, PagedParams, VarlenParams};

struct MyModelAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: Arc<RotaryEmbedding>,
    num_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
}

impl MyModelAttention {
    fn new(cfg: &MyModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: Linear::load(vb.pp("q_proj"), cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, false)?,
            k_proj: Linear::load(vb.pp("k_proj"), cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, false)?,
            v_proj: Linear::load(vb.pp("v_proj"), cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, false)?,
            o_proj: Linear::load(vb.pp("o_proj"), cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, false)?,
            rotary_emb: Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?),
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            softmax_scale: (cfg.head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
        let ops = ctx.ops;
        let bs = BatchState::no_lora();
        let total_q = x.dim(0)?;

        let q = self.q_proj.forward(x, &bs, ops)?;
        let k = self.k_proj.forward(x, &bs, ops)?;
        let v = self.v_proj.forward(x, &bs, ops)?;

        let kv_cache = ctx.paged_kv.map(|kv| (kv.key_cache, kv.value_cache, kv.slot_mapping));
        let (q, k) = ops.rope_and_cache(
            &q, &k, &v,
            &self.rotary_emb.cos, &self.rotary_emb.sin,
            ctx.position_ids, kv_cache,
        )?;

        // Dispatch: paged (decode/chunked-prefill) vs varlen (prefill without cache)
        let attn_out = if let Some(kv) = ctx.paged_kv {
            ops.paged_attention(&q, kv.key_cache, kv.value_cache, &PagedParams {
                block_tables: kv.block_tables,
                cu_seqlens_q: ctx.cu_seqlens_q, cu_seqlens_k: kv.cu_seqlens_k,
                max_seqlen_q: ctx.max_seqlen_q, max_seqlen_k: kv.max_seqlen_k,
                scale: self.softmax_scale, mask: MaskType::Causal, softcap: None,
            })?
        } else {
            ops.varlen_attention(&q, &k, &v, &VarlenParams {
                cu_seqlens_q: ctx.cu_seqlens_q, cu_seqlens_k: ctx.cu_seqlens_q,
                max_seqlen_q: ctx.max_seqlen_q, max_seqlen_k: ctx.max_seqlen_q,
                scale: self.softmax_scale, mask: MaskType::Causal, softcap: None,
            })?
        };

        self.o_proj.forward(&attn_out.reshape((total_q, self.num_heads * self.head_dim))?, &bs, ops)
    }
}
```

### Step 4 — Implement `ModelForward`

```rust
pub struct MyModelForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<MyModelDecoderLayer>,
    norm_weight: Tensor,
    lm_head: Linear,
    norm_eps: f32,
}

impl ModelForward for MyModelForCausalLM {
    fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
        let ops = ctx.ops;
        let mut h = self.embed_tokens.forward(packed_input)?;

        for layer in &mut self.layers {
            h = layer.forward(&h, ctx)?;
        }

        let h = ops.rms_norm(&h, &self.norm_weight, self.norm_eps)?;
        let last = last_token_select(&h, ctx.seq_lens)?;
        self.lm_head.forward(&last.unsqueeze(1)?, &BatchState::no_lora(), ops)
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.reset_kv_cache();
        }
    }
}
```

### Step 5 — Register with `ArchSpec`

Add a `mod meta {}` block at the bottom of the model file:

```rust
mod meta {
    use super::*;
    use crate::engine::{CommonModelConfig, RuntimeCaps, TaskKind, WeightsBackend};
    use crate::models::registry::{ArchSpec, ArchSpecEntry, ParsedModelConfig, candle_model_err, parse_json};
    use crate::loading::var_builder::VarBuilder;

    pub(crate) struct MyModelArchSpec;
    pub(crate) static MY_MODEL_ARCH_SPEC: MyModelArchSpec = MyModelArchSpec;
    inventory::submit!(ArchSpecEntry::new(&MY_MODEL_ARCH_SPEC));

    impl ArchSpec for MyModelArchSpec {
        fn name(&self) -> &'static str { "mymodel" }

        // Matched against the prefix of HF architecture strings e.g. "MyModelForCausalLM"
        fn architecture_aliases(&self) -> &'static [&'static str] { &["MyModel"] }

        // Matched against config.json "model_type" field
        fn model_type_aliases(&self) -> &'static [&'static str] { &["my_model"] }

        fn supported_tasks(&self) -> &'static [TaskKind] { &[TaskKind::Generate] }

        fn parse_config(
            &self,
            _task: TaskKind,
            _raw: &serde_json::Value,
            content: &str,
        ) -> Result<ParsedModelConfig, EngineError> {
            let cfg = parse_json::<MyModelConfig>(content, "MyModel config")?;
            Ok(ParsedModelConfig {
                common: CommonModelConfig {
                    vocab_size: cfg.vocab_size,
                    num_hidden_layers: cfg.num_hidden_layers,
                    max_position_embeddings: cfg.max_position_embeddings,
                    num_attention_heads: cfg.num_attention_heads,
                    num_key_value_heads: cfg.num_key_value_heads,
                    head_dim: cfg.head_dim,
                },
                deltanet: None,
                arch_config: Box::new(cfg),
            })
        }

        fn build_model(
            &self,
            arch_config: &dyn std::any::Any,
            vb: VarBuilder<'_>,
        ) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
            let cfg = arch_config.downcast_ref::<MyModelConfig>().unwrap();
            Ok(Box::new(MyModelForCausalLM::new(cfg, vb).map_err(candle_model_err)?))
        }

        fn runtime_caps(&self, task: TaskKind, backend: WeightsBackend, device: &Device) -> RuntimeCaps {
            let is_safetensors = backend == WeightsBackend::Safetensors;
            RuntimeCaps {
                supports_kv_cache:     is_safetensors && task == TaskKind::Generate,
                supports_prefix_cache: false,
                supports_paged_attn:   false,
                supports_varlen:       device.is_cuda() && is_safetensors,
                supports_deltanet:     false,
                supports_cuda_graph:   false,
            }
        }
    }
}
```

### Step 6 — Declare the module

Add one line to `crates/prelude-core/src/models/mod.rs`:

```rust
pub mod mymodel;
```

That's it. The `inventory::submit!` in step 5 auto-registers the model at startup — no other files need to change.

---

### Rules to follow

- **Only call `ops.xxx()`** for compute — no direct kernel calls, no `#[cfg(feature = "cuda")]`
- **No device branching** — the same forward runs on CUDA, CPU, and any future device
- **Use `Linear::load()`** for all weight tensors — it selects the right backend (CUTLASS, OneDNN, quantized) automatically
- **Use `model_config!`** for the config struct — it handles missing/default field warnings consistently

---

## Adding a New Attention Backend

Add a file in `prelude-cuda/src/attn/` and add a dispatch branch in `CudaOps::varlen_attention` / `CudaOps::paged_attention` in `prelude-cuda/src/cuda_ops.rs`. No changes needed in model code.

## Adding a New Device Backend

| | What |
|--|------|
| **MUST implement** | `impl Ops`: tensor primitives (via CubeCL or XLA), `matmul`, `varlen_attention`, `paged_attention`, `kv_cache` ops; plus `Executor` |
| **SHOULD implement** | `fused_add_rmsnorm`, `fused_silu_mul` — called on every forward pass |
| **CAN skip** | `rms_norm`, `layer_norm`, activation ops, conv ops — default composed implementations in `trait Ops` handle them automatically |

Steps:
1. Create `prelude-{device}/src/lib.rs` with `register()` calling `register_backend()` and `register_executor()`.
2. Implement `{Device}Ops` (unit struct or struct with fields) with `impl Ops for {Device}Ops`. Only override methods you have kernels for — unlisted fused methods automatically return `None` and fall through to composed.
3. Implement `{Device}Executor` for the `Executor` trait.
4. Add kernel sub-crates under `prelude-{device}/{kernel}/` with a `build.rs` that compiles from `third_party/`.
5. Add the feature gate in `prelude-server/Cargo.toml` and the `register()` call in `main.rs`.

Model code requires zero changes — no `#[cfg]`, no device-specific branches anywhere in model files.

## Modifying the Scheduler

The scheduler (`prelude-core/src/scheduler/`) is pure CPU — no GPU calls. The sequence state machine (`Waiting → Prefilling → Decoding → Finished`, with preemption back to `Waiting`) lives in `scheduler/state.rs`. Budget constraints (`max_running_requests`, `max_prefill_tokens`, `max_total_tokens`) are configurable via `SchedulerConfig`.
