# Models

## Overview

Each model is a self-contained file with its own structs and forward logic, mapping 1:1 to HuggingFace transformers. Models call `ops.xxx()` for all compute — device dispatch and fusion are invisible to model code. `models/commons/` holds only what is universally shared: weight containers (`Linear`, `Embedding`), `RotaryEmbedding`, and context structs.

Three properties hold for all model code:

- **Device-agnostic** — no `#[cfg(feature = "cuda")]`, no direct kernel calls
- **Fusion-agnostic** — no if/else between fused and unfused paths; the `Ops` trait implementation decides internally
- **Quant-agnostic** — `Linear::forward` dispatches to the right backend transparently

## File Layout

```
prelude-core/src/models/
├── commons/
│   ├── linear.rs         # Linear (backend-dispatched), RmsNorm, LinearBackend trait
│   ├── embedding.rs      # Embedding lookup
│   ├── attn_utils.rs     # RotaryEmbedding, attention dispatch helpers
│   ├── activation.rs     # Activation functions
│   └── mod.rs            # Context structs (BatchAttnContext, LayerAttnContext, BatchState)
├── forward.rs            # trait ModelForward + sub-traits (LogitsSplitModel, KvCacheModel, ...)
├── registry.rs           # ArchSpec trait + ArchSpecEntry inventory registration
├── config.rs             # Per-model config parsing helpers
├── qwen3.rs              # Qwen3 (GQA + QK-norm)
├── qwen3_moe.rs          # Qwen3-MoE (SparseMoeBlock)
├── qwen3_5.rs            # Qwen3.5 hybrid (gated attention + DeltaNet)
├── qwen3_next.rs         # Qwen3-Next
├── gemma3.rs             # Gemma3 (softcap, sliding window)
└── gemma4.rs             # Gemma4
```

Model-specific components stay in the model file. No forced abstraction into commons/.

## ModelForward Trait

All model architectures implement `ModelForward`:

```rust
// prelude-core/src/models/forward.rs

pub trait ModelForward: Send {
    /// Main forward pass. packed_input is flat token IDs [total_tokens].
    fn forward(
        &mut self,
        packed_input: &Tensor,
        ctx: &mut BatchAttnContext,
    ) -> Result<Tensor>;

    fn clear_kv_cache(&mut self);

    // Optional capability accessors — default to None
    fn as_logits_model(&self) -> Option<&dyn LogitsSplitModel> { None }
    fn as_logits_model_mut(&mut self) -> Option<&mut dyn LogitsSplitModel> { None }
    fn as_kv_cache_model(&mut self) -> Option<&mut dyn KvCacheModel> { None }
    fn as_classifier(&self) -> Option<&dyn ClassifierModel> { None }
    fn as_embedding(&self) -> Option<&dyn EmbeddingModel> { None }
    fn kv_cache_sharing(&self) -> Vec<Option<usize>> { vec![] }
    fn generate_direct(&mut self, ..) -> Result<Option<(Vec<u32>, Vec<f32>)>> { Ok(None) }
}
```

Sub-traits for optional capabilities:

| Sub-trait | Purpose |
|-----------|---------|
| `LogitsSplitModel` | Split forward into hidden states + lm_head (for prompt logprobs) |
| `KvCacheModel` | CPU sequential decode with internal KV cache |
| `ClassifierModel` | Expose `num_labels` + label map |
| `EmbeddingModel` | Expose `embedding_dim` |

## Model Registration

### Auto-registration via `ArchSpec`

Each model implements the `ArchSpec` trait and registers a static instance via the `inventory` crate. No changes to `registry.rs` when adding a new model.

```rust
// prelude-core/src/models/registry.rs

pub(crate) trait ArchSpec: Sync {
    fn name(&self) -> &'static str;
    fn architecture_aliases(&self) -> &'static [&'static str];
    fn model_type_aliases(&self) -> &'static [&'static str];
    fn supported_tasks(&self) -> &'static [TaskKind];

    fn parse_config(&self, task: TaskKind, raw: &serde_json::Value, content: &str)
        -> Result<ParsedModelConfig, EngineError>;

    fn build_model(&self, arch_config: &dyn Any, vb: VarBuilder<'_>)
        -> Result<Box<dyn ModelForward>, EngineError>;

    fn runtime_caps(&self, task: TaskKind, backend: WeightsBackend, device: &Device)
        -> RuntimeCaps;

    // Optional GGUF support
    fn gguf_aliases(&self) -> &'static [&'static str] { &[] }
    fn load_gguf(&self, ..) -> Result<GgufLoadResult, EngineError> { Err(..) }
}

pub(crate) struct ArchSpecEntry {
    pub spec: &'static dyn ArchSpec,
}
inventory::collect!(ArchSpecEntry);
```

Registration in each model file (one `inventory::submit!` call):

```rust
// prelude-core/src/models/qwen3.rs — inside mod meta {}

pub(crate) struct Qwen3ArchSpec;
pub(crate) static QWEN3_ARCH_SPEC: Qwen3ArchSpec = Qwen3ArchSpec;
inventory::submit!(ArchSpecEntry::new(&QWEN3_ARCH_SPEC));

impl ArchSpec for Qwen3ArchSpec {
    fn name(&self) -> &'static str { "qwen3" }
    fn architecture_aliases(&self) -> &'static [&'static str] { &["Qwen3", "Qwen3Model"] }
    fn model_type_aliases(&self) -> &'static [&'static str] { &["qwen3"] }
    fn supported_tasks(&self) -> &'static [TaskKind] {
        &[TaskKind::Generate, TaskKind::Classify, TaskKind::Embed]
    }
    fn parse_config(&self, task: TaskKind, raw: &serde_json::Value, content: &str)
        -> Result<ParsedModelConfig, EngineError> { .. }
    fn build_model(&self, arch_config: &dyn Any, vb: VarBuilder<'_>)
        -> Result<Box<dyn ModelForward>, EngineError> { .. }
    ..
}
```

Adding a new model = one new file + one `inventory::submit!`. Zero changes to existing code.

### Architecture Resolution

The registry resolves HuggingFace `architectures` strings (e.g., `"Qwen3ForCausalLM"`) by splitting on `"For"`:

- Prefix (`"Qwen3"`) → matched against `architecture_aliases()` (case/punctuation-insensitive)
- Suffix (`"CausalLM"`) → mapped to `TaskKind::Generate`, `TaskKind::Classify`, or `TaskKind::Embed`

GGUF uses a separate `gguf_aliases()` lookup.

### Weight Loading

Weight loading goes through `VarBuilder` inside `ArchSpec::build_model()`. Each model reads its own weights by name from the VarBuilder, which auto-detects format:

| Format | Detection |
|--------|-----------|
| Safetensors (sharded) | `model.safetensors.index.json` exists |
| Safetensors (single) | `model.safetensors` exists |
| GGUF | `.gguf` extension → `ArchSpec::load_gguf()` |

## Model Structure

### Linear as Backend Dispatcher

`Linear` is the unified weight container. It holds a `Box<dyn LinearBackend>` — the backend is chosen at load time based on device and quantization format. Model code always calls `Linear::forward()` and never branches on backend:

```rust
// prelude-core/src/models/commons/linear.rs

pub struct Linear {
    inner: Box<dyn LinearBackend>,
}

impl Linear {
    pub fn forward(&self, x: &Tensor, _ctx: &BatchState, _ops: &dyn Ops) -> Result<Tensor> {
        self.inner.forward(x)  // delegates entirely to backend
    }
}
```

The backend is selected when `Linear` is constructed:

| Device | Backend | GEMM dispatch |
|--------|---------|---------------|
| CUDA | `NaiveLinearBackend` | `Tensor::matmul()` → registered CUTLASS/DeepGEMM |
| CPU | `OnednnLinear` (via inventory) | OneDNN GEMM |
| GGUF | registered `QuantFormat` backend | MMQ / dequant GEMM |

> **LoRA and TP** — `Linear::forward` accepts `ctx` and `ops` for API stability. These will be used when LoRA and tensor-parallelism land. Currently unused (prefixed `_` in the implementation).

### Fusion Strategy

The `Ops` trait implementation owns all fusion decisions. Model code calls ops directly; the implementation tries the device kernel and falls back to composed ops internally:

```rust
// Model code calls ops — no branching, no fallback handling
let h = ops.rms_norm(x, &self.ln1_weight, eps)?;
let (x_res, h2) = ops.add_rmsnorm(x, &h, &self.ln2_weight, eps)?;
let out = ops.silu_mul(&gate, &up)?;
```

If a kernel developer adds a fused op to `CudaOps`, every model calling it benefits automatically.

## Model Code Patterns

### Decoder Layer (Qwen3)

```rust
// prelude-core/src/models/qwen3.rs — DecoderLayer

fn forward(&self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
    let ops = ctx.ops;

    // Pre-attention norm
    let h = ops.rms_norm(x, &self.ln1_weight, self.rms_norm_eps)?;

    // Attention (handles QK-norm + RoPE + KV cache write internally)
    let h = self.self_attn.forward(&h, ctx)?;

    // Residual + post-attention norm (fused when device supports it)
    let (x_res, h2) = ops.add_rmsnorm(x, &h, &self.ln2_weight, self.rms_norm_eps)?;

    // MLP + residual
    ops.add_or_fused(&x_res, &self.mlp.forward(&BatchState::no_lora(), ops, &h2)?)
}
```

### Attention Forward (Qwen3)

```rust
// prelude-core/src/models/qwen3.rs — Qwen3Attention

pub(crate) fn forward(&self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
    let ops = ctx.ops;
    let (q, k, v) = self.fused_qkv_projection(x, total_q, &bs, ops)?;

    // QK-norm + RoPE + optional paged KV cache write (ops picks optimal fusion)
    let kv_cache = ctx.paged_kv.map(|kv| (kv.key_cache, kv.value_cache, kv.slot_mapping));
    let (q, k) = ops.qknorm_rope_and_cache(
        &q, &k, &v,
        &self.q_norm_weight, &self.k_norm_weight,
        &self.rotary_emb.cos, &self.rotary_emb.sin,
        ctx.position_ids, self.rms_norm_eps as f32,
        kv_cache,
    )?;

    // Dispatch to paged (decode/chunked-prefill) or varlen (prefill without cache)
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

    self.o_proj.forward(&attn_out.reshape((total_q, self.hidden_size))?, &bs, ops)
}
```

`paged_attention` handles both decode (Q=1) and chunked prefill (Q>1). The `varlen` path is used when there is no paged KV cache (e.g., CPU sequential decode falls back to this path internally).

### MLP Forward (Qwen3)

```rust
// prelude-core/src/models/qwen3.rs — GatedMlp

fn forward(&self, ctx: &BatchState, ops: &dyn Ops, x: &Tensor) -> Result<Tensor> {
    // Fused gate+up projection if weights were merged at load time
    if let Some(ref gup) = self.gate_up_proj {
        let gate_up = gup.forward(x, ctx, ops)?;
        let (gate, up) = gate_up.split_at_dim_half(..)?;
        return self.down_proj.forward(&ops.silu_mul(&gate, &up)?, ctx, ops);
    }
    // Separate projections fallback
    let gate = self.gate_proj.forward(x, ctx, ops)?;
    let up   = self.up_proj.forward(x, ctx, ops)?;
    self.down_proj.forward(&ops.silu_mul(&gate, &up)?, ctx, ops)
}
```

## How Inference Kernels Are Reached

Model code never names FlashInfer, DeepGEMM, CUTLASS, or FA4 directly. They are reached through two dispatch chains that are invisible to the model:

**Attention kernels** — called via `ops.paged_attention()` / `ops.varlen_attention()`:

```
self.self_attn.forward(&h, ctx)         ← model code (qwen3.rs)
  └── ops.paged_attention(...)           ← ops.xxx() call (via `&dyn Ops`)
        └── CudaOps::paged_attention()   ← prelude-cuda/src/cuda_ops.rs
              ├── try FA4                ← prelude-cuda/src/attn/flash_v4.rs  (SM90+, best-effort)
              └── FlashInfer             ← prelude-cuda/src/attn/flashinfer.rs (SM80+ fallback)
```

**GEMM kernels** — called via `Linear::forward()`:

```
self.o_proj.forward(x, ctx, ops)        ← model code
  └── Linear::forward()                 ← commons/linear.rs (delegates to inner backend)
        └── NaiveLinearBackend::forward()
              └── Tensor::matmul()      ← intercepted by registered GEMM dispatch
                    ├── try DeepGEMM   ← prelude-cuda/src/ops/gemm.rs  (SM90+, BF16)
                    └── CUTLASS         ← prelude-cuda/cutlass-gemm/    (SM80+ fallback)
```

The kernel choice is made two layers below the model. Adding a new kernel to `CudaOps` benefits every model that calls the corresponding op — zero model changes required.
