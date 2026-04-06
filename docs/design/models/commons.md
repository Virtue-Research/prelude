## Models & Shared Layers

> **Note:** `modules/` has been replaced by `models/commons/`. This document describes the current architecture.

### Architecture

Models are **self-contained** (like vLLM): each model file defines its own structs and forward logic,
1:1 mapping to HuggingFace transformers. Models call `ops.xxx()` directly for all compute.

Shared components live in `models/commons/` — only things that are **universally common**:

```
models/
├── layers/
│   ├── linear.rs         # Linear (unified: quantized + float + LoRA), RmsNorm
│   ├��─ embedding.rs      # Embedding lookup
│   ├── attn_utils.rs     # RotaryEmbedding, attention dispatch helpers
│   └── mod.rs            # Context structs (BatchAttnContext, PagedKvContext), utilities
├── qwen3.rs              # Self-contained: Qwen3Attention, Qwen3MLP, Qwen3DecoderLayer
├── qwen3_5.rs            # Self-contained: gated attention, DeltaNet
├── qwen3_moe.rs          # Self-contained: SparseMoeBlock
└── ...
```

### Fusion Strategy

Fusion is handled entirely in **OpsBundle** — model code never does if/else on fusion:

```rust
// Model code — clean, no branching
let (residual, normed) = ops.fused_add_rmsnorm(&residual, &h, &weight, eps)?;
let (q, k) = ops.qknorm_rope_and_cache(&q, &k, &v, ..., paged_kv)?;
let activated = ops.fused_silu_mul(&gate, &up)?;
```

OpsBundle internally: try device fused kernel → auto-fallback to composed ops.

### Linear as Parameter Carrier

`Linear` holds weights + optional LoRA state. All compute decisions (fused QKV, LoRA dispatch,
quantization) are delegated to OpsBundle:

```rust
// Linear passes its weights to OpsBundle for compute
ops.qkv_projection(x, &q.weight, &k.weight, &v.weight, lora_state)
```

This ensures LoRA, quantization, and fusion work transparently across all devices.

### Device Crate Integration

```
prelude-cuda/
├── cuda_ops.rs           # impl all op traits, hot-path kernels
└── ops/                  # CUDA kernel wrappers (rmsnorm, rope, kv_cache, moe, gemm)

prelude-cpu/
├��─ cpu_ops.rs            # base() → CubeCL CPU, override specific ops (GGUF quant, oneDNN GEMM)
└── linear_backends.rs    # QuantFormat registration (Q4_0, Q4_K via inventory)
```

Device crates provide `OpsBundle` via builder pattern:
```rust
OpsBundle::from_all(composed)       // ComposedOps as base
    .with_tensor(cuda.clone())      // override TensorOps with CUDA kernels
    .with_attn(flash_attn)          // override attention with FlashAttn
    .with_session(cuda_session)     // override session for CUDA graphs
```

### Testing Strategy

Precision tests live in `tests/tensor_ops.rs` and `tests/modules.rs`, comparing against
**PyTorch reference values** generated at test time via subprocess:

```rust
let ref_json = require_pytorch!("import torch; ...; print(json.dumps(y.tolist()))");
let reference = parse_f32_2d(&ref_json);
assert_close(&ours, &reference, atol=1e-5, "rmsnorm");
```

- PyTorch subprocess ensures same-device reference (no cross-platform drift)
- Tests auto-skip if Python + PyTorch not installed
- Tolerance: f32 atol=1e-5, bf16 atol=1e-2
