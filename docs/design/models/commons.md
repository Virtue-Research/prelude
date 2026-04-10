## Models & Shared Layers

### Architecture

Models are **self-contained** (like vLLM): each model file defines its own structs and forward
logic, 1:1 mapping to HuggingFace transformers. Models call `ops.xxx()` directly for all compute.

Shared components live in `models/commons/` — only things that are **universally common**:

```
models/
├── commons/
│   ├── linear.rs         # Linear front-end + LinearBackend trait + DenseLinear, RmsNorm
│   ├── embedding.rs      # Embedding lookup
│   ├── attn_utils.rs     # RotaryEmbedding, fused_qkv_projection helper
│   └── mod.rs            # Context structs (BatchAttnContext, LayerAttnContext), utilities
├── qwen3.rs              # Self-contained: Attention, GatedMlp, DecoderLayer
├── qwen3_5.rs            # Self-contained: gated attention, DeltaNet
├── qwen3_moe.rs          # Self-contained: SparseMoeBlock
└── ...
```

### Fusion Strategy

Fusion is handled entirely in **Ops** — model code never does if/else on fusion:

```rust
// Model code — clean, no branching. Kernels hidden inside Ops.
let (residual, normed) = self.input_layernorm.forward_residual(hidden, residual, ops)?;
let hidden = self.self_attn.forward(&normed, ctx)?;
let hidden = self.mlp.forward(&hidden, ops)?;
```

Ops internally: try device fused kernel → auto-fallback to composed ops.

### Linear: front-end wrapper + `LinearBackend` trait

`Linear` is the concrete type model code sees. Internally it holds `Box<dyn LinearBackend>`, and
the backend is chosen at construction time from what the checkpoint contains:

```rust
pub struct Linear { inner: Box<dyn LinearBackend> }

impl Linear {
    pub fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &dyn Ops) -> Result<Tensor> {
        self.inner.forward(x)  // delegates to the active backend
    }
}
```

`LinearBackend` is the only backend trait in the model layer. It exists because weight format is
the one dimension that genuinely varies at runtime for a given model (fp dense vs GGUF Q4_0 vs
Q4_K vs FP8 vs future formats). Each backend handles its own weight storage and matmul kernel
dispatch:

| Backend          | Crate         | Format                                          |
|------------------|---------------|-------------------------------------------------|
| `DenseLinear`    | prelude-core  | fp16/bf16/f32 (routes via candle's `matmul`)    |
| `OnednnLinear`   | prelude-cpu   | CPU BF16/F32 with oneDNN packed weights         |
| `Q4_0Linear`     | prelude-cpu   | GGUF Q4_0                                       |
| `Q4KLinear`      | prelude-cpu   | GGUF Q4_K                                       |
| `GpuQuantLinear` | prelude-cuda  | GGUF Q4_0/Q4_1/Q5_*/Q8_0/Q2K-Q6K via quant-gemm |

All impls `impl Module + impl LinearBackend`. `Module::forward` does the compute; the extra
`LinearBackend` methods (`name`, `is_quantized`, `clone_box`, `as_any`) provide identity queries.

**`Attention` and `GatedMlp` deliberately don't have analogous backend traits.** Their structure
is fixed per model — different models get different structs (`Qwen3Attention`, `LlamaAttention`,
...), not different implementations of a shared trait. Attention *kernel* selection (FA4 vs
FlashInfer vs composed SDPA) lives inside `Ops.varlen_attention` / `Ops.paged_attention` as a
runtime fallback chain, not another trait layer. See
[`subsystem_independence.md`](../subsystem_independence.md) for the reasoning.

### Device Crate Integration

```
prelude-cuda/
├── cuda_ops.rs           # impl Ops trait, hot-path kernels
├── ops/                  # CUDA kernel wrappers (rmsnorm, rope, kv_cache, moe, gemm)
└── quant_backends.rs     # GpuQuantLinear (LinearBackend + QuantFormat registration)

prelude-cpu/
├── cpu_ops.rs            # impl Ops trait: CPU primitives + override specific ops
├── linear_backends.rs    # OnednnLinearFactory, Q4_0Linear, Q4KLinear (LinearBackend + QuantFormat)
└── onednn/               # oneDNN wrapper: OnednnLinear (LinearBackend impl)
```

Device crates register their `Ops` singleton via `register()` at startup. `LinearBackend` impls
register themselves via `inventory::submit!` — `Linear::from_qtensor(qtensor)` walks the registry
to find a handler for the GGML dtype, and `Linear::from_dense(dense)` on CPU consults
`CpuLinearFactoryEntry` to wrap with `OnednnLinear` when available.

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
