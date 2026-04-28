# Models Layout

Models are **self-contained**: each model lives in a single file at the top level of this
directory, with its own `Attention` / `GatedMlp` / `DecoderLayer` structs and forward logic,
1:1 mapping to the HuggingFace transformers reference.

```
models/
├── commons/              # Universally shared: Linear, RmsNorm, Embedding, RotaryEmbedding, contexts
├── config.rs             # Shared config types (Qwen3Config, ...)
├── forward.rs            # ModelForward trait + BatchAttnContext plumbing
├── registry.rs           # ModelRegistry: inventory-based architecture auto-registration
├── qwen3.rs              # Qwen3 (GQA + QK-norm)
├── qwen3_moe.rs          # Qwen3-MoE (sparse MoE MLP + shared decoder layer)
├── qwen3_5.rs            # Qwen3.5 hybrid (gated attention + DeltaNet)
├── qwen3_next.rs         # Qwen3-Next hybrid with extreme-sparsity MoE
├── gemma3.rs             # Gemma3
└── gemma4.rs             # Gemma4 with KV sharing
```

`commons/` holds only what is **universally common** across all models — weight containers
(`Linear`, `Embedding`), `RotaryEmbedding`, `RmsNorm`, `fused_qkv_projection` helper, and
context structs. Model-specific components (e.g. DeltaNet, MoE routing) stay in the model file.
Resist the urge to factor out anything model-pair-specific into `commons/`.

## Adding a new model

1. Create `<arch>.rs` with `Attention` / `Mlp` / `DecoderLayer` / `<Arch>ModelForCausalLM` structs.
2. Use `Linear`, `RmsNorm`, `Embedding`, `RotaryEmbedding` from `commons/` for weight storage.
3. Inside `forward`, call `ops.xxx()` for all compute (fused kernels auto-fallback).
4. Register the architecture via the `model_config!` macro and `inventory::submit!`.
5. Add a config struct in `config.rs` (or its own file if large).
