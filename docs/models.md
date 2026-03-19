# Supported Models

## Model Support Matrix

| Architecture | Models | GPU Backend | CPU | Continuous Batching | Quantization |
|--------------|--------|-------------|-----|---------------------|--------------|
| Qwen3 (Dense) | 0.6B, 1.7B, 4B, 8B, 14B, 32B | FA4 / FA3 / FA2 | Yes | Yes | GGUF |
| Qwen3-MoE | 30B-A3B | FA4 / FA3 / FA2 | No | Yes | -- |
| Qwen3.5 (Hybrid) | 0.8B-27B dense, 35B-A3B MoE | FA3 | No | Yes (DeltaNet pool) | -- |
| Qwen3-Next (Hybrid) | 80B-A3B | FA3 | No | Yes (DeltaNet pool) | -- |
| Gemma3 | All sizes | FA4 / FA3 / FA2 | Yes | Yes | -- |
| GGUF | Qwen3, LLaMA, Gemma, Phi3, Qwen2 | CPU only | Yes | No | All GGUF formats |

**Backend key:** FA4 = Flash Attention v4 (SM80+ prefill only), FA3 = v3 (Hopper SM90+), FA2 = v2 (Ampere SM80+).

## Task Support

| Architecture | Generate | Classify | Embed |
|--------------|:--------:|:--------:|:-----:|
| Qwen3 (Dense) | Yes | Yes | Yes |
| Qwen3-MoE | Yes | -- | -- |
| Qwen3.5 (Hybrid) | Yes | -- | -- |
| Qwen3-Next (Hybrid) | Yes | -- | -- |
| Gemma3 | Yes | Yes | Yes |
| GGUF | Yes | -- | -- |

Classification requires a model with a classification head (e.g., `Qwen/Qwen3-Reranker`). Embedding requires an embedding model (e.g., `Qwen/Qwen3-Embedding`). The server auto-detects task from `config.json` or you can force it with `--task classify|embedding|generation`.

## Tested HuggingFace Model IDs

These exact model IDs are known to work:

| Model | Task | Size | Notes |
|-------|------|------|-------|
| `Qwen/Qwen3-0.6B` | Generate | 0.6B | Fast testing, CPU-friendly |
| `Qwen/Qwen3-1.7B` | Generate | 1.7B | |
| `Qwen/Qwen3-4B` | Generate | 4B | Good balance of speed and quality |
| `Qwen/Qwen3-8B` | Generate | 8B | |
| `Qwen/Qwen3-14B` | Generate | 14B | |
| `Qwen/Qwen3-32B` | Generate | 32B | |
| `Qwen/Qwen3-30B-A3B` | Generate | 30B (3B active) | MoE |
| `Qwen/Qwen3-Reranker` | Classify | -- | Sequence classification |
| `Qwen/Qwen3-Embedding` | Embed | -- | Text embeddings |

## Which Model Should I Use?

| I need... | Recommended | Why |
|-----------|-------------|-----|
| A chat model | `Qwen/Qwen3-4B` or `Qwen/Qwen3-8B` | Good quality, fits single GPU |
| Fast classification | `Qwen/Qwen3-Reranker` | Dedicated classification head |
| Text embeddings | `Qwen/Qwen3-Embedding` | Dedicated embedding model |
| CPU inference | `Qwen/Qwen3-0.6B` (native) or any GGUF | Small enough for CPU |
| Largest possible | `Qwen/Qwen3-32B` (dense) or `Qwen/Qwen3-Next` (80B-A3B) | Requires Hopper GPU |
| Best tok/s on Hopper | `Qwen/Qwen3-30B-A3B` | MoE: only 3B params active |

## Running a Model

### HuggingFace Hub

```bash
./target/release/prelude-server --model Qwen/Qwen3-4B --port 8000
```

Model weights are downloaded and cached automatically via the HuggingFace Hub.

### Local path

```bash
./target/release/prelude-server --model-path /path/to/model --port 8000
```

The directory must contain `config.json`, safetensor weight files, and `tokenizer.json`.

### GGUF file

```bash
PRELUDE_DEVICE=cpu ./target/release/prelude-server --model /path/to/model.gguf --port 8000
```

GGUF files are auto-detected by the `.gguf` extension. The architecture is read from GGUF metadata (`general.architecture`). Supported GGUF architectures: qwen3, qwen3moe, llama, gemma3, gemma2, gemma, phi3, qwen2.

## Hybrid Models

Qwen3.5 and Qwen3-Next are hybrid architectures that combine Gated DeltaNet (linear recurrent attention) with standard attention layers. This gives sub-quadratic complexity for most layers while retaining full attention where it matters.

**How it works:** Every Nth layer (typically every 4th) uses full causal attention. All other layers use Gated DeltaNet, which maintains a recurrent state instead of KV cache, giving O(1) per-token cost during decode.

**DeltaNet state pool:** Concurrent multi-request decode is supported via a pre-allocated state pool. Each active decode request occupies one slot. Configure with:

```bash
PRELUDE_DELTANET_POOL_SLOTS=16  # default 8
```

Increase this if you expect more concurrent generation requests on a hybrid model.

**Limitations:**
- GPU only (FA3 required)
- Decode throughput depends on `PRELUDE_DELTANET_POOL_SLOTS`

## Known Limitations

- No multimodal support (vision, audio)
- No tool calling / function calling
- No LoRA adapter serving
- No tensor parallelism (single-GPU only)
- GGUF models are CPU-only (except `qwen3moe` which needs CUDA for fused MoE GEMM)
- No FP8 or INT8 quantization for native models (only GGUF quantization)
