# Supported Models

## Model Support Matrix

| Architecture | Models | GPU Backend | CPU | Continuous Batching |
|---|---|---|---|---|
| Qwen3 (dense) | 0.6B, 1.7B, 4B, 8B, 14B, 32B | FA4 / FlashInfer | Yes | Yes |
| Qwen3 MoE | 30B-A3B | FA4 / FlashInfer | No | Yes |
| Qwen3.5 (hybrid) | Dense + 35B-A3B MoE | FA4 / FlashInfer | No | Yes (DeltaNet pool) |
| Qwen3-Next (hybrid) | 80B-A3B | FA4 / FlashInfer | No | Yes (DeltaNet pool) |
| Gemma3 | All sizes | FA4 / FlashInfer | Yes | Yes |
| Gemma4 | All sizes | FA4 / FlashInfer | Yes | Yes |

**Backend key:** FA4 = CuTeDSL AOT (SM90+), FlashInfer = AOT attention (FA2 SM80+ / FA3 SM90+). GPU dispatch: FA4 → FlashInfer → composed F32 SDPA.

**GGUF loading** currently resolves `qwen35` and `qwen35moe` architectures via
the native Qwen3.5 implementation. Other GGUF architectures (llama, gemma,
phi3, …) are not wired on this branch.

**MoE caveat on Blackwell (SM103)**: the MoE dispatch in this branch hits a
silent CUDA error on the first request. Verified-working models on B300 as of
this PR are the dense Qwen3 family and Gemma3/Gemma4; MoE Blackwell support
needs the kernel-level fixes from the GPU-kernels stack (fused MoE GEMM +
GPU sort + routing write-race fix) that aren't in this PR.

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

GGUF files are auto-detected by the `.gguf` extension or from HuggingFace Hub repos (e.g., `--model unsloth/Qwen3.5-0.8B-GGUF`). The architecture is read from GGUF metadata (`general.architecture`). On this branch the native Qwen3.5 model serves `qwen35` and `qwen35moe`; other architectures (llama, gemma, phi3, …) fall through with `GGUF loading not supported`.

## Hybrid Models

Qwen3.5 and Qwen3-Next are hybrid architectures that combine Gated DeltaNet (linear recurrent attention) with standard attention layers. This gives sub-quadratic complexity for most layers while retaining full attention where it matters.

**How it works:** Every Nth layer (typically every 4th) uses full causal attention. All other layers use Gated DeltaNet, which maintains a recurrent state instead of KV cache, giving O(1) per-token cost during decode.

**DeltaNet state pool:** Concurrent multi-request decode is supported via a pre-allocated state pool. Each active decode request occupies one slot. Configure with:

```bash
PRELUDE_DELTANET_POOL_SLOTS=16  # default 8
```

Increase this if you expect more concurrent generation requests on a hybrid model.

**Limitations:**
- GPU only (FlashInfer or FA3 required for native weights)
- Decode throughput depends on `PRELUDE_DELTANET_POOL_SLOTS`

## Known Limitations

- No multimodal support (vision, audio)
- No tool calling / function calling
- No LoRA adapter serving
- No tensor parallelism (single-GPU only)
- GGUF models support both CPU and GPU (via llama.cpp with `ggml-quants` feature)
- No FP8 or INT8 quantization for native models (only GGUF quantization)
