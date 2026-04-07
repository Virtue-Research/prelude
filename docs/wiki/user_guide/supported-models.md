# Supported Models

## Support Matrix

| Architecture | Type | Sizes | Generate | Classify | Embed | CPU | GGUF |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| Qwen3 | Dense | 0.6B – 32B | ✓ | ✓ | ✓ | ✓ | ✓ |
| Qwen3-MoE | MoE | 30B-A3B | ✓ | — | — | — | — |
| Qwen3.5 | Hybrid (DeltaNet) | 0.8B, 2B, 4B, 9B, 27B | ✓ | — | — | — | ✓ |
| Qwen3.5-MoE | Hybrid (DeltaNet + MoE) | 35B-A3B | ✓ | — | — | — | ✓ |
| Qwen3-Next | Hybrid (DeltaNet + MoE) | 80B-A3B | ✓ | — | — | — | — |
| Gemma3 | Dense | All sizes | ✓ | ✓ | ✓ | ✓ | — |
| Gemma4 | Dense | All sizes | ✓ | — | — | — | — |

**GPU backend:** All GPU models use FlashInfer (SM80+) + FA4 (SM80+) with automatic dispatch. DeepGEMM GEMM acceleration applies to SM90+ (H100/H200).

---

## Tested HuggingFace Model IDs

| Model ID | Task | Notes |
|---|---|---|
| `Qwen/Qwen3-0.6B` | Generate | Fast, CPU-friendly |
| `Qwen/Qwen3-1.7B` | Generate | |
| `Qwen/Qwen3-4B` | Generate | Good quality/speed balance |
| `Qwen/Qwen3-8B` | Generate | |
| `Qwen/Qwen3-14B` | Generate | |
| `Qwen/Qwen3-32B` | Generate | |
| `Qwen/Qwen3-30B-A3B` | Generate | MoE — 3B active params |
| `Qwen/Qwen3-Reranker` | Classify | Sequence classification head |
| `Qwen/Qwen3-Embedding` | Embed | Text embeddings |
| `google/gemma-3-1b-it` | Generate | |
| `google/gemma-3-4b-it` | Generate | |
| `google/gemma-3-12b-it` | Generate | |
| `google/gemma-3-27b-it` | Generate | |

---

## Loading a Model

### HuggingFace Hub

```bash
./target/release/prelude-server --model Qwen/Qwen3-4B --port 8000
```

Weights are downloaded and cached automatically via the HuggingFace Hub.

### Local path

```bash
./target/release/prelude-server --model-path /path/to/model --model Qwen/Qwen3-4B --port 8000
```

The directory must contain `config.json`, safetensor weight files, and `tokenizer.json`. `--model` sets the reported model name in API responses.

### GGUF

```bash
# From a local file
./target/release/prelude-server --model-path /path/to/model.gguf --model my-model --port 8000

# From HuggingFace Hub (repo with .gguf files)
./target/release/prelude-server --model unsloth/Qwen3-4B-GGUF --port 8000
```

GGUF files are auto-detected by extension or by `general.architecture` in the GGUF metadata. Supported GGUF architectures: `qwen3`, `qwen35`, `qwen35moe`.

### Task detection

The server auto-detects task (`generate`, `classify`, `embed`) from `config.json`. To override:

```bash
./target/release/prelude-server --model Qwen/Qwen3-Reranker --task classify --port 8000
```

---

## Architecture Details

### Qwen3 (Dense)

Standard transformer with QK-Norm, GQA, and SiLU MLP. Supports all three tasks (generation, classification, embedding) via dedicated model heads.

- `config.json` `architectures`: `Qwen3ForCausalLM`, `Qwen3ForSequenceClassification`, `Qwen3ForTextEmbedding`
- Classification requires a model with a classification head (`id2label` in config)
- Embedding uses the last hidden state, optionally with a sentence-transformers Dense projection

### Qwen3-MoE

Same as Qwen3 Dense but with a Mixture-of-Experts FFN layer. `decoder_sparse_step` controls how often MoE layers appear.

- `config.json` `architectures`: `Qwen3MoeForCausalLM`
- GPU only (MoE routing requires CUDA)

### Qwen3.5 (Hybrid)

Alternates between Gated DeltaNet layers and standard attention layers every `full_attention_interval` layers (default: 4). Dense MLP throughout.

- Sizes: 0.8B, 2B, 4B, 9B, 27B (dense); 35B-A3B (MoE variant)
- `config.json` `model_type`: `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text`
- GPU only for native weights (CPU support requires GGUF)
- CUDA graph capture not supported (hybrid decode path)
- See [Hybrid Model Notes](#hybrid-model-notes) below

### Qwen3-Next (Hybrid + MoE)

Like Qwen3.5 but with an extreme-sparsity MoE FFN (512 experts, top-10 routing + 1 shared). 48 layers with `full_attention_interval=4`.

- `config.json` `architectures`: `Qwen3NextForCausalLM`
- GPU only; requires SM90+ (Hopper) for practical serving at 80B scale

### Gemma3

Dense transformer with sliding window attention on most layers and full attention every `sliding_window_pattern` layers (default: 6). Supports all three tasks.

- `config.json` `model_type`: `gemma3`, `gemma3_text`, `gemma2`, `gemma`
- Also handles Gemma2 (`gemma2` model type) via the same architecture spec

### Gemma4

Dense transformer with per-layer-type RoPE parameters. Generation only.

- `config.json` `model_type`: `gemma4`, `gemma4_text`, `gemma3n`

---

## Hybrid Model Notes

Qwen3.5 and Qwen3-Next interleave DeltaNet (linear recurrent attention) with standard attention layers. During decode, DeltaNet layers maintain a recurrent state instead of growing KV cache, giving O(1) per-token cost per layer.

**Concurrency is bounded by the DeltaNet state pool.** Each in-flight decode request occupies one slot:

```bash
PRELUDE_DELTANET_POOL_SLOTS=16  # default: 8
```

Increase this if you see requests being queued waiting for a slot.

**Limitations:**
- GPU only for native weights (requires FlashInfer or FA4)
- CUDA graph capture disabled on hybrid models
- GGUF available for Qwen3.5 (CPU + GPU)

---

## What Should I Use?

| I need... | Recommended |
|---|---|
| A chat model, fast | `Qwen/Qwen3-4B` |
| Best quality on a single GPU | `Qwen/Qwen3-32B` (80GB) |
| Best throughput on Hopper (H100/H200) | `Qwen/Qwen3-30B-A3B` (MoE — 3B active) |
| Classification / reranking | `Qwen/Qwen3-Reranker` |
| Text embeddings | `Qwen/Qwen3-Embedding` |
| CPU or resource-constrained | `Qwen/Qwen3-0.6B` or any Qwen3/Qwen3.5 GGUF |
| Largest scale | `Qwen/Qwen3-Next` (80B-A3B, requires Hopper) |

---

<!-- ## Known Limitations

- No multimodal support (vision, audio)
- No tool calling / function calling
- No LoRA adapter serving
- Single-GPU only (no tensor parallelism)
- No FP8 / INT8 quantization for native models (GGUF quantization only)
- GGUF models: generation only (`--task classify/embedding` not supported) -->

## Next Steps

- [Features](features.md) - key feature usage 