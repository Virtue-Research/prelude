# Features

## Continuous Batching

Prelude uses continuous batching: requests join and leave the batch dynamically at each decode step, rather than waiting for a fixed batch to complete. This maximizes GPU utilization under mixed load.

Two flags control batching behavior:

| Flag | Default | Description |
|---|---|---|
| `--max-batch-size` | `32` | Hard cap on requests per model forward pass |
| `--max-batch-wait-ms` | `5` | Max time to wait for more requests before dispatching |

Tune for your workload:

- **Latency-sensitive** (e.g. interactive chat): lower `--max-batch-wait-ms` to `1–2`
- **Throughput-optimized** (e.g. batch processing): raise `--max-batch-size` to `64–128`

### Adaptive batch scheduling

The scheduler also runs an EWMA-based adaptive layer that tracks arrival rate and observed GPU time per batch size, automatically tuning batch size and wait time within the hard caps above. This works out of the box — no configuration needed. Advanced knobs are in [Configuration](configuration.md#adaptive-scheduler-advanced).

---

## Paged KV Cache

KV cache memory is managed in fixed-size blocks (paged attention), similar to OS virtual memory paging. This eliminates memory fragmentation and allows sequences of different lengths to coexist efficiently.

### GPU memory allocation

Control how much GPU memory is reserved for the KV cache:

```bash
./target/release/prelude-server \
  --model Qwen/Qwen3-4B \
  --gpu-memory-utilization 0.85
```

`--gpu-memory-utilization` sets the fraction of **free** GPU memory to reserve for KV cache blocks (default `0.4`). Increase it to serve more concurrent requests — just leave headroom for model weights and activations.

For precise control, override the block count directly:

```bash
PRELUDE_PAGED_ATTN_BLOCKS=2048 ./target/release/prelude-server --model Qwen/Qwen3-4B
```

### Block size

The default block size is `128` tokens (adjusted automatically for FA4/FlashInfer backends). Override only if you have a specific reason:

```bash
PRELUDE_PAGED_BLOCK_SIZE=64 ./target/release/prelude-server --model Qwen/Qwen3-4B
```

---

## Prefix Caching

Prefix caching reuses the computed KV cache for shared prompt prefixes across requests — for example, a fixed system prompt or few-shot examples. On a cache hit, only the unique suffix needs to be prefilled.

Enable by setting the cache capacity in blocks:

```bash
PRELUDE_PREFIX_CACHE_BLOCKS=512 ./target/release/prelude-server \
  --model Qwen/Qwen3-4B
```

| Variable | Default | Description |
|---|---|---|
| `PRELUDE_PREFIX_CACHE_BLOCKS` | `0` | KV blocks reserved for prefix cache. `0` = disabled |
| `PRELUDE_PREFIX_BLOCK_SIZE` | `64` | Tokens per prefix cache block |

The cache uses a hash-trie index with LRU eviction. Blocks are ref-counted and shared between active sequences and the prefix cache, so they're only freed when no longer needed by either.

**When to use it:** Most effective when many requests share a long common prefix (system prompt, documents, few-shot examples). Less useful for highly diverse prompts.

---

## CUDA Graph Decode

CUDA graph capture records the sequence of GPU operations for a decode step and replays them with minimal CPU overhead. This reduces per-token decode latency at low batch sizes (the most common case for interactive serving).

Enabled by default. To disable (e.g. for debugging):

```bash
./target/release/prelude-server --model Qwen/Qwen3-4B --cuda-graph false
```

Graphs are captured at startup for every batch size from 1 up to `PRELUDE_CUDA_GRAPH_MAX_BS` (default: `32`). Control the upper bound:

```bash
PRELUDE_CUDA_GRAPH_MAX_BS=64 ./target/release/prelude-server --model Qwen/Qwen3-4B
```

!!! note
    CUDA graph capture is automatically disabled for hybrid models (Qwen3.5, Qwen3-Next) due to the variable DeltaNet recurrent state.

---


## Logprobs

Prelude can return log probabilities for generated tokens (output logprobs) and prompt tokens (prompt logprobs, a vLLM extension).

### Output logprobs (chat completions)

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Hello"}],
    logprobs=True,
    top_logprobs=5,  # return top-5 candidates per token
)
for token in response.choices[0].logprobs.content:
    print(token.token, token.logprob, token.top_logprobs)
```

### Output logprobs (text completions)

```python
response = client.completions.create(
    model="Qwen/Qwen3-4B",
    prompt="The capital of France is",
    logprobs=5,  # return top-5 candidates per token
)
```

### Prompt logprobs (text completions only)

```python
response = client.completions.create(
    model="Qwen/Qwen3-4B",
    prompt="The capital of France is Paris",
    prompt_logprobs=1,  # return top-1 logprob per prompt token
)
print(response.choices[0].prompt_logprobs)
```

---

## Attention Backends

Prelude compiles multiple attention kernels at build time and dispatches to the best available one at runtime:

**NVIDIA (CUDA)** — dispatch priority:

| Priority | Backend | Requirement | Best for |
|---|---|---|---|
| 1 | FA4 (CuTeDSL) | SM90+ | Prefill throughput |
| 2 | FlashInfer | SM80+ | Decode (CUDA graph compatible) |
| 3 | CPU fallback | — | No GPU |

**AMD (ROCm):** Uses HIP attention kernels via `prelude-rocm` (built with `--features rocm`).

**Apple Silicon (Metal):** Uses Metal compute shaders via `prelude-metal` (built with `--features metal`).

No configuration needed — the best kernel is selected automatically based on your device. See [Getting Started](setup.md) for build flag details.

---

## GEMM Backends

Matrix multiplication dispatch follows a similar priority:

| Backend | Requirement | Notes |
|---|---|---|
| DeepGEMM | SM90+ (H100/H200) | Up to 2× faster than cuBLAS |
| cuBLAS | Any CUDA GPU | Default NVIDIA GEMM |
| ROCm (CK/aiter) | AMD GPU | Enabled with `--features rocm` |
| Metal (simdgroup) | Apple Silicon | Enabled with `--features metal` |
| oneDNN | x86_64 CPU | BF16 GEMM, statically linked |
| Built-in | Any CPU | F32 fallback, no extra deps |

Enable DeepGEMM at build time (NVIDIA SM90+):

```bash
cargo build -p prelude-server --release --features flashinfer-v4,onednn,deepgemm
```

---

## Fused CUDA Kernels

Prelude includes fused kernels for common operation sequences, reducing memory bandwidth and kernel launch overhead:

- **QK-Norm + RoPE + KV cache write** — fuses Q/K normalization, rotary positional embedding, and KV cache scatter into a single kernel (enable with `PRELUDE_FUSED_KV_CACHE_WRITE=1`)
- **SiLU × Mul** — fused gate activation for FFN layers
- **Add + RMSNorm** — fused residual add and layer normalization

These are applied automatically when the CUDA backend is active.

---

## Quantization

Prelude supports quantized inference through the GGUF format. Prelude does not support quantization of native models at this point.


### Supported Formats

All standard GGML quantization types are supported:

| Format | Bits per weight | Notes |
|---|---|---|
| `Q2_K` | ~2.6 | Lowest quality; smallest footprint |
| `Q3_K` (`Q3_K_M`, `Q3_K_S`, `Q3_K_L`) | ~3.4 | |
| `Q4_0` | 4.5 | Simple 4-bit; fast on CPU |
| `Q4_1` | 5.0 | 4-bit with min-max scaling |
| `Q4_K` (`Q4_K_M`, `Q4_K_S`) | 4.5 | K-quant; recommended default |
| `Q5_0` | 5.5 | |
| `Q5_1` | 6.0 | |
| `Q5_K` (`Q5_K_M`, `Q5_K_S`) | 5.5 | |
| `Q6_K` | ~6.6 | Near-lossless |
| `Q8_0` | 8.5 | Highest quality; largest GGUF |
| `F16` / `BF16` / `F32` | 16 / 16 / 32 | Unquantized GGUF tensors |

**Recommended:** `Q4_K_M` for a good quality/size trade-off. `Q8_0` if you have the VRAM and want near-bf16 quality.

**Not supported:** IQ formats (`IQ2XXS`, `IQ3XXS`, etc.), FP8, INT8, AWQ, GPTQ. Native (non-GGUF) models always load in their original dtype (`bf16` on GPU, `f32` on CPU).

### Loading a Quantized Model

```bash
# Local GGUF file
./target/release/prelude-server \
  --model-path /data/models/Qwen3-4B-Q4_K_M.gguf \
  --model Qwen/Qwen3-4B

# HuggingFace Hub GGUF repo
./target/release/prelude-server --model unsloth/Qwen3-4B-GGUF
```

`--model` sets the name reported in API responses. When a Hub repo contains multiple GGUF files, Prelude auto-selects using the preference order: **Q8_0 → Q4_K_M → first available file**.

### Device Support

GGUF models run on both CPU and GPU with the same binary:

| Device | Kernel |
|---|---|
| GPU (SM80+) | Tiled MMQ (prefill) + MMVQ (decode); quantized blocks stay on device |
| CPU | AVX2/NEON quantized matmul; activations dynamically quantized to Q8 |

### Interaction with `--dtype`

`--dtype` is **ignored for GGUF files** — the quantization format is fixed by the file itself. It only applies to native safetensor models.

---

## Advanced features

Prelude supports non-AG decoding (e.g., difussion) and other features such as AF disaggregation, speculative decoding, KV scheduling is on the road map.

---


## Observability

### Per-request metrics

Every completed request logs structured metrics via `tracing`:

```
prompt_tokens=128 completion_tokens=64 ttft_ms=12.3 decode_ms=48.7 total_ms=61.0 decode_tps=1313.1 finish_reason=stop
```

### Log levels

Control verbosity with `RUST_LOG`:

```bash
# Verbose engine internals
RUST_LOG=prelude_core=debug ./target/release/prelude-server --model Qwen/Qwen3-4B

# Quiet HTTP layer
RUST_LOG=prelude_core=info,tower_http=warn ./target/release/prelude-server --model Qwen/Qwen3-4B
```

### NVTX profiling

Build with `nvtx` to annotate GPU kernels for Nsight Systems:

```bash
cargo build -p prelude-core --release --features nvtx
nsys profile ./target/release/prelude-server --model Qwen/Qwen3-4B
```

---
