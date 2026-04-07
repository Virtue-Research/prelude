# Configuration

Complete reference for all CLI flags, environment variables, and API request parameters.

---

## CLI Flags

Start the server with:

```bash
./target/release/prelude-server [FLAGS]
```

### Model

| Flag | Default | Description |
|---|---|---|
| `--model <MODEL>` | `Qwen/Qwen3-0.6B` | HuggingFace repo ID (e.g. `Qwen/Qwen3-4B`) |
| `--model-path <PATH>` | — | Local directory (with `config.json` + safetensors + `tokenizer.json`) or a `.gguf` file. When set, `--model` is used as the reported model name only |
| `--task <TASK>` | `auto` | Force task detection: `auto`, `classify`, `embedding`, `generation` |
| `--dtype <DTYPE>` | auto | Override weight dtype: `f32` or `bf16`. Auto selects BF16 on GPU, F32 on CPU |

### Server

| Flag | Default | Description |
|---|---|---|
| `--host <HOST>` | `0.0.0.0` | Bind address |
| `--port <PORT>` | `8000` | HTTP port |
| `--api-key <KEY>` | — | Bearer token for auth (repeatable; also reads `PRELUDE_API_KEY`). No auth is enforced when empty |
| `--cors-allow-origin <ORIGIN>` | — | Allowed CORS origin (repeatable). No CORS headers are emitted unless at least one origin is set |

### Device & Memory

| Flag | Default | Description |
|---|---|---|
| `--device <DEVICE>` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` |
| `--gpu-memory-utilization <F>` | `0.4` | Fraction of free GPU memory to reserve for the paged KV cache (0.0–1.0). Ignored when `PRELUDE_PAGED_ATTN_BLOCKS` is set |
| `--cuda-graph <BOOL>` | `true` | Enable CUDA graph capture for decode steps |

### Scheduler

| Flag | Default | Description |
|---|---|---|
| `--max-batch-size <N>` | `32` | Max requests dispatched per model call |
| `--max-batch-wait-ms <MS>` | `5` | Max time to wait before dispatching a batch |
| `--max-running-requests <N>` | `8` | Max requests concurrently in-flight in the scheduler |
| `--max-prefill-tokens <N>` | `4096` | Max total prefill tokens per scheduling step |
| `--max-total-tokens <N>` | `32768` | Max total tokens (prompt + decode) across all running requests |
| `--decode-reservation-cap <N>` | `4096` | Per-request cap for reserving future decode slots in the scheduler |

#### Examples

**Interactive chat** — minimize latency, small wait time:
```bash
./target/release/prelude-server \
  --model Qwen/Qwen3-4B \
  --max-batch-wait-ms 1 \
  --max-batch-size 16 \
  --max-running-requests 16
```

**Batch / throughput workload** — maximize GPU utilization:
```bash
./target/release/prelude-server \
  --model Qwen/Qwen3-4B \
  --max-batch-wait-ms 20 \
  --max-batch-size 64 \
  --max-running-requests 64 \
  --max-total-tokens 131072
```

**Long-context workload** (e.g. 32K prompts) — raise token budgets:
```bash
./target/release/prelude-server \
  --model Qwen/Qwen3-4B \
  --max-prefill-tokens 32768 \
  --max-total-tokens 65536 \
  --max-running-requests 4 \
  --decode-reservation-cap 8192
```

### Development / Testing

| Flag | Default | Description |
|---|---|---|
| `--pseudo` | `false` | Use mock engine — no model loaded, returns fake tokens |
| `--pseudo-latency-ms <MS>` | `25` | Simulated latency per token in mock engine |

---

## Environment Variables

All `PRELUDE_*` variables are parsed once at startup. CLI flags take precedence over environment variables where both exist.

### Device & Auth

| Variable | Default | Description |
|---|---|---|
| `PRELUDE_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`. Overridden by `--device` |
| `CUDA_VISIBLE_DEVICES` | (all) | Standard CUDA device visibility |
| `PRELUDE_API_KEY` | — | API key merged with any `--api-key` CLI args |

### KV Cache

| Variable | Default | Description |
|---|---|---|
| `PRELUDE_PAGED_BLOCK_SIZE` | `128`* | Tokens per paged KV block. Adjusted automatically for FA3 (128) and other backends (16). *Override only if you know what you're doing* |
| `PRELUDE_PAGED_ATTN_BLOCKS` | `0` | Explicit number of paged KV blocks. `0` = auto from `--gpu-memory-utilization` |
| `PRELUDE_PREFIX_CACHE_BLOCKS` | `0` | Prefix cache capacity in blocks. `0` = disabled |
| `PRELUDE_PREFIX_BLOCK_SIZE` | `64` | Tokens per prefix cache block |
| `PRELUDE_DELTANET_POOL_SLOTS` | `8` | Max concurrent DeltaNet state slots (for hybrid models). `0` = disabled |

### Sampling Defaults

Applied when a request does not specify the parameter.

| Variable | Default | Description |
|---|---|---|
| `PRELUDE_DEFAULT_TEMPERATURE` | `0.7` | Default sampling temperature |
| `PRELUDE_DEFAULT_TOP_P` | `1.0` | Default top-p nucleus sampling |
| `PRELUDE_DEFAULT_MAX_TOKENS` | `4096` | Default max new tokens when `max_tokens` is not in the request |

### Runtime Toggles

| Variable | Default | Description |
|---|---|---|
| `PRELUDE_CUDA_GRAPH_MAX_BS` | `32` | Max batch size for CUDA graph capture. Graphs are captured for powers of 2 up to this value |
| `PRELUDE_FUSED_KV_CACHE_WRITE` | `0` | Set `1` to enable fused QK-Norm + RoPE + KV cache write kernel |
| `PRELUDE_FORCE_VARLEN_PREFILL` | `0` | Set `1` to force variable-length prefill path even for uniform batches |
| `PRELUDE_SYNC_TIMING` | `0` | Set `1` to enable CUDA sync timing for profiling |
| `PRELUDE_NO_SCHEDULER` | `0` | Set `1` to bypass the scheduler (single-request mode; used for accuracy testing) |
| `PRELUDE_MOCK` | `0` | Set `1` to use mock engine — equivalent to `--pseudo` |
| `SGLANG_CPU_OMP_THREADS_BIND` | — | CPU thread IDs for NUMA-aware binding (e.g. `0-7`) |
| `RUST_LOG` | `info` | Log level. Example: `prelude_core=debug,tower_http=warn` |

### Adaptive Scheduler (advanced)

EWMA parameters for the adaptive batch scheduler. Defaults are tuned for most workloads.

| Variable | Default | Description |
|---|---|---|
| `PRELUDE_ADAPTIVE_ARRIVAL_ALPHA` | `0.5` | EWMA smoothing for arrival rate. Higher = more responsive to bursts |
| `PRELUDE_ADAPTIVE_GPU_ALPHA` | `0.4` | EWMA smoothing for GPU time. Lower = more stable estimates |
| `PRELUDE_ADAPTIVE_INITIAL_LAMBDA` | `1000.0` | Initial arrival rate assumption (req/s) for cold start |
| `PRELUDE_ADAPTIVE_MAX_RATE` | `10000.0` | Maximum instantaneous rate before clamping |

### Build-time Variables

These are read during `cargo build`, not at runtime.

| Variable | Default | Description |
|---|---|---|
| `PRELUDE_FA4_ARCHS` | (detected) | Comma-separated SM architectures for FA4 AOT compilation (e.g. `sm_80,sm_90`). Auto-detected from `nvidia-smi` if unset |
| `PRELUDE_FA4_WORKERS` | (CPU count, max 8) | Parallel workers for FA4 kernel compilation |
| `PRELUDE_FLASHINFER_ARCHS` | `sm_80,sm_90` | SM architectures for FlashInfer AOT compilation |
| `PRELUDE_FLASHINFER_HEAD_DIMS` | `64,96,128,192,256,512` | Head dimensions to compile FlashInfer variants for |
| `PRELUDE_FLASHINFER_DTYPES` | `bf16,fp16` | Dtypes to compile FlashInfer variants for |
| `PRELUDE_FLASHINFER_WORKERS` | (CPU count) | Parallel workers for FlashInfer kernel compilation |
| `FLASHINFER_SRC` | — | Override FlashInfer source path (skips `third_party/flashinfer` submodule) |

---

## API Reference

For endpoint details, request parameters, and response formats, see the [API Reference](api.md).

## Next Steps

- [Supported Models](supported-models.md) — model compatibility and GGUF support
- [Features](features.md) - key feature usage 