# Server Configuration Reference

## CLI Flags

### Server

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--host` | string | `0.0.0.0` | Bind address |
| `--port` | u16 | `8000` | Bind port |
| `--model` | string | `Qwen/Qwen3-0.6B` | HF Hub repo ID or model name |
| `--model-path` | string | none | Local path to model directory or `.gguf` file (bypasses HF Hub) |
| `--task` | enum | `auto` | Force model task detection: `auto`, `classify`, `embedding`, `generation` |
| `--dtype` | string | none | Override dtype: `f32` or `bf16`. Default: bf16 for CPU, auto for GPU |
| `--cors-allow-origin` | string | none | Allowed CORS origin (repeatable). No CORS headers emitted unless set |

### Batching / Scheduler

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-batch-size` | usize | `32` | Max requests per batched model call |
| `--max-batch-wait-ms` | u64 | `5` | Max wait time (ms) before dispatching a batch |
| `--max-running-requests` | usize | `8` | Max concurrent running requests in the scheduler |
| `--max-prefill-tokens` | usize | `4096` | Max prefill tokens per scheduling step |
| `--max-total-tokens` | usize | `32768` | Max total tokens across all running requests |
| `--decode-reservation-cap` | usize | `4096` | Per-request cap for reserving future decode tokens |

### Memory / Cache

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gpu-memory-utilization` | f32 | `0.4` | Fraction of free GPU memory for KV cache (0.0--1.0). Ignored when `PRELUDE_PAGED_ATTN_BLOCKS` is set |

### Authentication

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--api-key` | string | none | API key for authentication (repeatable). Merged with `PRELUDE_API_KEY` env var. When set, `/v1/*` routes require `Authorization: Bearer <key>`; `/health` is always open |

### Mock / Debug

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--pseudo` | bool | `false` | Use mock engine (no model loaded) |
| `--pseudo-latency-ms` | u64 | `25` | Simulated per-token latency in mock mode |

---

## Environment Variables

### Device

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PRELUDE_DEVICE` | string | `auto` | Device selection: `auto`, `cpu`, `cuda`, `cuda:N` |
| `SGLANG_CPU_OMP_THREADS_BIND` | string | none | CPU IDs for NUMA-aware thread binding |

### Cache

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PRELUDE_PAGED_BLOCK_SIZE` | usize | `128` (flash-attn-v3) / `16` (other) | Tokens per paged KV cache block |
| `PRELUDE_PAGED_ATTN_BLOCKS` | usize | `0` (auto) | Explicit paged attention block count. 0 = derive from `--gpu-memory-utilization` |
| `PRELUDE_PREFIX_CACHE_BLOCKS` | usize | `0` (disabled) | Max cached prefix blocks. 0 = prefix cache disabled |
| `PRELUDE_PREFIX_BLOCK_SIZE` | usize | `64` | Tokens per prefix cache block |
| `PRELUDE_DELTANET_POOL_SLOTS` | u32 | `8` | Max concurrent DeltaNet recurrent state slots (hybrid models) |

### Sampling Defaults

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PRELUDE_DEFAULT_TEMPERATURE` | f32 | `0.7` | Default sampling temperature when not specified by request |
| `PRELUDE_DEFAULT_TOP_P` | f32 | `1.0` | Default top-p when not specified by request |
| `PRELUDE_DEFAULT_MAX_TOKENS` | u32 | `4096` | Default max new tokens when not specified by request |

### Performance

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PRELUDE_FUSED_KV_CACHE_WRITE` | bool (`1`) | off | Enable fused K-Norm + RoPE + KV paged cache write kernel (saves 1 kernel launch per layer) |
| `PRELUDE_FORCE_VARLEN_PREFILL` | bool | off | Force variable-length prefill path even when all sequences are same length |
| `PRELUDE_ADAPTIVE_ARRIVAL_ALPHA` | f64 | `0.5` | EWMA smoothing factor for arrival rate (higher = more responsive) |
| `PRELUDE_ADAPTIVE_GPU_ALPHA` | f64 | `0.4` | EWMA smoothing factor for GPU time (lower = more stable) |
| `PRELUDE_ADAPTIVE_INITIAL_LAMBDA` | f64 | `1000.0` | Initial arrival rate assumption (req/s) for adaptive batch cold start |
| `PRELUDE_ADAPTIVE_MAX_RATE` | f64 | `10000.0` | Maximum instantaneous rate before clamping |

### Debug

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RUST_LOG` | string | `prelude_server=info,prelude_core=info,tower_http=info` | Logging verbosity filter (tracing-subscriber `EnvFilter` syntax) |
| `PRELUDE_SYNC_TIMING` | bool | off | Enable CUDA device sync for timing measurements |
| `PRELUDE_PROFILE` | bool | off | Enable profiling output |
| `PRELUDE_ATTN_PROFILE` | bool | off | Enable attention sub-section profiling |
| `PRELUDE_NO_SCHEDULER` | bool (`1`) | off | Bypass ScheduledEngine, use base Engine directly. Disables streaming. Debug only |

### Authentication

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PRELUDE_API_KEY` | string | none | API key, merged with `--api-key` CLI args |

### Build-time

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PRELUDE_FA4_ARCHS` | string | local GPU arch | Comma-separated SM architectures for FA4 AOT compilation (e.g., `sm_90,sm_120`) |
| `ONEDNN_FFI_DIR` | string | auto | Override oneDNN FFI build directory |

---

## Feature Flags

Build-time Cargo features defined in `prelude-core`. Pass via `--features` when building `prelude-server`.

| Feature | Implies | Description |
|---------|---------|-------------|
| `cuda` | `paged-attn` | GPU support via CUDA. Enables paged attention KV cache and fused CUDA kernels |
| `flash-attn` | `cuda` | Flash Attention V2 (Ampere+, SM80). GQA native, no varlen-paged (no prefix cache) |
| `flash-attn-v3` | `cuda` | Flash Attention V3 (Hopper, SM90). Full varlen-paged support including prefix cache |
| `flash-attn-v4` | `cuda` | Flash Attention V4 (CuTeDSL AOT, SM80+). Prefill only, no paged KV. Statically linked |
| `deepgemm` | `cuda` | DeepGEMM BF16 GEMM kernels replacing cuBLAS. SM90+, statically linked. Decode 17%--2x faster on H200 |
| `onednn` | none | oneDNN BF16 GEMM for CPU. Auto-built from source via CMake. Adds ~33MB to binary |

### Recommended combinations

| Target | Features |
|--------|----------|
| CPU (F32) | none |
| CPU (BF16) | `onednn` |
| GPU (Ampere) | `flash-attn` |
| GPU (Hopper) | `flash-attn-v4,flash-attn-v3` |
| GPU (Hopper, full) | `flash-attn-v4,flash-attn-v3,onednn` |
| GPU (Hopper, DeepGEMM) | `flash-attn-v4,flash-attn-v3,deepgemm` |

---

## Deployment Recipes

### Development (mock mode, fast iteration)

```bash
cargo run -p prelude-server -- --pseudo --pseudo-latency-ms 10 --port 8000
```

### Single-GPU Hopper (production)

```bash
cargo build -p prelude-server --release --features flash-attn-v4,flash-attn-v3

CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server \
  --model Qwen/Qwen3-4B \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-batch-size 64 \
  --max-running-requests 32 \
  --api-key sk-prod-key
```

### Single-GPU Ampere (production)

```bash
cargo build -p prelude-server --release --features flash-attn

CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server \
  --model Qwen/Qwen3-4B \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-batch-size 64
```

### CPU-only (BF16)

```bash
cargo build -p prelude-server --release --features onednn

PRELUDE_DEVICE=cpu ./target/release/prelude-server \
  --model Qwen/Qwen3-0.6B \
  --port 8000 \
  --max-batch-size 16
```

### High-throughput classification / embedding

```bash
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server \
  --model Qwen/Qwen3-Reranker \
  --task classify \
  --port 8000 \
  --max-batch-size 128 \
  --max-batch-wait-ms 10 \
  --gpu-memory-utilization 0.8
```
