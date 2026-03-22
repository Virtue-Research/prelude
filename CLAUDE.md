# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What's next

Unified backend refactor — see `docs/unified-backend-refactor.md` for full design doc and `docs/architecture.html` for visual diagram. Key changes:
- Two schedulers only: Batch (max_new=1) and Continuous Batch (max_new>1, paged required)
- Both emit `GpuPacket`s into a single FIFO GPU queue consumed by one GPU worker
- Serial model forward: prefix cache → varlen → conditional paged KV → post-processing
- Remove: concat KV path, non-paged fallback, CUDA graph decode, multiple forward entry points

## Notes

See `notes.md` (untracked, not committed) for development notes, benchmark caveats, and performance history. Do not
commit or gitignore it.
Always record things there or here after you have talked with users for finishing something. Your last todo in the plan
should always update docs

## Prerequisites

- **Rust** (stable toolchain)
- **CUDA Toolkit** (for GPU features)
- **CMake** ≥ 3.18 — required to build `crates/onednn-ffi`
- **oneDNN** — auto-downloaded and built from source on first `cargo build --features onednn`. No manual setup needed.
  Static linking (~33MB added to binary). Override FFI dir with `ONEDNN_FFI_DIR` env var.

## Build

```bash
cargo build -p prelude-server --release                                          # CPU only
cargo build -p prelude-server --release --features onednn                        # CPU + oneDNN BF16 GEMM
cargo build -p prelude-server --release --features flashinfer-v4,onednn,deepgemm # Full GPU (recommended)
cargo build -p prelude-server --profile dist --features flashinfer-v4,onednn,deepgemm  # Distribution (smallest binary)
```

**Recommended GPU config**: `flashinfer-v4,onednn,deepgemm` — FlashInfer (attention) + FA4 (prefill) + DeepGEMM (GEMM) + oneDNN (CPU BF16).
Binary size: ~98MB (`--release`) or ~98MB (`--profile dist` with fat LTO).

**Build profiles**:
- `--release` — thin LTO, fast incremental builds (~27s)
- `--profile dist` — fat LTO + codegen-units=1 + panic=abort, smallest binary (~3.5min). Output in `target/dist/`

### Feature flags

Feature flags (defined in `prelude-core`): `cuda`, `flash-attn`, `flash-attn-v3`, `flash-attn-v4`, `flashinfer`, `flashinfer-v4`, `deepgemm`, `onednn`.

| Feature | What it does | Implies |
|---|---|---|
| `flashinfer` | FlashInfer AOT attention (FA2 SM80+ / FA3 SM90+) | `cuda` |
| `flash-attn-v4` | FA4 CuTeDSL AOT attention (SM80+) | `cuda` |
| `flashinfer-v4` | FlashInfer + FA4 combined (shared tvm_ffi) | `flashinfer`, `flash-attn-v4` |
| `deepgemm` | DeepGEMM BF16 GEMM (replaces cuBLAS, SM90+) | `cuda` |
| `onednn` | oneDNN BF16 GEMM for CPU | — |
| `flash-attn-v3` | FA3 Hopper attention (legacy, replaced by FlashInfer) | `cuda` |
| `flash-attn` | FA2 Ampere attention (legacy) | `cuda`, `paged-attn` |
| `cuda` | GPU fused ops + paged KV infrastructure | — |

**Attention dispatch priority**: FA4 → FlashInfer → FA3 → FA2 → CPU. With `flashinfer-v4`, FA4 handles prefill,
FlashInfer handles CUDA graph decode (32 graphs, no seqlen bucketing) and serves as fallback.

**FlashInfer AOT kernels**: 128 variants (head_dim 64/96/128/192/256 × BF16/FP16 × swa × softcap).
Auto-compiles if no `.o` files exist (needs Python + CUDA GPU + FlashInfer source at `FLASHINFER_SRC`).
Pre-compile with: `python3 crates/prelude-flashinfer/scripts/compile_kernels.py -j 8`

**FA4 kernels**: ~400 variants per arch, statically linked. Multi-arch: `PRELUDE_FA4_ARCHS=sm_90,sm_120`.
Pre-compile with: `python3 crates/prelude-flash-attn-v4/scripts/compile_kernels.py -j 8`

`deepgemm` uses DeepGEMM BF16 GEMM kernels (replaces cuBLAS for matmul). SM90+, JIT compiled at runtime.
On H200: decode 17%~2x faster than cuBLAS, prefill parity. See `crates/prelude-deepgemm/README.md`.

Note: workspace uses `[patch.crates-io]` to override `candle-core` with a local copy at `crates/candle-core/`. This
directory is gitignored and must exist for compilation (symlink or copy of candle-core source).

## Test

```bash
cargo test -p prelude-core                    # All core unit tests (CPU)
cargo test -p prelude-core -- prefix_cache     # Single test module
cargo test -p prelude-core -- block_manager    # Another module
cargo test -p prelude-core -- scheduler        # Scheduler tests
```

Tests are in-source (`#[cfg(test)]` modules) in `block_manager.rs`, `prefix_cache.rs`, `scheduler.rs`.

### Accuracy Tests (Numerical Precision Regression)

Accuracy tests generate reference outputs on-the-fly using HuggingFace transformers, then compare against prelude.

**Pass criteria** (vLLM-style):
1. L0: Exact text match → PASS
2. L1: At first token divergence, bidirectional cross-containment in top-5 → PASS

**Files:**

- `tests/accuracy/golden/prompts.json` — 8 test prompts (short/long/multilingual/code/completion)
- `tests/accuracy/run_accuracy_test.py` — Test runner: generates reference, starts server, compares output + logprobs
- `tests/accuracy/generate_golden.py` — Optional standalone golden data generator (for debugging)

**Run accuracy test locally:**

```bash
# Build server first
cargo build -p prelude-server --release

# Set up test venv (one-time)
uv venv .venv --seed && uv pip install --python .venv/bin/python \
    transformers torch requests numpy

# Run test
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-f32 \
    --server prelude --binary target/release/prelude-server \
    --model Qwen/Qwen3-0.6B
```

**Against a pre-started server:**

```bash
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-f32 \
    --server prelude=http://localhost:8001 --model Qwen/Qwen3-0.6B
```

**Text-only mode** (skip logprob comparison):

```bash
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-f32 \
    --server prelude --binary target/release/prelude-server \
    --model Qwen/Qwen3-0.6B --no-logprobs
```

**Test oneDNN BF16 path** (needs `--features onednn`):

```bash
cargo build -p prelude-server --release --features onednn
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-bf16 \
    --server prelude --binary target/release/prelude-server \
    --model Qwen/Qwen3-0.6B
```

**How it works:**

1. Loads model in transformers with same dtype as variant, greedy decodes all prompts to get reference tokens + logprobs
2. Starts prelude server (uses ScheduledEngine with CPU continuous runtime for streaming decode)
3. Sends requests with `logprobs=5` to get per-token log probabilities
4. Comparison (vLLM-style):
    - **L0 Text match**: exact output text equality → PASS
    - **L1 Cross-containment**: at first diverging token, both directions must be in the other's top-5 → PASS
    - Logprob diff reported as informational metric

**Supported variants:**

| Variant     | Build flags                             | Dtype | Requirements     |
|-------------|-----------------------------------------|-------|------------------|
| `cpu-f32`   | none                                    | F32   | any (CI default) |
| `cpu-bf16`  | `--features onednn`                     | BF16  | oneDNN (auto-built) |
| `gpu`       | `--features flashinfer-v4` + GPU        | BF16  | CUDA GPU (SM80+) |

### Qwen3-Next Accuracy Test

Dedicated accuracy test for the Qwen3-Next hybrid model (80B total / 3B active). Compares against HF transformers
golden reference using the same L0/L1 criteria.

```bash
# Step 1: Generate golden reference (one-time, ~2 min to load 80B model)
.venv/bin/python tests/qwen3_next/test_accuracy.py --generate-golden \
    --model <MODEL_PATH>

# Step 2a: Test with pre-started server
.venv/bin/python tests/qwen3_next/test_accuracy.py --server-url http://localhost:8001

# Step 2b: Auto-start server and test
.venv/bin/python tests/qwen3_next/test_accuracy.py --model <MODEL_PATH>
```

**Current results (cpu-f32):** 6/6 passed (exact=3, close=3). Close results are chat prompts where the only
difference is the trailing `<|im_end|>` stop token (stripped by API, included in HF reference).

## Run

```bash
# GPU server
PRELUDE_DEVICE=auto cargo run -p prelude-server --release --features cuda -- \
  --host 0.0.0.0 --port 8001 --model Qwen/Qwen3-0.6B \
  --max-batch-size 32 --max-batch-wait-ms 5

# CPU server (F32, default without onednn feature)
PRELUDE_DEVICE=cpu cargo run -p prelude-server --release -- \
  --host 0.0.0.0 --port 8001 --model Qwen/Qwen3-0.6B

# CPU server (BF16, requires onednn for BF16 GEMM)
PRELUDE_DEVICE=cpu cargo run -p prelude-server --release --features onednn -- \
  --host 0.0.0.0 --port 8001 --model Qwen/Qwen3-0.6B

# Mock mode (no model needed)
PRELUDE_MOCK=1 cargo run -p prelude-server --release

# With API key authentication
cargo run -p prelude-server --release -- --api-key sk-my-secret --model Qwen/Qwen3-0.6B
```

Server CLI arguments:

- `--host`, `--port` — network binding (default `127.0.0.1:8000`)
- `--model` — HF Hub repo ID or local path to model dir / `.gguf` file (auto-detected)
- `--max-batch-size`, `--max-batch-wait-ms` — dynamic batching tuning
- `--max-running-requests`, `--max-prefill-tokens`, `--max-total-tokens` — scheduler memory budgets
- `--dtype` — override dtype: `f32` or `bf16` (default: auto — GPU uses BF16, CPU uses F32 unless onednn provides BF16 GEMM)
- `--api-key` — API key for authentication (repeatable). Also reads `PRELUDE_API_KEY` env var. When set, `/v1/*` routes require `Authorization: Bearer <key>` header; `/health` is always open
- `--cuda-graph` — enable CUDA graph capture for decode steps (also `PRELUDE_CUDA_GRAPH=1`)

Key environment variables:

- `PRELUDE_API_KEY=<key>` — API key (merged with `--api-key` CLI args)
- `PRELUDE_MOCK=1` — use mock engine (no model needed)
- `PRELUDE_MOCK_LATENCY_MS=<int>` — simulated latency in mock mode (default 25)
- `PRELUDE_NO_SCHEDULER=1` — bypass `ScheduledEngine`, use `Engine` directly (debug only, disables streaming)
- `PRELUDE_DEVICE=auto|cpu|cuda|cuda:N` — device selection
- `PRELUDE_PAGED_BLOCK_SIZE=<int>` — paged block size (default 128 with flash-attn-v3, 16 otherwise)
- `PRELUDE_PREFIX_CACHE_BLOCKS=<int>` — max cached prefix blocks (0 = disabled)
- `PRELUDE_PREFIX_BLOCK_SIZE=<int>` — tokens per prefix block (default 64)
- `PRELUDE_FUSED_KV_CACHE_WRITE=1` — fused K-norm + RoPE + KV paged cache write (saves 1 kernel launch per layer)
- `PRELUDE_CUDA_GRAPH=1` — enable CUDA graph capture for decode (reduces kernel launch overhead)
- `PRELUDE_CUDA_GRAPH_MAX_BS=32` — max batch size for CUDA graph capture (default 32)
- `PRELUDE_SYNC_TIMING=1` — enable CUDA device sync for timing
- `RUST_LOG=prelude_core=debug` — logging verbosity

## Architecture

### Engine Hierarchy

```
InferenceEngine (trait: engine/mod.rs)
├── PseudoEngine (engine/pseudo.rs)       — mock engine for testing, no real inference
├── Engine (engine/engine_struct.rs)      — real inference via Candle tensors, generation/classification/embedding
└── ScheduledEngine (runtime/engine.rs)   — wraps Engine with dynamic batching
```

#### Engine module layout (`engine/`)

| File | Content |
|------|---------|
| `mod.rs` | `InferenceEngine` trait, `EngineError` enum, re-exports |
| `pseudo.rs` | `PseudoEngine` (mock engine) |
| `engine_struct.rs` | `Engine` struct, `ModelExecutor`, accessors, `InferenceEngine` impl |
| `plan_types.rs` | All plan/batch/cache type definitions |
| `config_parse.rs` | Model config parsing (`load_model_config`, `resolve_model_type`) |
| `weights.rs` | Safetensor/weight loading |
| `device.rs` | Device/dtype selection, CPU runtime init |
| `load.rs` | Engine construction (local/hub/gguf) + model builder |
| `planner.rs` | Prefix reuse, cache allocation planning |
| `tokenize.rs` | Tokenize helpers (`tokenize_prompt_input`, `tokenize_batch_inputs`) |
| `forward/prefill.rs` | Unified prefill pipeline for all task types (`flash-attn-v3` only) |
| `forward/generate.rs` | Generation execution + post-processing |
| `forward/classify.rs` | Classification execution + post-processing (`flash-attn-v3` only) |
| `forward/embed.rs` | Embedding execution + post-processing (`flash-attn-v3` only) |
| `forward/stubs.rs` | Stub types and error returns when `flash-attn-v3` absent |
| `forward/paged_prefill.rs` | `batch_prefill_paged` (`flash-attn-v3`) |
| `forward/paged_decode.rs` | Paged decode + stream decode (`flash-attn-v3`) |

#### Runtime module layout (`runtime/`)

| File | Content |
|------|---------|
| `mod.rs` | Re-exports from scheduler |
| `engine.rs` | `ScheduledEngine` + `InferenceEngine` impl |
| `request_state.rs` | Request state types + scheduler message enums |
| `batch_runtime.rs` | Batch scheduler loop with `BatchRuntimeQueues` |
| `continuous_runtime.rs` | Continuous generation loop |
| `gpu_queue.rs` | GPU packet queue + worker |
| `adaptive_batch.rs` | Adaptive batch sizing (EWMA) |

`ScheduledEngine` is the production engine. It wraps `Engine` and adds:

- Dynamic batching (configurable max batch size + wait time)
- Pipelined tokenization (overlaps CPU tokenization via rayon with GPU compute)
- Streaming request partitioning (streaming vs batch requests handled separately)
- Adaptive batch sizing for classification requests (EWMA-based)

### Scheduler (`scheduler.rs`)

Minimal continuous-batching scheduler (SGLang-inspired), separate from `ScheduledEngine`:

- **Sequence state machine**: `Waiting` → `Prefilling` → `Decoding` → `Finished` (with preemption)
- **Budget constraints**: `max_running_requests`, `max_prefill_tokens`, `max_total_tokens`
- **Preemption**: Retracts lowest-priority decode sequences to free KV when memory is tight
- **SchedulePolicy trait**: Pluggable waiting queue ordering (FCFS default)

### GPU Queue (`runtime/gpu_queue.rs`)

All GPU-bound work is serialized through a single FIFO queue consumed by one dedicated OS thread. Schedulers
(batch runtime, continuous runtime) produce `GpuPacket`s; the worker executes them sequentially.

```
All Requests → CPU Tokenization → GpuPacket → GPU Queue (FIFO) → GPU Worker → execute_gpu_packet()
```

**Packet variants** (`GpuPacket` enum):
- `GenerateBatch` — prefill-only generation (max_new=1), used by batch runtime
- `PrefillPaged` — varlen prefill + paged KV write, used by continuous runtime
- `DecodePaged` — batch decode with paged KV (Q=1 per seq), used by continuous runtime
- `ClassifyBatch` — classification forward pass
- `EmbedBatch` — embedding forward pass
The GPU worker runs on its own OS thread (not a tokio task) to avoid `spawn_blocking` overhead — critical for
decode steps where per-token GPU time is only a few milliseconds.

### CUDA Graph Decode (`runtime/cuda_graph.rs`)

Optional CUDA graph capture for decode (Q=1) steps, reducing kernel launch overhead. Enabled via
`--cuda-graph` or `PRELUDE_CUDA_GRAPH=1`. Graphs are captured eagerly at startup and replayed
during inference.

- **FlashInfer mode** (recommended): graph key is `(batch_size)` only — no seqlen bucketing needed
  (KV lengths via device tensors, not scalar kernel args). 32 graphs for max_bs=32, warmup ~750ms.
  Pre-allocated metadata buffers (`fi_indptr`, `fi_indices`, `fi_last_page_len`) in `DecodeGraphBuffers`
  ensure fixed GPU addresses across capture and replay (SGLang's `paged_kv_*_buffer` pattern).
- **FA3 mode** (legacy): graph key is `(batch_size, seqlen_bucket)` — 160 graphs, warmup ~2.5s.
  Seqlen > 4096 falls back to eager.
- **Pre-allocated buffers**: input tensors are fixed-size GPU buffers, updated via `memcpy_htod` before replay
- **Limitations**: dense models only (no DeltaNet/hybrid), decode only (not prefill)
- **Performance** (H200, Qwen3-0.6B BF16): ~1220 tps with CUDA graph vs ~966 tps eager (FlashInfer),
  ~1050 tps eager (FA3). CUDA graph eliminates per-layer CPU overhead and closes the gap.

### Unified Prefill Pipeline (`engine/prefill_pipeline.rs`)

`Engine::prefill_pipeline()` provides a unified forward path for **all task types**: classify, embed, and generation
(prefill-only, max_new=1). Returns a raw `Tensor` (`PrefillForwardResult`) — callers handle post-processing:

1. **Prefix cache lookup** — finds common prefix, block_size alignment check, paged cache match
2. **Token packing** — packs tokens into varlen format, skipping cached prefix
3. **Block allocation** — uses `build_cache_allocation_plan()` + `allocate_block_tables_from_plan()` from planner.rs
4. **Varlen forward** — `model.forward()` via `BatchAttnContext` (single entry point)
5. **Prefix cache populate** — on cache miss, inserts block tables for future reuse
6. **Block cleanup** — frees temporarily allocated paged blocks (ref-counted, cached blocks survive)

Post-processing per task type:
- **Classify**: `.to_dtype(F32).to_vec2()` → softmax → class predictions
- **Embed**: `.to_dtype(F32).to_vec2()` → embedding vectors
- **Generate (max_new=1)**: `.squeeze(1).argmax()` + logprob extraction on GPU tensor

### Model Forward Modes

All models implement a single `ModelForward::forward()` trait method. The execution mode is determined by
`BatchAttnContext` fields, not by separate methods:

- **Varlen prefill** — `paged_kv: None` → pure flash-attn varlen (classify/embed/prefill-only)
- **Varlen + paged KV** — `paged_kv: Some(PagedKvBatchContext)` → varlen + write KV to paged cache
- **Paged decode** — Q=1 per sequence with paged KV read/write (continuous batch decode)
- **Standard** — single-sequence, concat KV cache fallback (CPU-only, no flash-attn)

The varlen path uses `PagedKvContext`/`PagedKvBatchContext` structs (defined in `models/layers/mod.rs`) to optionally
enable paged prefix cache: when `paged_kv` is `None`, it does pure prefill; when `Some`, it writes KV to paged cache
and uses paged attention.

### Attention Backend Abstraction (`models/layers/attn/`)

Attention dispatch is split into per-backend modules under `attn/`. This is the **only** place with
attention backend `#[cfg]` gates. Model architectures have zero `flash-attn` / `paged-attn` gates.

```
models/layers/attn/
  mod.rs         — dispatch: FA4 → FlashInfer → FA3 → FA2 → CPU (single cfg location)
  flashinfer.rs  — FlashInfer AOT wrappers (FA2 SM80+ / FA3 SM90+, plan caching, CUDA graph)
  flash_v4.rs    — FA4 CuTeDSL wrappers (SM80+, prefill + paged decode)
  flash_v3.rs    — FA3 wrappers (Hopper, legacy)
  flash_v2.rs    — FA2 wrappers (Ampere+, legacy)
  paged.rs       — paged cache ops (scatter_kv_cache_flash PTX kernel, FA2 v1 layout ops)
  cpu.rs         — CPU matmul SDPA + BF16 tiled attention
```

| Dispatch function | Purpose |
|---|---|
| `varlen_attention()` | Causal varlen + optional paged KV write+read |
| `varlen_attention_bidirectional()` | Non-causal varlen attention |
| `varlen_attention_windowed()` | Sliding window attention (Gemma3) |
| `varlen_attention_paged()` | Paged attention read-only (fused paths) |
| `reshape_and_cache()` | Write K/V to paged cache (flash or v1 layout) |

Backend priority: FA4 → FlashInfer → FA3 → FA2 → CPU. GQA handled natively by all backends.

**FlashInfer**: Complete attention backend covering all dispatch paths. SM90+ uses FA3 tensor core
kernel for both prefill and decode (SGLang's `use_tensor_cores` approach). Plan caching eliminates
27/28 per-step plan calls. CUDA graph support with pre-allocated metadata buffers (32 graphs, no
seqlen bucketing, 3.6x faster warmup than FA3's 160 graphs).

**Paged KV cache write**: uses custom vectorized PTX kernel (`scatter_kv_cache_flash` in
`ops/gpu/kv_cache.rs`), independent of `candle-paged-attn`. FA2 v1 layout ops still use
`candle-paged-attn` (gated behind `paged-attn` feature, only pulled in by `flash-attn`).

Feature flag separation:
- `flashinfer` / `flash-attn-v4` / `flash-attn-v3` / `flash-attn` — attention backend → `attn/mod.rs` only
- `cuda` — GPU fused ops + paged KV infrastructure
- `onednn` — CPU GEMM backend → `models/layers/linear.rs` type alias

### KV Cache

Paged KV cache with `BlockManager` (vLLM-style memory management). Block size auto-adjusted for
flash-attn-v3 kernel compatibility. Override with `PRELUDE_PAGED_BLOCK_SIZE`.

### Prefix Cache (`prefix_cache.rs`)

Hash-trie structure for automatic prompt prefix caching:

- Matches incoming prompts against cached token blocks using hash chains
- LRU eviction of leaf blocks
- `AssembledKvCache` avoids repeated `Tensor::cat` for the same prefix chain
- Integrates with paged KV cache (optional paged block IDs) via ref-counted `BlockManager`

### Server (`prelude-server`)

Split into `lib.rs` (backend-agnostic library) and `main.rs` (binary entrypoint with concrete engine wiring).

- **`lib.rs`** — exports `Server` and `build_router(engine, api_keys) -> Router`. Only depends on the
  `InferenceEngine` trait; has zero knowledge of Engine, ScheduledEngine, or any concrete backend.
  Any `Arc<dyn InferenceEngine>` implementation can be plugged in.
- **`main.rs`** — CLI parsing, `build_engine()` (constructs Engine/ScheduledEngine/PseudoEngine),
  tracing/OTel init, shutdown signal handling. The only file that knows about concrete engine types.
- **`auth.rs`** — API key authentication middleware
- **`error.rs`** — unified `ApiError` type
- **`sse.rs`** — SSE streaming helper (shared by completions and chat)
- **`logprobs.rs`** — logprob format conversion (OpenAI completion + chat formats)
- **`utils.rs`** — metrics logging, usage aggregation
- **`routes/`** — one file per endpoint group, file names match API paths

Routes (OpenAI / vLLM / SGLang compatible):

- `GET /health` — health check
- `GET /v1/models`, `GET /v1/models/{model}` — model listing
- `POST /v1/completions` — text completion (batch + stream), supports `logprobs`
- `POST /v1/chat/completions` — chat completion (SSE streaming + batch), supports `logprobs` + `top_logprobs`
- `POST /v1/embeddings` — embedding generation (batched)
- `POST /v1/classify` — text classification

Authentication: optional API key via `--api-key` / `PRELUDE_API_KEY`. When enabled, `/v1/*` routes require
`Authorization: Bearer <key>`; `/health` is always open.

Logprobs API is OpenAI/vLLM/SGLang compatible: completions uses flat dict format, chat uses structured list with
`bytes`.

### Model Architectures (`models/architectures/`)

- `qwen3/` — Dense Qwen3 with multiple attention backends (dispatched via ops.rs)
- `qwen3_moe/` — Qwen3 Mixture-of-Experts with fused grouped GEMM
- `qwen3_next/` — Qwen3-Next hybrid model: Gated DeltaNet (linear attention) + Gated Attention + 512-expert MoE (80B
  total, 3B active). CPU F32 and GPU BF16 (flash-attn-v3). ScheduledEngine supported with DeltaNet state
  pool for concurrent multi-request decode. Uses residual RMSNorm `(1+weight)` parameterization.
- `qwen3_5/` — Qwen3.5 hybrid model: Gated DeltaNet + Gated Attention. Supports both dense (0.8B–27B) and MoE (35B-A3B).
  Config types: `qwen3_5`/`qwen3_5_text` (dense), `qwen3_5_moe`/`qwen3_5_moe_text` (MoE). VL wrapper auto-detected. MoE
  uses fused expert weights `[E, 2*inter, hidden]` with shared expert + sigmoid gate. CPU F32 and GPU BF16 (
  flash-attn-v3). ScheduledEngine supported with DeltaNet state pool for concurrent multi-request decode.
- `gguf` — Universal GGUF quantized model loader. Auto-detects architecture from GGUF metadata (`general.architecture`).
  Supported: qwen3, qwen3moe (GPU-only, FusedMoeGGUF needs CUDA), qwen35 (hybrid DeltaNet, CPU), llama,
  gemma3/gemma2/gemma, phi3, qwen2. Qwen3 wraps candle_transformers; Qwen3.5 is a custom implementation
  (`qwen3_5_gguf.rs`) with DeltaNet recurrence + gated attention + conv1d state, ported from llama.cpp.
- `gemma3/` — Gemma3 text and classifier model

Model linear/norm layers use conditional type aliases (`QwenLinear`/`QwenRmsNorm`) that switch between
cpu_ops and oneDNN implementations based on available features.

**Attention backend dispatch**: Models never import `candle_flash_attn_v3` or `candle_paged_attn` directly.
All attention dispatch goes through `models/layers/ops.rs` functions:
- `varlen_attention()` — causal, with optional paged KV (writes + reads)
- `varlen_attention_paged()` — paged-only (reads from pre-written cache, e.g. after fused KV write)
- `varlen_attention_windowed()` — sliding window with configurable left/right window
- `varlen_attention_bidirectional()` — non-causal
- `reshape_and_cache()` — write K/V to paged cache

Attention backend feature flags (`flash-attn-v3`, `flash-attn`) are confined to `attn/mod.rs`
dispatch; model files have zero attention backend compile-time gates.
Adding a new backend = one file in `attn/` + one dispatch line in `attn/mod.rs`.

### Custom CUDA Kernels

- `crates/prelude-flashinfer/` — FlashInfer AOT kernels (128 variants: h64/96/128/192/256, BF16/FP16, swa, softcap)
- `crates/prelude-flash-attn-v4/` — FA4 CuTeDSL AOT kernels (~400 variants per arch)
- `crates/prelude-deepgemm/` — DeepGEMM BF16 GEMM (JIT, SM90+)
- `crates/candle-paged-attn/` — paged attention v1/v2 + reshape_and_cache (legacy, FA2 only)
- `crates/candle-flash-attn-v3/` — Hopper SM90 flash attention (legacy, replaced by FlashInfer)
- `ops/gpu/kv_cache.rs` — vectorized PTX scatter_kv_cache_flash (replaces candle-paged-attn for flash layout)
- Fused ops in `fused_ops.rs` — vectorized BF16 add, fused SiLU×Mul, fused Add+RMSNorm, fused QKNorm+RoPE, fused
  K-Norm+RoPE+KV paged cache write (compiled to PTX at build time, loaded via cudarc at runtime)

### CPU Optimization

Two layers of CPU-optimized kernels (listed in priority order):

**Pure Rust kernels (`cpu_ops/`)** — default CPU path, zero external dependencies, AVX-512 + scalar fallback:
- `cpu_rmsnorm` / `cpu_fused_add_rmsnorm` — BF16 RMSNorm with rayon parallelization
- `cpu_silu_and_mul` — fused SiLU×Mul activation
- `cpu_rotary_embedding` — RoPE (NeoX split-half)
- `cpu_prefill_attention` / `decode_attention_bf16` — FlashAttention-style tiled attention (online softmax)
- `numa.rs` — NUMA-aware rayon pool initialization with physical core binding
- All use runtime `is_x86_feature_detected!()` dispatch, BF16-only

**oneDNN FFI (`crates/onednn-ffi/`, feature flag `onednn`)** — BF16 GEMM only (attention removed, tiled always wins):
- C++ wrapper: `crates/onednn-ffi/src/onednn_ffi.cpp`, header: `crates/onednn-ffi/include/onednn_ffi.h`
- Rust FFI bindings: `onednn_ffi.rs`, safe wrappers: `onednn_ops.rs`
- `bf16_linear` / `bf16_matmul` — unpacked BF16 GEMM
- `PackedWeight::pack()` + `bf16_linear_packed` — pre-packed weights
- Primitive cache keyed by (M, K, N) avoids JIT recompilation
- Static linking: libdnnl.a (~151MB) + libonednn_ffi.a → +33MB to final binary
- Auto-built via CMake on first `cargo build --features onednn`

## Key Design Decisions

- Dynamic batching logic in ScheduledEngine must not be modified when adding new features
- Streaming requests are partitioned from batch requests in the scheduler loop
- SSE streaming uses `futures_util::stream::unfold` over mpsc channels
- Fused CUDA kernels in `fused_ops.rs` can be individually disabled at runtime via atomic flags for numeric drift
  debugging (no recompilation needed)

### DeltaNet State Pool (`deltanet_pool.rs`)

Pre-allocated pool for DeltaNet recurrent and convolutional state, enabling multi-request concurrent decode for hybrid
models (Qwen3.5, Qwen3-Next). Design inspired by SGLang's MambaPool (Apache 2.0, attributed in source header).

- **Recurrent state**: `[max_slots, num_v_heads, head_k_dim, head_v_dim]` per layer, always F32 (numerical stability)
- **Conv state**: `[max_slots, conv_dim, kernel-1]` per layer, model dtype
- **Slot allocation**: Simple free-list (`VecDeque<u32>`), allocated in `batch_prefill_paged`, freed in
  `batched_stream_decode`
- **Pool sizing**: `PRELUDE_DELTANET_POOL_SLOTS` env var (default 8), or `max_running_requests` from scheduler
- **Backward compat**: When pool is `None` (non-hybrid models or `PRELUDE_NO_SCHEDULER=1`), models fall back to per-layer state

## Benchmark Testing

See `notes.md` for remote server setup (together_h200), benchmark workflows, and engine dependency details.

### How bench.sh works

`bench.sh` is the orchestrator. For each engine it: starts the server → waits for health check → runs `genai-bench` →
extracts metrics → kills the server. All engine-specific logic is split across two files:

- **`bench.sh`**: Engine registry (port, gpu_only, health_path, timeout), `start_engine()` case block, and the `all`
  dispatch order.
- **`bench_utils.py`**: `check-engine` (dependency checks), `extract-metrics` (JSON → CSV), `print-summary` (CSV →
  table).

### How to add a new benchmark engine

**Step 1: `benchmark/bench.sh`** — add 3 things:

1. **Engine registry entry** (pipe-delimited: `label|display_name|port|gpu_only|health_path|timeout`):
   ```bash
   [myengine]="myengine|MyEngine|8007|yes|/v1/models|300"
   ```
2. **`start_engine()` case block** — the command to start the server in background (`&`).
3. **Dispatch in `case "$target"`** — add engine name to the appropriate GPU/CPU section.

**Step 2: `benchmark/bench_utils.py`** — add a check in `check_engine()`:

Add one entry to the `checks` dict. The lambda returns `(bool, reason_string)`.

**Step 3: `docs/benchmark.md`** — add install instructions and update Quick Start / Manual Flow sections.

## CPU Performance (2026-03-16, h200 Xeon 8480+)

Pure Rust ops/cpu + oneDNN (ops/onednn). GemmPool 3-phase spin (spin→yield→park). No libtorch/OpenMP.
CPU KV cache for decode: pre-allocated KvBuf with `cpu_prefill_attention` (prefill) + `decode_attention_bf16` (decode).
F32 linear layers use oneDNN F32 GEMM (not candle fallback).

E2E: tokens=1 15ms (**43% faster**), tokens=128 38ms (**27% faster**), tokens=512 83ms (parity), tokens=4096 532ms (**29% faster**).
genai-bench D(128,1): c=1 **3629 t/s** (vs SGLang 2298 = 1.58x), c=4 **5710 t/s** (vs SGLang 3970 = 1.44x).
genai-bench D(32,32): c=1 **109.6 out/s**, TTFT=22ms, TPOT=8.2ms (vs SGLang 57.1/120ms/13.9ms).
genai-bench D(32,64): c=1 **116.2 out/s**, TTFT=25ms, TPOT=8.0ms (vs SGLang 55.6/230ms/14.5ms).
genai-bench D(128,128): c=1 **106.5 out/s**, TTFT=52ms, TPOT=8.7ms (vs SGLang 55.6/481ms/14ms).

