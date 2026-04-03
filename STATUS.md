## Roadmap

### Quantization
- **INT4 weight GEMM via CUTLASS** (near-term)
  - CUTLASS example 55: `55_hopper_int4_bf16_gemm` — INT4 x BF16 mixed GEMM with group-wise scale/zero-point
  - Same CollectiveBuilder API as our existing SM90 BF16 path
  - Supports GPTQ/AWQ-style per-group dequant natively
  - Need: weight loader (INT4 packed format), `cutlass_wrapper_int4.cu`, dispatch by weight dtype
  - Blackwell: example 86 (`86_blackwell_mixed_dtype_gemm`) for SM100+
- **FP8 KV cache** (near-term)
  - FlashInfer already supports `kv_cache_dtype="fp8"` — just need BF16→FP8 cast on paged KV write
  - 2x KV memory reduction, enables more concurrent sequences
- **TurboQuant-style 3-bit KV cache** (research, blocked on open-source)
  - https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
  - PolarQuant + QJL: 3-bit KV cache, zero accuracy loss, 8x attention speedup on H100
  - Requires custom attention kernels (no existing backend supports 3-bit input)
  - Wait for open-source implementation before evaluating

### Other
- SGLang linear attention DP support is weak (seen in their issues)
- Support loading unsloth GGUF (Qwen3.5-9B) via sglang

# Current Status (2026-03-24)

## What's Done

### cuBLAS Removal — Single Binary, No CUDA Toolkit Dependency
- **DeepGEMM** (SM90+): 105 AOT kernel variants covering all Qwen3 + Gemma3 model dims. Primary GEMM backend on Hopper.
- **CUTLASS 3.x** (SM80+): Universal fallback. SM90 uses CollectiveBuilder (example 49 pattern), SM80 uses manual CollectiveMma with unpredicated mainloop.
- **Binary**: 118MB, `ldd` shows only `libcuda.so.1` + system libs. No cuBLAS, cuRAND, nvrtc, cudart.
- **Accuracy**: 10/10 passed vs transformers (Qwen3-0.6B GPU). GEMM correctness 14/14 vs CPU F64 reference.

### GEMM Microbenchmark (`gpu_ops_bench`)
- Compares: dispatch (DeepGEMM → CUTLASS) vs CUTLASS direct vs SM80 fallback vs cuBLAS
- Correctness verification: `--verify` flag checks all backends against CPU F64
- 6 SM80 tile configs benchmarkable via `cutlass_gemm_sm80(config=0..5)`
- Models: all 10 supported (Qwen3 0.6B-32B + MoE-A3B, Gemma3 1B/4B/27B)

### Benchmark Infrastructure
- `benchmark.py` with classify/embed/complete/mix tasks
- Per-engine serve scripts (`serve_prelude.sh`, `serve_sglang.sh`, `serve_vllm.sh`)

### E2E Performance (H200, Qwen3-0.6B, CUDA graph enabled)
| Scenario | Prelude | SGLang 0.5.9 | vLLM 0.18.0 |
|---|---|---|---|
| Prefill D(128,1) c=1 | **9,851 in t/s** | 4,929 | 4,574 |
| Prefill D(128,1) c=4 | **19,702 in t/s** | 5,730 | 11,009 |
| Decode D(32,32) c=4 | 1,086 out t/s | 947 | **1,234** |
| Startup | **4s** | 52s | 66s |

CUDA graph: ~10% decode improvement (720 vs 657 out t/s in A/B test).

## What Needs Updating

### 1. Build flags in docs
`cutlass-gemm` feature is implemented but not in the recommended build command.

**Files to update:**
- `README.md`: `--features flashinfer-v4,onednn,deepgemm` → add `,cutlass-gemm`
- `docs/getting-started.md`: same
- `docs/getting-started.md`: add `cutlass-gemm` to feature flags table

### 2. Benchmark — Docker for baselines
SGLang/vLLM can be run via Docker to avoid pip dependency conflicts:
```bash
docker pull lmsysorg/sglang:latest
docker pull vllm/vllm-openai:latest
```
The private repo's `bench.sh` had Docker integration; this repo uses `serve_sglang.sh` (native python). Consider adding Docker as an option.

### 3. Benchmark scenarios to add
- D(128,32) c=4 (decode with longer prefill)
- D(32,32) c=4 with 400 requests (currently in results.md)

### 4. CUTLASS SM80 — performance on actual A100
All SM80 benchmarks were run on H200 (SM80 PTX on SM90 hardware). Need A100 numbers for accurate SM80 fallback characterization. Expected: 1.2-2x vs cuBLAS (vs 3-7x on H200 due to instruction set mismatch).

### 5. `bench-cublas` feature
The `bench-cublas` feature (optional cuBLAS for microbenchmark comparison) exists in the private repo but is not wired in `prelude-server/Cargo.toml` here. Low priority — only needed for GEMM microbench, not for production.

## Architecture Notes

### GEMM Dispatch Path
```
Tensor::matmul() → candle gemm_dispatch
  → DeepGEMM (SM90+, 105 AOT variants)
  → CUTLASS SM90 (CollectiveBuilder, KernelScheduleAuto, 128x128x64)
  → CUTLASS SM80 (CollectiveMma, unpredicated, 128x128x64/3-stage)
  → error
```

### Key Files
| File | What |
|---|---|
| `crates/prelude-cutlass-gemm/src/cutlass_wrapper.cu` | CUTLASS 3.x SM90 + SM80 kernels |
| `crates/prelude-cutlass-gemm/build.rs` | Sparse CUTLASS clone + nvcc compile |
| `crates/prelude-deepgemm/src/deepgemm_wrapper.cu` | DeepGEMM 105 AOT variants |
| `crates/prelude-core/src/ops/gpu/gemm.rs` | GEMM dispatch registration |
| `crates/prelude-core/src/bin/gpu_ops_bench/` | Microbenchmark + correctness |
| `crates/candle-core/src/cuda_backend/gemm_dispatch.rs` | candle matmul → our dispatch |
