# CPU Performance: Prelude vs SGLang

Goal: achieve parity with SGLang on all CPU scenarios (h200 Xeon 8480+, 56 cores).

## Current TODO

- ~~**P0**: CPU KV cache for decode~~ ✅ done — concat KV cache with optimized BF16 kernels (cpu_extend_attention for prefill, decode_attention_bf16 for decode)
- **P1**: Fuse softmax F32→BF16 conversion — produce BF16 scores inside softmax, skip separate pass
- ~~**P2**: Eliminate redundant K/V gather~~ ✅ done
- **P3**: Increase BLOCK_N to 768 for slen > 4096 (match SGLang tuning)

## Current Status (2026-03-14, post-rebase)

h200 Xeon 8480+ (2 sockets, 112 physical cores).

| Scenario | Prelude | SGLang | Ratio | Status |
|----------|-----------|--------|-------|--------|
| E2E (tokens=1) | 15.1±2.8ms | 26.4ms | **0.57x** ✅ | **43% FASTER** |
| E2E (tokens=128) | 37.7±11.2ms | 51.8ms | **0.73x** ✅ | **27% FASTER** |
| E2E (tokens=512) | 82.7±14.2ms | 77.8ms | **1.06x** | **NEAR PARITY** |
| E2E (tokens=1024) | 141.8±14.4ms | 121.9ms | 1.16x | distributed overhead |
| E2E (tokens=4096) | 531.5±66.9ms | 749.5ms | **0.71x** ✅ | **29% FASTER** |
| E2E (tokens=8192) | 1466.4ms | (crashed) | — | SGLang OOM/timeout |

**tokens=1-128: beat SGLang. tokens=512: near parity. tokens=4096: 29% faster. tokens=8192: SGLang crashed.**

### genai-bench Throughput: D(128,1) prefill-only

| Engine                  | c | Reqs | Startup(s) | TTFT(s) | E2E(s) | Input t/s | RPM  |
|-------------------------|---|------|------------|---------|--------|-----------|------|
| **Prelude**           | 1 | 500  | 4          | 0.0340  | 0.0341 | **3513**  | 1573 |
| **Prelude**           | 4 | 500  | 4          | 0.1129  | 0.1130 | **4451**  | 1996 |
| **SGLang-CPU** (docker) | 1 | 500  | 64         | 0.0496  | 0.0497 | **2435**  | 1092 |
| **SGLang-CPU** (docker) | 4 | 500  | 60         | 0.1226  | 0.1228 | **4118**  | 1846 |

**Prelude c=1: 3513 t/s vs SGLang c=1: 2435 t/s = 1.44x faster.**
**Prelude c=4: 4451 t/s vs SGLang c=4: 4118 t/s = 1.08x faster.**

### genai-bench Throughput: D(32,32) decode

| Engine                  | c | Reqs | TTFT(s) | TPOT(s) | E2E(s) | Out t/s | RPM  |
|-------------------------|---|------|---------|---------|--------|---------|------|
| **Prelude**           | 1 | 50   | 0.0226  | 0.0079  | 0.2662 | **118.7**| 223 |
| **SGLang-CPU** (docker) | 1 | 50   | 0.1316  | 0.0140  | 0.5665 | **55.1**| 103  |

**Prelude: 118.7 out t/s vs SGLang: 55.1 out t/s = 2.15x faster.**
TTFT 23ms vs 132ms (5.7x faster). TPOT 7.9ms vs 14.0ms (1.77x faster).

### genai-bench Throughput: D(32,64) decode

| Engine                  | c | Reqs | TTFT(s) | TPOT(s) | E2E(s) | Out t/s | RPM  |
|-------------------------|---|------|---------|---------|--------|---------|------|
| **Prelude**           | 1 | 20   | 0.0248  | 0.0080  | 0.5288 | **116.2**| 109 |
| **SGLang-CPU** (docker) | 1 | 20   | 0.2303  | 0.0145  | 1.1410 | **55.6**| 52   |

**Prelude: 116.2 out t/s vs SGLang: 55.6 out t/s = 2.09x faster.**
TTFT 25ms vs 230ms (9.3x faster). TPOT 8.0ms vs 14.5ms (1.81x faster).

### genai-bench Throughput: D(128,128) long context + decode

| Engine                  | c | Reqs | TTFT(s) | TPOT(s) | E2E(s) | Out t/s | RPM  |
|-------------------------|---|------|---------|---------|--------|---------|------|
| **Prelude**           | 1 | 10   | 0.0524  | 0.0087  | 1.1558 | **106.5**| 50  |
| **SGLang-CPU** (docker) | 1 | 10   | 0.4811  | 0.0140  | 2.2560 | **55.6**| 26   |

**Prelude: 106.5 out t/s vs SGLang: 55.6 out t/s = 1.92x faster.**
TTFT 52ms vs 481ms (9.2x faster). TPOT 8.7ms vs 14.0ms (1.61x faster).

(2026-03-16, post CPU continuous runtime with per-token streaming. Same session for RI and SGLang.)

Reproduce:
```bash
# On h200: setup .venv with genai-bench fork (one-time)
cd /path/to/prelude
uv venv .venv --python 3.12 --clear
source .venv/bin/activate
uv pip install -e /path/to/genai-bench

# Prelude
source .venv/bin/activate
cargo build -p prelude-server --features onednn --release
# Prefill-only D(128,1)
INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=1 \
  ./benchmark/bench.sh prelude --cpu
INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=4 \
  ./benchmark/bench.sh prelude --cpu
# Decode D(32,32)
INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=50 CONCURRENCY=1 \
  ./benchmark/bench.sh prelude --cpu
# SGLang (via docker, sglang-cpu image required)
source .venv/bin/activate
INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=1 \
  ./benchmark/bench.sh sglang-cpu
INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=50 CONCURRENCY=1 \
  ./benchmark/bench.sh sglang-cpu
```

### Local Results (Ryzen 9 9950X, 16 cores, single socket)

| Scenario | Prelude | SGLang | Ratio | Status |
|----------|-----------|--------|-------|--------|
| E2E (tokens=1) | 27.8±2.8ms | 52.8±10.0ms | **0.53x** ✅ | **47% FASTER** |
| E2E (tokens=128) | 61.1±3.6ms | 101.2±3.4ms | **0.60x** ✅ | **40% FASTER** |
| E2E (tokens=512) | 166.2±8.2ms | 338.9±17.5ms | **0.49x** ✅ | **51% FASTER** |
| E2E (tokens=1024) | 304.3±7.3ms | 661.9±14.1ms | **0.46x** ✅ | **54% FASTER** |

**All token counts 2x faster on consumer CPU.** SGLang's Python/PyTorch overhead dominates
on fewer cores (16 vs 112) — cannot amortize framework overhead with parallelism.

### Gap analysis (h200 only)

tokens=512-1024 gap analysis:
- Gap is ~1.7ms/layer distributed across all components, not a single bottleneck
- SGLang's per-component profiling only captures 15% of e2e time (85% is Python overhead)
- Component-level comparison between SGLang and Prelude is unreliable
- Our norm kernel (660µs) is 4.3x faster than SGLang's actual `forward_native` (2857µs)
- SGLang's `_is_cpu=False` on h200, so it uses PyTorch native ops (not sgl-kernel C++)
  for norms and possibly GEMM — the C++ optimized path is not active

## Architecture

All CPU hot paths use **pure Rust cpu_ops** (AVX-512) + **oneDNN** (brgemm GEMM).
No libtorch, no OpenMP. CPU path is BF16-only (no F32 fallbacks).
Single thread pool: GemmPool (3-phase spin, pinned to all physical cores across NUMA nodes).

- **GemmPool**: spin→yield→park (5ms default). Threads pinned to all physical cores across NUMA nodes
- **GEMM**: brgemm with 2D (M×N) tiling, VNNI-packed weights
- **Attention**: brgemm for both QK^T and V accumulation (AMX-accelerated)
- **MLP**: raw forward path (zero Tensor allocations in hot path)
- **Norms/RoPE**: fused GemmPool dispatch (deinterleave+QK norm+RoPE in one pass)
- **VNNI packing**: AVX-512 vectorized for V (pack_vnni_avx512) and K^T (gather-based)

## Remaining Optimization Opportunities

### P1: Fuse softmax F32→BF16 conversion
Currently softmax produces F32, then brgemm_score_v_accum converts to BF16 separately.
SGLang converts inside softmax with _mm512_cvtneps_pbh. Saves one pass over score matrix.

### P2: Eliminate redundant K/V gather
Each M-block gathers K/V from strided source into contiguous buffer, then brgemm packs
from contiguous. SGLang's pack_vnni reads directly from strided source with index array,
skipping the intermediate gather. Saves one full memcpy per M-block.

### P3: Increase BLOCK_N for long sequences
SGLang uses BLOCK_N=768 for slen > 4096, we use 512. Larger BLOCK_N means fewer
N-block iterations and better cache reuse per block.

## Completed Optimizations

### ~~Eliminate per-block AMX context churn~~ (done 2026-03-14)
Removed per-block set_hw_context/release_hw_context from brgemm_qk_gemm and
brgemm_score_v_accum. Uses tls_current pointer for dedup + single
brgemm_attn_release() per M-block.

### ~~Vectorize K^T VNNI packing~~ (done 2026-03-14)
AVX-512 32-bit gather: each `_mm512_i32gather_epi32` loads 16 × {K[j,d0], K[j,d0+1]}
pairs — already VNNI format. Replaces scalar triple-nested loop in brgemm_qk_gemm.

### ~~Beta=1 direct SV accumulation~~ (done 2026-03-14)
brgemm_score_v_accum now uses `add_C=1` (beta=1) kernel to accumulate directly into
v_prime, eliminating temp buffer + manual F32 addition loop.

### ~~M-block parallelism~~ (done 2026-03-14)
Dispatch (req, head, m_block) triples to GemmPool instead of just (req, head).
For tokens=512: 64 work items instead of 16. Each M-block independently gathers
its K/V and processes its N-blocks.

### ~~AVX-512 output normalization~~ (done 2026-03-14)
Fused scale + F32→BF16 output using `_mm512_mul_ps` + bit-manipulation BF16 rounding +
`_mm512_cvtepi32_epi16`. Replaces scalar per-element `f32_to_bf16(v * inv_sum)`.

### ~~Q zero-copy~~ (done 2026-03-14)
Pass strided Q pointer directly to brgemm_qk_gemm (which already supports q_stride).
Eliminates m_size × head_dim memcpy per M-block.

### ~~All-core threading~~ (done 2026-03-14)
GemmPool now detects and pins to ALL physical cores across both NUMA nodes
(was NUMA node 0 only, capped at 48). On 2-socket Xeon 8480+: 112 threads instead of 48.
A/B test: tokens=4096 859ms→628ms (-27%), tokens=1024 265ms→164ms (-38%).

### ~~Zero-gather K/V~~ (done 2026-03-14)
brgemm_qk_gemm and brgemm_score_v_accum now accept k_stride and v_stride parameters.
Rust passes strided K/V pointers directly — no memcpy gather per M-block.
Eliminates up to 2.3MB redundant copies at tokens=1024 (8 M-blocks × overlapping gathers).

### ~~Load+transpose K^T VNNI packing~~ (done 2026-03-14)
Replace AVX-512 gather with 16×16 32-bit transpose (same as SGLang's pack_vnni_Nx32).
Each row loads full cache line (64B), 100% utilization vs gather's 6.25%.

### ~~Fuse sm_scale into softmax~~ (done 2026-03-14)
Remove sm_scale multiply loop from brgemm_qk_gemm. Softmax now computes
`exp(score * sm_scale - max)` inline, saving one full memory pass per N-block.

### ~~Adaptive norm thread count~~ (done 2026-03-14)
GemmPool norm dispatch limits threads based on MIN_ELEMS_PER_THREAD (16384).
Prevents over-parallelization for small workloads.

### ~~BLOCK_N=512 for slen 513-1024~~ (done 2026-03-14)
Halves N-block count (4→2 per M-block). A/B test: tokens=1024 135ms→128ms (-5%).

Combined result (h200): tokens=1-128 beat SGLang, tokens=4096 **32% faster** (0.68x).
Combined result (local Ryzen 9950X): **all token counts 2x faster** (0.46-0.60x).

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `CPU_GEMM_THREADS` | all physical cores | Number of GemmPool spinning threads. Controls parallelism for GEMM, attention, and MLP dispatch |
| `CPU_GEMM_SPIN_MS` | 5 | Spin-wait duration (ms) before parking idle threads. Higher = lower wake latency, more CPU usage |
| `PRELUDE_ATTN_PROFILE` | off | Set to `1` to print per-head attention profiling (gather/QK/softmax/SV breakdown) |

## How to Reproduce

### Local (any Linux x86_64 with AVX-512 or AVX2)

```bash
# 1. Build
cargo build -p prelude-server --release --features onednn

# 2. E2E sweep (RI vs SGLang)
SGLANG_PYTHON=/path/to/sglang/.venv/bin/python \
  PROMPT_TOKENS="1,128,512,1024" \
  ./benchmark/bench_mlp_sweep.sh all

# 3. Prelude only (no SGLang needed)
./benchmark/bench_mlp_sweep.sh prelude
```

### h200 (remote)

```bash
# 1. Rsync from local to h200 (exclude build artifacts + .venv)
rsync -avz --exclude='target/' --exclude='.git/' \
  --exclude='crates/onednn-ffi/build/' --exclude='.venv/' \
  . <remote-host>:/path/to/prelude/

# 2. SSH to h200
ssh <remote-host>
cd /path/to/prelude

# 3. Build (server + bench binary)
cargo build -p prelude-server --release --features onednn

# 4. E2E sweep (RI vs SGLang, 10 trials with avg±stdev)
SGLANG_PYTHON=/path/to/sglang/.venv/bin/python \
  ./benchmark/bench_mlp_sweep.sh all

# 5. Per-layer profiling
PRELUDE_DEVICE=cpu PRELUDE_PROFILE=1 RUST_LOG=info \
  target/release/prelude-server --port 8198 --model Qwen/Qwen3-0.6B --dtype bf16
```

## Servers

- **h200**: `ssh <remote-host>`, project at `/path/to/prelude`
- **surfi1**: `ssh surfi1`, project at `/path/to/prelude`
- **local (office_arch)**: Ryzen 9 9950X, SGLang at `/path/to/sglang`
