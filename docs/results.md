# Benchmark Results

- **Date**: 2026-04-09
- **Model**: Qwen/Qwen3-0.6B (BF16)
- **CPU**: AMD EPYC 9575F 64-Core Processor
- **GPU**: NVIDIA H200 (single GPU)
- **Tool**: genai-bench (rucnyz fork)
- **Engines**: Prelude (native), vLLM (Docker), SGLang (Docker)

## Prefill-Only (128 in, 1 out, c=1, 100 requests)

```bash
CUDA_VISIBLE_DEVICES=4 INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=100 CONCURRENCY=1 \
  ./benchmark/bench.sh sglang prelude vllm --gpu
```

| Engine  | Startup(s) | TTFT(s) | E2E(s) | In tok/s | RPM    |
|---------|------------|---------|--------|----------|--------|
| Prelude | 2          | 0.0051  | 0.0052 | 16,311.1 | 7,316.4|
| vLLM    | 44         | 0.0133  | 0.0135 | 7,843.1  | 3,514.9|
| SGLang  | 34         | 0.0282  | 0.0283 | 4,236.1  | 1,898.1|

**Prelude vs vLLM**: 2.08x throughput, 2.6x lower TTFT, 22x faster startup.
**Prelude vs SGLang**: 3.85x throughput, 5.5x lower TTFT, 17x faster startup.

## Decode (128 in, 32 out, c=4, 400 requests)

```bash
CUDA_VISIBLE_DEVICES=4 INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh sglang prelude vllm --gpu
```

| Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | In tok/s | Out tok/s | RPM    |
|---------|------------|---------|---------|--------|----------|-----------|--------|
| Prelude | 2          | 0.0146  | 0.0021  | 0.0802 | 6,167.8  | 1,474.0   | 2,763.8|
| vLLM    | 42         | 0.0309  | 0.0017  | 0.0833 | 5,979.2  | 1,430.4   | 2,682.0|
| SGLang  | 34         | 0.0483  | 0.0018  | 0.1033 | 4,917.3  | 1,176.7   | 2,206.2|

**Prelude vs vLLM**: 1.03x throughput, 2.1x lower TTFT, 21x faster startup.
**Prelude vs SGLang**: 1.25x throughput, 3.3x lower TTFT, 17x faster startup.

## Topicguard Qwen3-MoE 15B BF16 (1900 in, 3 out, real chatml dataset)

Updated 2026-05-21. Uses the `benchmark/local/bench_chatml.py` helper
(async aiohttp client, same data + concurrency sweep across engines)
instead of genai-bench's synthetic D(in,out) traffic — the chatml file
ships a shared policy-classifier system prompt that exercises prefix
caching the way a real deployment does.

```bash
# Build:
cargo build -p prelude-server --release --features full   # zero env vars on CUDA-13 hosts
# Datasets / engines: see dev/topicguard_qwen3_15b_bench.md
python3 benchmark/local/bench_chatml.py --concurrency 1,8,32,64,128 \
  --samples 1000 --max-tokens 3 --model <topicguard> --url <engine>
```

| Concurrency | prelude RPS | vLLM (cu130 v0.20.0) RPS | prelude / vLLM |
|---:|---:|---:|---:|
| 1   |  29.0 |  24.4 | **1.19×** |
| 8   | 182.5 |  92.8 | **1.97×** |
| 32  | 400.8 | 130.7 | **3.07×** |
| 64  | 562.4 | 210.4 | **2.67×** |
| 128 | **692.6** | 227.0 | **3.05×** |

Single-request latency at c=1: prelude **p50 33.8 ms** vs vLLM 38.4 ms.
At c=128 prelude pushes **1.32M input tok/s / 2.08K output tok/s**;
vLLM tops out at 0.43M / 681. SGLang on the same workload runs at
~46K input tok/s — see `dev/topicguard_qwen3_15b_bench.md` for the
full 3-engine table and the nsys kernel breakdown that explains the
gap (prelude wins on fused QKNorm+RoPE, fused Add+RMSNorm, and
DeepGEMM-vs-cuBLAS dense matmul; the MoE GroupGEMM kernel is the same
TRT-LLM CUTLASS kernel in both engines).
