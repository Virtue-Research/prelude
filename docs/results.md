# Benchmark Results

- **Date**: 2026-04-09
- **CPU**: AMD EPYC 9575F 64-Core Processor
- **GPU**: NVIDIA H200 (single GPU)
- **Tool**: genai-bench (rucnyz fork)
- **Engines**: Prelude (native), vLLM (Docker), SGLang (Docker)

## Qwen/Qwen3-0.6B (BF16)

### Prefill-Only (128 in, 1 out, c=1, 100 requests)

| Engine  | Startup(s) | TTFT(s) | E2E(s) | In tok/s | RPM    |
|---------|------------|---------|--------|----------|--------|
| Prelude | 2          | 0.0051  | 0.0052 | 16,311.1 | 7,316.4|
| vLLM    | 44         | 0.0133  | 0.0135 | 7,843.1  | 3,514.9|
| SGLang  | 34         | 0.0282  | 0.0283 | 4,236.1  | 1,898.1|

**Prelude vs vLLM**: 2.08x throughput, 2.6x lower TTFT, 22x faster startup.
**Prelude vs SGLang**: 3.85x throughput, 5.5x lower TTFT, 17x faster startup.

### Decode (128 in, 32 out, c=4, 400 requests)

| Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | In tok/s | Out tok/s | RPM    |
|---------|------------|---------|---------|--------|----------|-----------|--------|
| Prelude | 2          | 0.0146  | 0.0021  | 0.0802 | 6,167.8  | 1,474.0   | 2,763.8|
| vLLM    | 42         | 0.0309  | 0.0017  | 0.0833 | 5,979.2  | 1,430.4   | 2,682.0|
| SGLang  | 34         | 0.0483  | 0.0018  | 0.1033 | 4,917.3  | 1,176.7   | 2,206.2|

**Prelude vs vLLM**: 1.03x throughput, 2.1x lower TTFT, 21x faster startup.
**Prelude vs SGLang**: 1.25x throughput, 3.3x lower TTFT, 17x faster startup.

## Qwen/Qwen3-8B (BF16)

### Prefill-Only (128 in, 1 out, c=1, 100 requests)

```bash
MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=4 INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=100 CONCURRENCY=1 \
  ./benchmark/bench.sh sglang prelude vllm --gpu
```

| Engine  | Startup(s) | TTFT(s) | E2E(s) | In tok/s | RPM    |
|---------|------------|---------|--------|----------|--------|
| Prelude |            |         |        |          |        |
| vLLM    |            |         |        |          |        |
| SGLang  |            |         |        |          |        |

### Decode (128 in, 32 out, c=4, 400 requests)

```bash
MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=4 INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh sglang prelude vllm --gpu
```

| Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | In tok/s | Out tok/s | RPM    |
|---------|------------|---------|---------|--------|----------|-----------|--------|
| Prelude |            |         |         |        |          |           |        |
| vLLM    |            |         |         |        |          |           |        |
| SGLang  |            |         |         |        |          |           |        |
