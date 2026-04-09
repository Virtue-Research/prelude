# Benchmark Results

- **Date**: 2026-04-09
- **Model**: Qwen/Qwen3-0.6B (BF16)
- **CPU**: AMD EPYC 9575F 64-Core Processor
- **GPU**: NVIDIA H200 (single GPU)
- **Tool**: genai-bench (rucnyz fork)
- **Engines**: Prelude (native), vLLM (Docker), SGLang (Docker)

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
