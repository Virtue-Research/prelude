# Benchmark Results

- **Date**: 2026-04-07
- **Model**: Qwen/Qwen3-0.6B (BF16)
- **CPU**: AMD EPYC 9575F 64-Core Processor
- **GPU**: NVIDIA H200 (single GPU, CUDA_VISIBLE_DEVICES=2)
- **Tool**: genai-bench 0.0.3 (rucnyz fork)
- **Engines**: Prelude (native), vLLM 0.18.0 (Docker), SGLang (Docker)

## Prefill-Only (128 in → 1 out, c=1, 100 requests)

```bash
CUDA_VISIBLE_DEVICES=2 INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=100 CONCURRENCY=1 \
  ./benchmark/bench.sh <engine> --gpu
```

| Engine  | Startup(s) | TTFT(s) | E2E(s) | Output tok/s | RPM    | Overall tok/s |
|---------|------------|---------|--------|-------------|--------|---------------|
| Prelude | 2          | 0.0080  | 0.0081 | 86.5        | 5191.6 | 11,162        |
| vLLM    | 46         | 0.0137  | 0.0139 | 58.3        | 3495.5 | 7,517         |
| SGLang  | 42         | 0.0438  | 0.0439 | 21.1        | 1268.7 | 2,728         |

**Prelude vs vLLM: 1.48x throughput, 1.72x lower TTFT**
**Prelude vs SGLang: 4.09x throughput, 5.48x lower TTFT**

## Decode (32 in → 32 out, c=4, 400 requests)

```bash
CUDA_VISIBLE_DEVICES=2 INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh <engine> --gpu
```

| Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | Output tok/s | RPM    | Overall tok/s |
|---------|------------|---------|---------|--------|-------------|--------|---------------|
| Prelude | 6          | 0.0226  | 0.0000  | 0.0227 | 143.5       | 8607.9 | 9,176         |
| vLLM    | 48         | 0.0292  | 0.0016  | 0.0800 | 1488.2      | 2790.3 | 2,976         |
| SGLang  | 36         | 0.0574  | 0.0018  | 0.1127 | 1081.0      | 2026.9 | 2,162         |

**Prelude vs vLLM: 3.08x RPM, 3.53x lower E2E latency**
**Prelude vs SGLang: 4.25x RPM, 4.96x lower E2E latency**

## Decode (128 in → 32 out, c=4, 400 requests)

```bash
CUDA_VISIBLE_DEVICES=2 INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh <engine> --gpu
```

| Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | Output tok/s | RPM    | Overall tok/s |
|---------|------------|---------|---------|--------|-------------|--------|---------------|
| Prelude | 2          | 0.0252  | 0.0000  | 0.0253 | 126.0       | 7557.7 | 20,154        |
| vLLM    | 46         | 0.0302  | 0.0016  | 0.0812 | 1448.3      | 2715.5 | 7,241         |
| SGLang  | 38         | 0.0520  | 0.0018  | 0.1088 | 1106.0      | 2073.8 | 5,530         |

**Prelude vs vLLM: 2.78x RPM, 3.21x lower E2E latency**
**Prelude vs SGLang: 3.64x RPM, 4.30x lower E2E latency**

## Notes

- TPOT=0.000 for Prelude is a genai-bench measurement artifact — it reports E2E-level timing, not per-token streaming latency.
- vLLM and SGLang report per-token TPOT because they use chunked streaming responses.
- Overall tok/s = RPM * (input_tokens + output_tokens) / 60.
- vLLM/SGLang input_tps values come from genai-bench CSV; Prelude reports 0.0 due to the same measurement artifact.
- Docker images: `vllm/vllm-openai:latest` (v0.18.0), `lmsysorg/sglang:latest`.
- Results may vary with machine load. GPUs 0,1,4-7 were occupied during this run.
