# Benchmark Results

- **Date**: 2026-04-07
- **Model**: Qwen/Qwen3-0.6B (BF16)
- **CPU**: AMD EPYC 9575F 64-Core Processor
- **GPU**: NVIDIA H200 (single GPU, CUDA_VISIBLE_DEVICES=2)
- **Tool**: genai-bench 0.0.3 (rucnyz fork)
- **Engines**: Prelude (native), vLLM 0.18.0 (Docker), SGLang (Docker)

## Prefill-Only (128 in → 1 out, c=1, 100 requests)

| Engine  | Startup(s) | TTFT(s) | E2E(s) | Input tok/s | Output tok/s | RPM    |
|---------|------------|---------|--------|-------------|-------------|--------|
| Prelude | 4          | 0.0068  | 0.0069 | 14,520.9    | 108.6       | 6,514.8 |
| vLLM    | 60         | 0.0107  | 0.0108 | 9,926.3     | 74.2        | 4,452.4 |
| SGLang  | 38         | 0.0407  | 0.0409 | 3,098.2     | 23.2        | 1,389.7 |

**Prelude vs vLLM: 1.46x RPM, 1.57x lower TTFT, 15x faster startup**
**Prelude vs SGLang: 4.69x RPM, 5.99x lower TTFT**

## Decode (32 in → 32 out, c=4, 400 requests)

| Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | Output tok/s | RPM    |
|---------|------------|---------|---------|--------|-------------|--------|
| Prelude | 2          | 0.0135  | 0.0441  | 1.3800 | 92.0        | 172.5  |
| vLLM    | 50         | 0.0215  | 0.0016  | 0.0705 | 1,713.8     | 3,213.5 |

SGLang result missing (benchmark timed out).

## Decode (128 in → 32 out, c=4, 400 requests)

| Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | Output tok/s | RPM    |
|---------|------------|---------|---------|--------|-------------|--------|
| Prelude | 14         | 0.0178  | 0.0442  | 1.3888 | 91.4        | 171.3  |
| vLLM    | 42         | 0.0295  | 0.0016  | 0.0803 | 1,474.5     | 2,764.8 |
| SGLang  | 32         | 0.0669  | 0.0018  | 0.1236 | 983.5       | 1,844.0 |

## Analysis

**Prefill**: Prelude is fastest — 1.46x vLLM, 4.69x SGLang in RPM. TTFT of 6.8ms beats vLLM (10.7ms) and SGLang (40.7ms). Startup is 4s vs 60s/38s.

**Decode**: Prelude TPOT of ~44ms is significantly slower than vLLM (1.6ms) and SGLang (1.8ms). This is a known issue — the decode path uses `batch_decode_paged` which currently lacks CUDA graph replay and has per-step overhead from the scheduling loop. The origin codebase achieves ~5ms TPOT via CUDA graphs and tighter decode loop integration. This is the top priority for performance work.

## Notes

- GPUs 0,1,4-7 were occupied during this run; only GPU 2 was idle.
- Docker images: `vllm/vllm-openai:latest` (v0.18.0), `lmsysorg/sglang:latest`.
