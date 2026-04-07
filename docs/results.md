# Benchmark Results

- **Date**: 2026-04-07
- **Model**: Qwen/Qwen3-0.6B (BF16)
- **Tool**: genai-bench 0.0.3 (rucnyz fork)
- **Engines**: Prelude (native), vLLM 0.18.0 (Docker), SGLang (Docker)

| Setup | CPU | GPU | Notes |
|-------|-----|-----|-------|
| H200  | AMD EPYC 9575F 64-Core | NVIDIA H200 (single GPU) | Docker: `lmsysorg/sglang:latest` (CUDA 12.9) |
| B300  | Intel Xeon 6776P | NVIDIA B300 SXM6 AC (single GPU) | Docker: `lmsysorg/sglang:latest-cu130` (CUDA 13.0) |

## Prefill-Only (128 in → 1 out, c=1, 100 requests)

```bash
CUDA_VISIBLE_DEVICES=<N> INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=100 CONCURRENCY=1 \
  ./benchmark/bench.sh <engine> --gpu
```

| GPU  | Engine  | Startup(s) | TTFT(s) | E2E(s) | Input tok/s | Output tok/s | RPM    |
|------|---------|------------|---------|--------|-------------|-------------|--------|
| H200 | Prelude | 4          | 0.0068  | 0.0069 | 14,520.9    | 108.6       | 6,514.8 |
| H200 | vLLM    | 60         | 0.0107  | 0.0108 | 9,926.3     | 74.2        | 4,452.4 |
| H200 | SGLang  | 38         | 0.0407  | 0.0409 | 3,098.2     | 23.2        | 1,389.7 |
| B300 | vLLM    | 108        | 0.0199  | 0.0201 | 5,756.2     | 43.0        | 2,580.7 |
| B300 | SGLang  | 120        | 0.1404  | 0.1405 | 931.6       | 7.0         | 418.1   |

**H200 — Prelude vs vLLM: 1.46x RPM, 1.57x lower TTFT, 15x faster startup**
**H200 — Prelude vs SGLang: 4.69x RPM, 5.99x lower TTFT**

## Decode (32 in → 32 out, c=4, 400 requests)

```bash
CUDA_VISIBLE_DEVICES=<N> INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh <engine> --gpu
```

| GPU  | Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | Output tok/s | RPM    |
|------|---------|------------|---------|---------|--------|-------------|--------|
| H200 | Prelude | 2          | 0.0135  | 0.0441  | 1.3800 | 92.0        | 172.5  |
| H200 | vLLM    | 50         | 0.0215  | 0.0016  | 0.0705 | 1,713.8     | 3,213.5 |
| B300 | vLLM    | 110        | 0.0326  | 0.0028  | 0.1209 | 995.7       | 1,866.9 |
| B300 | SGLang  | 106        | 0.1851  | 0.0015  | 0.2305 | 539.5       | 1,011.6 |

H200 SGLang result missing (benchmark timed out).

## Decode (128 in → 32 out, c=4, 400 requests)

```bash
CUDA_VISIBLE_DEVICES=<N> INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh <engine> --gpu
```

| GPU  | Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | Output tok/s | RPM    |
|------|---------|------------|---------|---------|--------|-------------|--------|
| H200 | Prelude | 14         | 0.0178  | 0.0442  | 1.3888 | 91.4        | 171.3  |
| H200 | vLLM    | 42         | 0.0295  | 0.0016  | 0.0803 | 1,474.5     | 2,764.8 |
| H200 | SGLang  | 32         | 0.0669  | 0.0018  | 0.1236 | 983.5       | 1,844.0 |
| B300 | vLLM    | 114        | 0.0465  | 0.0052  | 0.2068 | 487.4       | 914.0  |
| B300 | SGLang  | 100        | 0.1338  | 0.0014  | 0.1775 | 691.4       | 1,296.3 |

## Analysis

**Prefill**: Prelude is fastest — 1.46x vLLM, 4.69x SGLang in RPM. TTFT of 6.8ms beats vLLM (10.7ms) and SGLang (40.7ms). Startup is 4s vs 60s/38s.

**Decode**: Prelude TPOT of ~44ms is significantly slower than vLLM (1.6ms) and SGLang (1.8ms). This is a known issue — the decode path runs `batch_decode_paged` eagerly every step (6 tensor allocations + H2D copies per step). CUDA graph capture is implemented but blocked by a stream isolation issue (weight tensors created on a different stream than the capture stream). This is the top priority for performance work.

**B300 vs H200**: B300 (Blackwell) should be faster than H200 (Hopper), but both vLLM and SGLang show 33–70% of their H200 RPM on B300. This reflects immature Blackwell support in both frameworks, not hardware capability. SGLang's default image failed entirely on B300 (required CUDA 13 image); vLLM's decode TPOT regressed 3x (1.6ms → 5.2ms). These B300 numbers should not be used for hardware comparisons.

## Notes

- **H200 run**: GPUs 0,1,4-7 were occupied; only GPU 2 was idle. Docker: `vllm/vllm-openai:latest` (v0.18.0), `lmsysorg/sglang:latest`.
- **B300 run**: Single GPU. Docker: `vllm/vllm-openai:latest`, `lmsysorg/sglang:latest-cu130` (tagged as `latest` locally). Original `lmsysorg/sglang:latest` (CUDA 12.9) failed on B300 with `no kernel image is available for execution on the device` (SM_120 not supported).
