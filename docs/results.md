# Benchmark Results

- **Tool**: genai-bench (rucnyz fork)
- **Engines**: Prelude (native), vLLM (Docker), SGLang (Docker)

| Setup | CPU | GPU |
|-------|-----|-----|
| H200  | AMD EPYC 9575F 64-Core | NVIDIA H200 (single GPU) |
| B300  | Intel Xeon 6776P | NVIDIA B300 SXM6 AC (single GPU) |

---

## H200 — Qwen/Qwen3-0.6B (BF16)

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

## H200 — Qwen/Qwen3-8B (BF16)

### Prefill-Only (128 in, 1 out, c=1, 100 requests)

```bash
MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=4 INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=100 CONCURRENCY=1 \
  ./benchmark/bench.sh sglang prelude vllm --gpu
```

| Engine  | Startup(s) | TTFT(s) | E2E(s) | In tok/s | RPM    |
|---------|------------|---------|--------|----------|--------|
| Prelude | 4          | 0.0133  | 0.0134 | 8,068.4  | 3,619.2|
| vLLM    | 48         | 0.0149  | 0.0150 | 7,336.9  | 3,290.5|
| SGLang  | 40         | 0.0309  | 0.0310 | 3,922.3  | 1,759.2|

### Decode (128 in, 32 out, c=4, 400 requests)

```bash
MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=4 INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh sglang prelude vllm --gpu
```

| Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | In tok/s | Out tok/s | RPM    |
|---------|------------|---------|---------|--------|----------|-----------|--------|
| Prelude | 4          | 0.0300  | 0.0061  | 0.2186 | 2,369.7  | 566.9     | 1,062.9|
| vLLM    | 46         | 0.0331  | 0.0053  | 0.1976 | 2,631.1  | 628.8     | 1,178.9|
| SGLang  | 40         | 0.0606  | 0.0056  | 0.2349 | 2,202.4  | 526.7     | 987.5  |

**Prelude vs vLLM**: 0.90x throughput, 0.91x TTFT, 12x faster startup.
**Prelude vs SGLang**: 1.08x throughput, 0.50x TTFT, 10x faster startup.

## H200 — Qwen/Qwen3-32B (BF16)

### Decode (128 in, 32 out, c=4, 400 requests)

```bash
MODEL=Qwen/Qwen3-32B CUDA_VISIBLE_DEVICES=4 INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh sglang prelude vllm --gpu
```

| Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | In tok/s | Out tok/s | RPM    |
|---------|------------|---------|---------|--------|----------|-----------|--------|
| Prelude | 8          | 0.0899  | 0.0203  | 0.7189 | 732.8    | 175.2     | 328.4  |
| vLLM    | 58         | 0.0875  | 0.0193  | 0.6868 | 770.2    | 184.1     | 345.1  |
| SGLang  | 80         | 0.1001  | 0.0195  | 0.7059 | 750.6    | 179.4     | 336.4  |

**Prelude vs vLLM**: 0.95x throughput, 1.03x TTFT, 7x faster startup.
**Prelude vs SGLang**: 0.98x throughput, 0.90x TTFT, 10x faster startup.

---

## B300 — Qwen/Qwen3-0.6B (BF16)

### Prefill-Only (128 in, 1 out, c=1, 100 requests)

```bash
MODEL=Qwen/Qwen3-0.6B CUDA_VISIBLE_DEVICES=<N> INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=100 CONCURRENCY=1 \
  ./benchmark/bench.sh prelude vllm sglang --gpu
```

| Engine  | Startup(s) | TTFT(s) | E2E(s) | In tok/s | RPM    |
|---------|------------|---------|--------|----------|--------|
| Prelude |            |         |        |          |        |
| vLLM    |            |         |        |          |        |
| SGLang  |            |         |        |          |        |

### Decode (128 in, 32 out, c=4, 400 requests)

```bash
MODEL=Qwen/Qwen3-0.6B CUDA_VISIBLE_DEVICES=<N> INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh prelude vllm sglang --gpu
```

| Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | In tok/s | Out tok/s | RPM    |
|---------|------------|---------|---------|--------|----------|-----------|--------|
| Prelude |            |         |         |        |          |           |        |
| vLLM    |            |         |         |        |          |           |        |
| SGLang  |            |         |         |        |          |           |        |

## B300 — Qwen/Qwen3-8B (BF16)

### Prefill-Only (128 in, 1 out, c=1, 100 requests)

```bash
MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=<N> INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=100 CONCURRENCY=1 \
  ./benchmark/bench.sh prelude vllm sglang --gpu
```

| Engine  | Startup(s) | TTFT(s) | E2E(s) | In tok/s | RPM    |
|---------|------------|---------|--------|----------|--------|
| Prelude |            |         |        |          |        |
| vLLM    |            |         |        |          |        |
| SGLang  |            |         |        |          |        |

### Decode (128 in, 32 out, c=4, 400 requests)

```bash
MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=<N> INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh prelude vllm sglang --gpu
```

| Engine  | Startup(s) | TTFT(s) | TPOT(s) | E2E(s) | In tok/s | Out tok/s | RPM    |
|---------|------------|---------|---------|--------|----------|-----------|--------|
| Prelude |            |         |         |        |          |           |        |
| vLLM    |            |         |         |        |          |           |        |
| SGLang  |            |         |         |        |          |           |        |

> **Note**: Docker images (CUDA 13, default): vLLM `vllm/vllm-openai:latest-cu130`, SGLang `lmsysorg/sglang:latest-cu130`. Use `--cu12` flag for CUDA 12 images.
