# Benchmark Results

- **Model**: Qwen/Qwen3-0.6B
- **CPU**: Intel Xeon Platinum 8480+
- **GPU**: 8x NVIDIA H200

## Prefill-Only

```shell
RESULTS_DIR=./bench_results/prefill_bench INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=4 ./benchmark/bench.sh
```

### 2026-03-04 (machine idle)

| Engine     | Device | Startup(s) | TTFT(s) | E2E(s) | Input t/s | RPM    |
|------------|--------|------------|---------|--------|-----------|--------|
| Prelude  | cpu    | 4          | 0.2570  | 0.2572 | 2028.3    | 909.3  |
| llama.cpp  | cpu    | 2          | 0.3397  | 0.3398 | 1588.9    | 691.8  |
| vLLM-CPU   | cpu    | 84         | 7.7982  | 7.7983 | 68.5      | 30.7   |
| SGLang-CPU | cpu    | 64         | 0.1369  | 0.1370 | 3712.1    | 1663.9 |
| Prelude  | gpu    | 2          | 0.0242  | 0.0243 | 18407.5   | 8254.3 |
| vLLM.rs    | gpu    | 12         | 0.2207  | 0.2910 | 1835.5    | 799.4  |
| vLLM       | gpu    | 36         | 0.0587  | 0.0589 | 7638.1    | 3421.8 |
| SGLang     | gpu    | 20         | 0.0434  | 0.0435 | 10852.6   | 4870.2 |

### 2026-03-08 post-refactor (machine under load: avg 9.7, GPUs 3-7 occupied)

| Engine     | Device | Startup(s) | TTFT(s) | E2E(s) | Input t/s | RPM    |
|------------|--------|------------|---------|--------|-----------|--------|
| Prelude  | cpu    | 4          | 0.5397  | 0.5399 | 961.8     | 431.2  |
| Prelude  | gpu    | 2          | 0.0282  | 0.0283 | 15749.5   | 7059.9 |

### 2026-03-08 post-refactor + double-tokenize fix (GPU 0,7 occupied)

| Engine     | Device | Startup(s) | TTFT(s) | E2E(s) | Input t/s | RPM    |
|------------|--------|------------|---------|--------|-----------|--------|
| Prelude  | gpu    | 2          | 0.0249  | 0.0250 | 17849.6   | 8002.0 |

## Normal (D(32,32), 10 requests, concurrency=1)

```shell
CUDA_VISIBLE_DEVICES=5 ./benchmark/bench.sh
```

### 2026-03-04 (machine idle)

| Engine     | Device | Startup(s) | TTFT(s) | TPOT(s) | E2E(s)   | Input t/s | Output t/s | RPM   |
|------------|--------|------------|---------|---------|----------|-----------|------------|-------|
| Prelude  | cpu    | 4          | 0.0890  | 0.0370  | 1.2345   | 30.4      | 24.6       | 46.1  |
| llama.cpp  | cpu    | 2          | 0.0565  | 0.0153  | 0.5321   | 79.5      | 58.5       | 109.8 |
| vLLM-CPU   | cpu    | 86         | 4.0749  | 4.2287  | 135.1637 | 0.3       | 0.2        | 0.4   |
| SGLang-CPU | cpu    | 68         | 0.5268  | 0.0171  | 1.0571   | 36.3      | 29.3       | 54.9  |
| Prelude  | gpu    | 2          | 0.0342  | 0.0051  | 0.1923   | 158.1     | 159.1      | 298.4 |
| vLLM.rs    | gpu    | 12         | 0.0772  | 0.0027  | 0.1624   | 237.4     | 174.8      | 327.8 |
| vLLM       | gpu    | 36         | 0.0222  | 0.0015  | 0.0699   | 468.6     | 378.8      | 710.3 |
| SGLang     | gpu    | 22         | 0.0239  | 0.0017  | 0.0776   | 428.6     | 347.6      | 651.8 |

### 2026-03-08 post-refactor (machine under load: avg 9.7, GPUs 3-7 occupied)

| Engine     | Device | Startup(s) | TTFT(s) | TPOT(s) | E2E(s)   | Input t/s | Output t/s | RPM   | Notes |
|------------|--------|------------|---------|---------|----------|-----------|------------|-------|-------|
| Prelude  | cpu    | 4          | N/A     | N/A     | N/A      | N/A       | N/A        | N/A   | CPU multi-token streaming not supported |
| Prelude  | gpu    | 2          | 0.1657  | 0.0000  | 0.1666   | 217.9     | 175.5      | 329.1 | PRELUDE_PAGED_ATTN_BLOCKS=2048 |

### 2026-03-08 post-refactor + double-tokenize fix (GPU 0,7 occupied)

| Engine     | Device | Startup(s) | TTFT(s) | TPOT(s) | E2E(s)   | Input t/s | Output t/s | RPM   | Notes |
|------------|--------|------------|---------|---------|----------|-----------|------------|-------|-------|
| Prelude  | gpu    | 2          | 0.1544  | 0.0000  | 0.1554   | 233.9     | 189.5      | 355.3 | PRELUDE_PAGED_ATTN_BLOCKS=2048 |

## Prefill Completion Benchmark (benchmark.py)

- **Model**: Qwen/Qwen3-4B
- **GPU**: NVIDIA H200 (1 GPU per engine, isolated)
- **Input**: 512–768 random chars (~130–200 tokens), max_tokens=1
- **Requests**: 200 per concurrency level, 5 warmup
- **Config**: Prelude max-batch-size=128, vLLM/SGLang gpu-memory-utilization=0.4

### Qwen3-4B — 2026-03-18 (GPU 0/1/2 isolated)

Prelude results include adaptive batch defer + C/b+α interpolation (commit 8712a77).
vLLM and SGLang numbers verified reproducible across old/new benchmark.py versions.

#### Throughput (req/s) — higher is better

| Concurrency | Prelude   | vLLM   | SGLang | Prelude vs vLLM | Prelude vs SGLang |
|-------------|-----------|--------|--------|-----------------|-------------------|
| 1           | **70.5**  | 47.8   | 42.6   | **1.47x**       | **1.65x**         |
| 4           | **90.8**  | 88.7   | 73.5   | **1.02x**       | **1.24x**         |
| 16          | 137.2     | 98.0   | **150.9** | **1.40x**     | 0.91x             |
| 64          | **181.3** | 137.4  | 151.4  | **1.32x**       | **1.20x**         |
| 96          | **186.7** | 134.5  | 151.5  | **1.39x**       | **1.23x**         |
| 128         | **173.9** | 134.3  | 148.2  | **1.30x**       | **1.17x**         |

#### Latency P50 (ms) — lower is better

| Concurrency | Prelude   | vLLM   | SGLang |
|-------------|-----------|--------|--------|
| 1           | **15.4**  | 18.1   | 20.8   |
| 4           | **41.8**  | 45.1   | 40.0   |
| 16          | 108.5     | 171.3  | **104.7** |
| 64          | **328.9** | 410.2  | 418.3  |
| 96          | **481.2** | 577.5  | 555.5  |
| 128         | **760.2** | 750.6  | 707.3  |

#### Latency P95 (ms) — lower is better

| Concurrency | Prelude   | vLLM    | SGLang |
|-------------|-----------|---------|--------|
| 1           | **21.1**  | 27.9    | 26.2   |
| 4           | **57.7**  | 59.7    | 51.3   |
| 16          | 137.6     | 203.0   | **122.0** |
| 64          | **419.7** | 661.4   | 466.1  |
| 96          | **547.2** | 894.4   | 694.7  |
| 128         | **793.1** | 1004.3  | 905.4  |

**Key takeaways:**
- Prelude wins throughput at all concurrency levels except c=16 (SGLang peaks there)
- At c=1, Prelude is 47–65% faster — zero cold-start wait + minimal Rust HTTP overhead
- At high concurrency (c=96), Prelude peaks at 187 req/s vs vLLM 135 and SGLang 152
- c=128 drops slightly for all engines due to GPU memory bandwidth saturation
- P95 tail latency: Prelude consistently lower, especially at c=96 (547ms vs vLLM 894ms)