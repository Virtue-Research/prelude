# Release Checklist

Run these checks before every release. All must pass.

## 1. Build (full features)

```bash
cargo build -p prelude-server --release --features full
```

## 2. Unit tests

```bash
cargo test --workspace --release
```

## 3. Accuracy tests (on a GPU host)

All tests run from the project root. Pick a free GPU first:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

### 3.0 Benchmark-guide endpoint suite

Unified numerical regression for `/v1/completions`, `/v1/classify`, and `/v1/embeddings`.
The runner auto-starts Prelude sequentially per model unless `--server-url` is used for a
single endpoint. Reports are written to `temp/accuracy/benchmark-guide-endpoint-accuracy-<timestamp>.json`.

```bash
# Default models:
#   completion = Qwen/Qwen3-0.6B
#   classify   = tomaarsen/Qwen3-Reranker-0.6B-seq-cls
#   embedding  = Qwen/Qwen3-Embedding-0.6B
CUDA_VISIBLE_DEVICES=<N> PRELUDE_DEVICE=auto \
  python3 tests/accuracy/test_endpoint_accuracy.py

# Single endpoint against a pre-started server
python3 tests/accuracy/test_endpoint_accuracy.py \
  --endpoint classify \
  --server-url http://localhost:8001
```

Note: this suite uses only publicly resolvable models. The classify default is
`tomaarsen/Qwen3-Reranker-0.6B-seq-cls`.

### 3.1 Qwen3 (dense, 0.6B)

```bash
# CPU BF16 (oneDNN path)
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-bf16 \
  --server prelude --model Qwen/Qwen3-0.6B

# GPU BF16 (flash-attn-v3 + paged-attn, with ScheduledEngine)
CUDA_VISIBLE_DEVICES=<N> .venv/bin/python tests/accuracy/run_accuracy_test.py --variant gpu \
  --server prelude --model Qwen/Qwen3-0.6B
```

**Baseline**: CPU bf16 8/8, GPU bf16 8/8.

### 3.2 Qwen3.5 Dense (0.8B)

```bash
# CPU F32 (PRELUDE_NO_SCHEDULER=1)
.venv/bin/python tests/qwen3_5/test_accuracy.py \
  --model /path/to/models/qwen3.5-0.8b

# GPU BF16 (PRELUDE_NO_SCHEDULER=1)
CUDA_VISIBLE_DEVICES=<N> .venv/bin/python tests/qwen3_5/test_accuracy.py \
  --model /path/to/models/qwen3.5-0.8b --gpu

# GPU BF16 with ScheduledEngine (pre-start server)
CUDA_VISIBLE_DEVICES=<N> PRELUDE_DEVICE=auto ./target/release/prelude-server \
  --model /path/to/models/qwen3.5-0.8b \
  --host 0.0.0.0 --port 8001 --max-running-requests 1
# In another terminal:
.venv/bin/python tests/qwen3_5/test_accuracy.py --server-url http://localhost:8001

# GPU BF16 with ScheduledEngine + batched decode (DeltaNet pool)
CUDA_VISIBLE_DEVICES=<N> PRELUDE_DEVICE=auto ./target/release/prelude-server \
  --model /path/to/models/qwen3.5-0.8b \
  --host 0.0.0.0 --port 8001 --max-running-requests 4
# In another terminal:
.venv/bin/python tests/qwen3_5/test_accuracy.py --server-url http://localhost:8001
```

**Baseline**: CPU F32 7/7, GPU BF16 7/7.

### 3.3 Qwen3.5 MoE (35B-A3B)

```bash
# CPU F32 (PRELUDE_NO_SCHEDULER=1)
.venv/bin/python tests/qwen3_5_moe/test_accuracy.py \
  --model /path/to/models/qwen3.5-35b-a3b

# GPU BF16 (PRELUDE_NO_SCHEDULER=1)
CUDA_VISIBLE_DEVICES=<N> .venv/bin/python tests/qwen3_5_moe/test_accuracy.py \
  --model /path/to/models/qwen3.5-35b-a3b --gpu

# GPU BF16 with ScheduledEngine
CUDA_VISIBLE_DEVICES=<N> PRELUDE_DEVICE=auto ./target/release/prelude-server \
  --model /path/to/models/qwen3.5-35b-a3b \
  --host 0.0.0.0 --port 8001 --max-running-requests 1
# In another terminal:
.venv/bin/python tests/qwen3_5_moe/test_accuracy.py --server-url http://localhost:8001

# GPU BF16 with ScheduledEngine + batched decode (DeltaNet pool)
CUDA_VISIBLE_DEVICES=<N> PRELUDE_DEVICE=auto ./target/release/prelude-server \
  --model /path/to/models/qwen3.5-35b-a3b \
  --host 0.0.0.0 --port 8001 --max-running-requests 4
# In another terminal:
.venv/bin/python tests/qwen3_5_moe/test_accuracy.py --server-url http://localhost:8001
```

**Baseline**: CPU F32 7/7, GPU BF16 7/7.

### 3.4 Qwen3-Next (80B-A3B)

```bash
# Step 1: Generate golden reference (one-time, slow — 80B model load)
.venv/bin/python tests/qwen3_next/test_accuracy.py --generate-golden \
  --model /path/to/models/qwen3-next

# CPU F32 (PRELUDE_NO_SCHEDULER=1, auto-start server)
.venv/bin/python tests/qwen3_next/test_accuracy.py \
  --model /path/to/models/qwen3-next

# GPU BF16 with ScheduledEngine
CUDA_VISIBLE_DEVICES=<N> PRELUDE_DEVICE=auto ./target/release/prelude-server \
  --model /path/to/models/qwen3-next \
  --host 0.0.0.0 --port 8001 --max-running-requests 1
# In another terminal:
.venv/bin/python tests/qwen3_next/test_accuracy.py --server-url http://localhost:8001

# GPU BF16 with ScheduledEngine + batched decode (DeltaNet pool)
CUDA_VISIBLE_DEVICES=<N> PRELUDE_DEVICE=auto ./target/release/prelude-server \
  --model /path/to/models/qwen3-next \
  --host 0.0.0.0 --port 8001 --max-running-requests 4
# In another terminal:
.venv/bin/python tests/qwen3_next/test_accuracy.py --server-url http://localhost:8001
```

**Baseline**: CPU F32 6/6.

### 3.5 GGUF (universal quantized)

Supports: qwen3, qwen3moe, qwen35 (hybrid DeltaNet), llama, gemma3, phi3, qwen2. Auto-detected from GGUF metadata.

```bash
# Single-model smoke test (Qwen3 Q8_0, CPU)
.venv/bin/python tests/gguf/test_gguf.py

# Multi-architecture test (downloads small models, tests all 5 dense archs on CPU)
.venv/bin/python tests/gguf/test_gguf_models.py

# Accuracy test: GGUF Q8_0 vs transformers F32 (start server first)
PRELUDE_DEVICE=cpu PRELUDE_NO_SCHEDULER=1 ./target/release/prelude-server \
  --model /path/to/gguf-test/qwen3/Qwen3-0.6B-Q8_0.gguf \
  --host 0.0.0.0 --port 8099 &
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-f32 \
  --server prelude=http://localhost:8099 --model Qwen/Qwen3-0.6B

# MoE GGUF (requires GPU — FusedMoeGGUF is CUDA-only)
CUDA_VISIBLE_DEVICES=<N> PRELUDE_DEVICE=auto PRELUDE_NO_SCHEDULER=1 ./target/release/prelude-server \
  --model /path/to/gguf-test/qwen3-moe/Qwen3-30B-A3B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8099 &
# Then send a test request:
curl http://localhost:8099/v1/completions -d '{"model":"test","prompt":"Hello","max_tokens":10,"temperature":0}'
```

```bash
# Qwen3.5 GGUF (hybrid DeltaNet, CPU)
huggingface-cli download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q8_0.gguf --local-dir /tmp/gguf-test
PRELUDE_DEVICE=cpu ./target/release/prelude-server \
  --model-path /tmp/gguf-test/Qwen3.5-0.8B-Q8_0.gguf \
  --model unsloth/Qwen3.5-0.8B \
  --host 0.0.0.0 --port 8099 &
curl http://localhost:8099/v1/completions -d '{"model":"test","prompt":"Hello","max_tokens":20,"temperature":0}'
```

**Baseline**: smoke=3/3, multi-arch=5/5 (qwen3+qwen2+llama+gemma3+phi3), accuracy=8/8 (Q8 vs F32), MoE=PASS (GPU).

### Accuracy test summary

| Model                 | CPU F32 | CPU BF16 | GPU (no-sched) | GPU + Scheduler | GPU + Scheduler (batch) |
|-----------------------|---------|----------|----------------|-----------------|-------------------------|
| Qwen3-0.6B            | 8/8     | 8/8      | 8/8            | 8/8             | 8/8                     |
| Qwen3.5-0.8B (dense)  | 7/7†    | —        | 7/7            | 7/7             | 7/7                     |
| Qwen3.5-35B-A3B (MoE) | 7/7†    | —        | 7/7            | 7/7             | 7/7                     |
| Qwen3-Next-80B-A3B    | 6/6†    | —        | untested       | N/A (OOM)       | N/A (OOM)               |
| GGUF Qwen3 Q8_0       | 8/8‡    | —        | 8/8‡‡          | —               | —                       |
| GGUF multi-arch       | 5/5     | —        | —              | —               | —                       |
| GGUF Qwen3-MoE Q4_K_M | —       | —        | 8/8§           | —               | —                       |

† F32 golden reference. All other GPU tests use BF16 golden (matching vLLM convention: `hf_dtype="auto"`).
‡ GGUF Q8 vs transformers F32, CPU: exact=2, close=6 (quantization drift, all pass top-5 cross-containment).
‡‡ GGUF Q8 vs transformers F32, GPU: exact=3, close=5 (quantization drift, all pass top-5 cross-containment).
§ GGUF MoE Q4_K_M vs transformers F32, GPU: exact=1, close=7 (Q4 quantization drift larger than Q8, all pass bidir cross-containment).

**GPU (no-sched)** = `PRELUDE_NO_SCHEDULER=1` with BF16 golden → tests model correctness without attention kernel differences.
**GPU + Scheduler** = ScheduledEngine with flash-attn-v3 + paged-attn. Divergences are from different
attention backend numerics vs HF transformers, not model bugs. Logprobs now supported for bidirectional cross-containment.
**GPU + Scheduler (batch)** = `--max-running-requests 4` → tests DeltaNet state pool batched decode.
Results identical to single-request scheduler, confirming pool introduces zero accuracy regression.
**N/A (OOM)** = Qwen3-Next 80B requires ~160GB BF16, exceeds single H200 (144GB). Needs multi-GPU TP (not supported).
**Cross-containment** = Bidirectional: server's diverging token in ref's top-5 AND ref's diverging token in server's top-5.
**GGUF MoE** = `quantized_qwen3_moe` uses `FusedMoeGGUF` which requires CUDA. Dense GGUF architectures work on both CPU and GPU.

## 4. Performance benchmarks (on h200)

Run after any change to inference hot paths. Compare against previous results in `docs/results.md`.

```bash
# CPU prefill
INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=4 \
  ./benchmark/bench.sh prelude --cpu

# CPU normal
INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=10 CONCURRENCY=1 \
  ./benchmark/bench.sh prelude --cpu

# GPU prefill
CUDA_VISIBLE_DEVICES=<N> INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=4 \
  ./benchmark/bench.sh prelude --gpu

# GPU normal
CUDA_VISIBLE_DEVICES=<N> INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=10 CONCURRENCY=1 \
  ./benchmark/bench.sh prelude --gpu
```

Compare against previous results in `docs/results.md`.
Watch for regressions: >5% drop in throughput or >10% increase in latency warrants investigation.
