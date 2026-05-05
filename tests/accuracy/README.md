# Accuracy Test Guide

This directory contains the endpoint accuracy runners used to compare Prelude against
HuggingFace reference models.

The two main entry points are:

- `test_endpoint_accuracy.py`
  Runs the benchmark-guide endpoint suite for `completion`, `classify`, and `embedding`.
- `run_gpu_stress_accuracy.py`
  Generates broader GPU-only stress cases and optionally runs the same endpoint suite on them.

## Prerequisites

Run all commands from the repo root.

Python dependencies:

```bash
python3 -m pip install -r tests/accuracy/requirements.txt
python3 -m pip install torch transformers requests numpy
```

Build the server:

```bash
PYTHON_EXECUTABLE=$(command -v python3) \
PATH=/usr/local/cuda-12.8/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.8 \
cargo build -p prelude-server --release --features flash-attn-v3,paged-attn
```

Quick syntax smoke test:

```bash
python3 -m py_compile \
  tests/accuracy/test_endpoint_accuracy.py \
  tests/accuracy/run_gpu_stress_accuracy.py
```

## Pick A GPU

Do not type `CUDA_VISIBLE_DEVICES=<N>` literally. Replace it with a real GPU index such as `0` or `2`.

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

Example below uses GPU `2`.

## Benchmark-Guide Endpoint Suite

This is the default regression suite. It auto-starts Prelude once per endpoint/model and prints
human-readable `PASS` / `FAIL` lines as it runs.

Full GPU run:

```bash
CUDA_VISIBLE_DEVICES=2 \
PRELUDE_DEVICE=auto \
PRELUDE_PAGED_ATTN_BLOCKS=2048 \
python3 tests/accuracy/test_endpoint_accuracy.py \
  --endpoint all \
  --binary target/release/prelude-server \
  --output temp/accuracy/benchmark-guide-all-endpoints-gpu.json | tee temp/accuracy/benchmark-guide-all-endpoints-gpu.log
```

Single endpoint:

```bash
CUDA_VISIBLE_DEVICES=2 \
PRELUDE_DEVICE=auto \
PRELUDE_PAGED_ATTN_BLOCKS=2048 \
python3 tests/accuracy/test_endpoint_accuracy.py \
  --endpoint embedding \
  --binary target/release/prelude-server \
  --output temp/accuracy/embedding-gpu.json
```

CPU checks:

```bash
PRELUDE_DEVICE=cpu \
python3 tests/accuracy/test_endpoint_accuracy.py \
  --endpoint completion \
  --binary target/release/prelude-server \
  --output temp/accuracy/completion-cpu.json
```

```bash
PRELUDE_DEVICE=cpu \
python3 tests/accuracy/test_endpoint_accuracy.py \
  --endpoint embedding \
  --binary target/release/prelude-server \
  --output temp/accuracy/embedding-cpu.json
```

## GPU Stress Suite

Generate broader cases only:

```bash
python3 tests/accuracy/run_gpu_stress_accuracy.py \
  --generate-only \
  --samples-per-endpoint 100 \
  --cases-output temp/accuracy/gpu-stress-cases-100.json
```

Run the full GPU stress sweep:

```bash
python3 tests/accuracy/run_gpu_stress_accuracy.py \
  --gpu 2 \
  --samples-per-endpoint 100 \
  --cases-output temp/accuracy/gpu-stress-cases-100.json \
  --output temp/accuracy/gpu-stress-report-100.json | tee temp/accuracy/gpu-stress-report-100.log
```

## What The Terminal Output Means

The endpoint runner prints one line per case:

```text
[completion] model=Qwen/Qwen3-0.6B
  - single_prompt ... PASS
  - batch_prompts ... FAIL
      completion request failed with 503: ...
```

At the end it prints a summary:

```text
Summary
  completion PASS 5/5
  classify   PASS 5/5
  embedding  FAIL 3/4

Overall: FAIL 13/14 cases passed
```

Artifacts:

- JSON report goes to the path passed with `--output`
- `tee ...log` keeps the same terminal output in a text log for later review

## Common Gotchas

`bash: N: No such file or directory`

- You typed `CUDA_VISIBLE_DEVICES=<N>`. Replace `<N>` with a real integer GPU id.

`completion request failed with 503: multi-token decode requires paged attention pool`

- You forgot `PRELUDE_PAGED_ATTN_BLOCKS=2048` on GPU.
- Without it, only single-token completion cases are expected to pass.

HF reference says `(cuda/bfloat16)` in the log

- That means the HuggingFace comparison model is running on GPU.
- If you exported `PRELUDE_DEVICE=auto`, the auto-started Prelude server also uses GPU.

CPU completion is slow

- The HF causal-lm reference is expensive on CPU, especially for long prompts.
- Prefer `--endpoint classify` or `--endpoint embedding` for quick CPU smoke tests.

## Current Expected Results

On the validated GPU path:

- benchmark-guide:
  - `completion`: pass
  - `classify`: pass
  - `embedding`: pass
- 100-sample GPU stress:
  - `completion`: pass
  - `classify`: pass
  - `embedding`: pass for per-item cosine/norm checks; pairwise drift uses a BF16-tolerant default and can be tightened with `PRELUDE_EMBED_PAIRWISE_DRIFT_THRESHOLD`

On CPU:

- `completion` multi-token decode is expected to fail with explicit `503` capability errors
- `completion` single-token should pass
- `embedding` normalization is fixed, but the deeper CPU backbone mismatch is not fully resolved
