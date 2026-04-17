# Running Benchmarks

Prelude has two levels of benchmarking: **end-to-end serving benchmarks** (throughput and latency against a live server) and **micro-benchmarks** (kernel-level operator performance).

## Prerequisites

```bash
# Python venv with benchmark dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r benchmark/requirements.txt

# Docker (for SGLang and vLLM comparison runs)
docker pull lmsysorg/sglang:latest
docker pull vllm/vllm-openai:latest
```

## End-to-End Serving Benchmarks

### Quick start

```bash
source .venv/bin/activate

# Benchmark a single engine
CUDA_VISIBLE_DEVICES=0 ./benchmark/bench.sh prelude --gpu
CUDA_VISIBLE_DEVICES=0 ./benchmark/bench.sh sglang --gpu
CUDA_VISIBLE_DEVICES=0 ./benchmark/bench.sh vllm --gpu

# Benchmark all GPU engines at once
CUDA_VISIBLE_DEVICES=0 ./benchmark/bench.sh --gpu

# CPU engines
./benchmark/bench.sh --cpu
```

SGLang and vLLM run automatically in Docker containers — no pip install needed. The HuggingFace model cache (`~/.cache/huggingface`) is mounted into containers automatically.

### Presets

Use `benchmark.py` with a preset config for reproducible runs:

```bash
# Start the server
MODEL=Qwen/Qwen3-4B GPU=0 ./benchmark/serve_prelude.sh

# Run a preset
python benchmark/benchmark.py --config benchmark/presets/complete_prefill.toml \
  --model Qwen/Qwen3-4B --url http://localhost:8000
```

| Preset | Task | Data | Measures |
|---|---|---|---|
| `complete_prefill.toml` | completion | synthetic 256–768 tok | Prefill throughput (max_tokens=1) |
| `complete_generation.toml` | completion | openai/gsm8k | Generation TTFT/TPOT |
| `complete_generation_syn.toml` | completion | synthetic 256–768 tok | Generation with synthetic data |
| `classify.toml` | classify | synthetic 256–768 tok | Classification throughput |
| `classify_realworld.toml` | classify | openai/gsm8k | Classification on real data |
| `embed.toml` | embed | synthetic 256–768 tok | Embedding throughput |
| `embed_realworld.toml` | embed | openai/gsm8k | Embedding on real data |
| `prefix_cache.toml` | completion | synthetic + 3K prefix | Prefix cache hit benefit |
| `mix_prefill_gen.toml` | mix | synthetic | Mixed prefill + generation |

All presets require `--model` and `--url` on the command line.

### Reproducing the published results

```bash
# Prefill-only (128 in, 1 out)
CUDA_VISIBLE_DEVICES=0 INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=100 CONCURRENCY=1 \
  ./benchmark/bench.sh --gpu

# Decode (32 in, 32 out)
CUDA_VISIBLE_DEVICES=0 INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh --gpu

# Decode (128 in, 32 out)
CUDA_VISIBLE_DEVICES=0 INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh --gpu
```

### Comparing against vLLM and SGLang

```bash
# Start all three servers on different ports
MODEL=Qwen/Qwen3-4B GPU=0 PORT=8000 ./benchmark/serve_prelude.sh
MODEL=Qwen/Qwen3-4B GPU=0 PORT=8001 ./benchmark/serve_vllm.sh
MODEL=Qwen/Qwen3-4B GPU=0 PORT=8002 ./benchmark/serve_sglang.sh

# Run the same preset against each
for port in 8000 8001 8002; do
  python benchmark/benchmark.py --config benchmark/presets/complete_prefill.toml \
    --model Qwen/Qwen3-4B --url http://localhost:$port \
    --output results_${port}.json
done
```

### Custom runs

```bash
# Tight token range for stable results
python benchmark/benchmark.py complete \
  --model Qwen/Qwen3-4B --prompt-tokens 500-530 --max-tokens 1

# Generation with streaming (measures TTFT + TPOT)
python benchmark/benchmark.py complete \
  --model Qwen/Qwen3-4B --prompt-tokens 256-768 --max-tokens 128 --streaming

# Prefix cache benefit
python benchmark/benchmark.py complete \
  --model Qwen/Qwen3-4B --max-tokens 1 \
  --prefix --prefix-tokens 3072 --num-unique-prefixes 1
```

See `benchmark/README.md` for the full CLI reference.

### Reading the output

```
  Conc   Reqs  Fail        RPS    In-tok/s   Out-tok/s    Avg(ms)    P50(ms)    P95(ms)
---------------------------------------------------------------------------------------
     1     50     0      77.09       41292          77      12.88      13.30      14.63
    64    400     0     165.26       83672         165     360.63     375.45     409.51
```

| Column | Meaning |
|---|---|
| Conc | Concurrency level |
| Reqs | Successful requests |
| RPS | Requests per second |
| In-tok/s | Input (prompt) tokens processed per second |
| Out-tok/s | Output (completion) tokens generated per second |
| Avg/P50/P95 | Latency percentiles in milliseconds |

For prefill benchmarks (`max_tokens=1`), **In-tok/s** is the key metric.
For generation benchmarks, focus on **Out-tok/s**, **TTFT**, and **TPOT**.

Results are saved to `bench_results/<timestamp>/summary.csv`.

## Micro-Benchmarks (Kernel-Level)

Low-level operator benchmarks for individual kernels:

```bash
# CPU operators (GEMM, attention, RMSNorm, RoPE, SiLU) — no special features needed
cargo run --release --bin cpu_ops_bench -p prelude-core

# GPU operators (GEMM dispatch, elementwise, etc.) — requires CUDA
cargo run --release --bin gpu_ops_bench -p prelude-core --features cuda

# Tokenizer (fastokens vs HuggingFace tokenizers)
cargo run --release --bin tokenizer_bench -p prelude-core --features hf_tokenizer -- \
  --model-path /path/to/model

# Qwen3 end-to-end single-model benchmark — requires CUDA
cargo run --release --bin qwen3_bench -p prelude-core --features cuda

# Fused ops correctness test — requires CUDA
cargo run --release --bin fused_ops_test -p prelude-core --features cuda
```

## Accuracy Tests

```bash
# GPU
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant gpu \
  --server prelude --binary target/release/prelude-server \
  --model Qwen/Qwen3-0.6B

# CPU F32
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-f32 \
  --server prelude --binary target/release/prelude-server \
  --model Qwen/Qwen3-0.6B --timeout 600

# CPU BF16
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-bf16 \
  --server prelude --binary target/release/prelude-server \
  --model Qwen/Qwen3-0.6B
```

## Tips

- Always use an idle GPU — check `nvidia-smi` before running
- CPU benchmarks are slow — use small traffic (e.g., 16 in / 16 out) and few requests
- `CUDA_VISIBLE_DEVICES` is passed into Docker containers automatically
- Start comparison servers sequentially to avoid concurrent HuggingFace Hub downloads
