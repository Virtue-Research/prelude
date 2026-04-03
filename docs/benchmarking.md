# Benchmarking

The benchmark suite lives in `benchmark/` with its own [README](../benchmark/README.md).

## Quick Start

```bash
# Start server
MODEL=Qwen/Qwen3-4B GPU=0 ./benchmark/serve_prelude.sh

# Run a preset
python benchmark/benchmark.py --config benchmark/presets/complete_prefill.toml \
  --model Qwen/Qwen3-4B --url http://localhost:8000
```

## Available Presets

| Preset | Task | Data | What it measures |
|--------|------|------|-----------------|
| `classify.toml` | classify | synthetic 256-768 tok | Classification throughput |
| `classify_realworld.toml` | classify | openai/gsm8k | Classification on real data |
| `embed.toml` | embed | synthetic 256-768 tok | Embedding throughput |
| `embed_realworld.toml` | embed | openai/gsm8k | Embedding on real data |
| `complete_prefill.toml` | complete | synthetic 256-768 tok | Prefill throughput (max_tokens=1) |
| `complete_generation.toml` | complete | openai/gsm8k | Generation with TTFT/TPOT |
| `complete_generation_syn.toml` | complete | synthetic 256-768 tok | Generation with synthetic data |
| `prefix_cache.toml` | complete | synthetic + 3K prefix | Prefix cache hit benefit |
| `mix_prefill_gen.toml` | mix | synthetic | Mixed prefill + generation |

All presets require `--model` and `--url` on the command line.

## Comparing Against vLLM / SGLang

```bash
# Start all three servers (different ports)
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

## Understanding the Output

```
  Conc   Reqs  Fail        RPS    In-tok/s   Out-tok/s    Avg(ms)    P50(ms)    P95(ms)
---------------------------------------------------------------------------------------
     1     50     0      77.09       41292          77      12.88      13.30      14.63
    64    400     0     165.26       83672         165     360.63     375.45     409.51
```

| Column | Meaning |
|--------|---------|
| Conc | Concurrency level |
| Reqs | Successful requests |
| RPS | Requests per second |
| In-tok/s | Input (prompt) tokens processed per second |
| Out-tok/s | Output (completion) tokens generated per second |
| Avg/P50/P95 | Latency percentiles |

For prefill benchmarks (max_tokens=1), **In-tok/s** is the key metric.
For generation benchmarks, look at **Out-tok/s**, **TTFT**, and **TPOT** (printed per-level).

## Custom Benchmarks

```bash
# Tight token range for stable results
python benchmark/benchmark.py complete \
  --model Qwen/Qwen3-4B --prompt-tokens 500-530 --max-tokens 1

# Generation with streaming (TTFT + TPOT)
python benchmark/benchmark.py complete \
  --model Qwen/Qwen3-4B --prompt-tokens 256-768 --max-tokens 128 --streaming

# Prefix cache test
python benchmark/benchmark.py complete \
  --model Qwen/Qwen3-4B --max-tokens 1 \
  --prefix --prefix-tokens 3072 --num-unique-prefixes 1
```

See `benchmark/README.md` for the full CLI reference.

## Micro-Benchmarks (Kernel-Level)

Low-level operator benchmarks live as binary targets in `prelude-core`.
Each requires specific feature flags:

```bash
# CPU operator benchmarks (GEMM, attention, RMSNorm, RoPE, SiLU) — no special features needed
cargo run --release --bin cpu_ops_bench -p prelude-core

# GPU operator benchmarks (GEMM dispatch, elementwise, etc.) — requires CUDA
cargo run --release --bin gpu_ops_bench -p prelude-core --features cuda

# Tokenizer benchmark (fastokens vs HuggingFace tokenizers comparison)
cargo run --release --bin tokenizer_bench -p prelude-core --features hf_tokenizer -- \
  --model-path /path/to/model

# Qwen3 end-to-end single-model benchmark — requires CUDA
cargo run --release --bin qwen3_bench -p prelude-core --features cuda

# Fused ops correctness test — requires CUDA
cargo run --release --bin fused_ops_test -p prelude-core --features cuda

# Gemma3 model test — requires CUDA
cargo run --release --bin gemma3_test -p prelude-core --features cuda
```

These binaries use `required-features` in `Cargo.toml`, so `cargo check` will
skip them when the needed features are not enabled.
