# Benchmark

Benchmark suite for inference servers (Prelude, vLLM, SGLang).

## Usage

```bash
# Start server
MODEL=Qwen/Qwen3-4B GPU=0 ./benchmark/prelude/serve_prelude.sh

# Run with a preset
python benchmark/prelude/benchmark.py --config benchmark/prelude/presets/complete_prefill.toml \
  --model Qwen/Qwen3-4B --url http://localhost:8000

# Run inline
python benchmark/prelude/benchmark.py complete --model Qwen/Qwen3-4B --max-tokens 1 --prompt-tokens 500-530
```

Two modes: `--config <preset.toml> --model NAME --url URL`, or `<task> [options]` directly.
Presets don't contain `--model` or `--url` — always pass them on the CLI.

## Tasks

| Task | Endpoint | What it measures |
|------|----------|-----------------|
| `classify` | `/v1/classify` | Batch classification throughput |
| `embed` | `/v1/embeddings` | Batch embedding throughput |
| `complete` | `/v1/completions` | Prefill (`max_tokens=1`) or generation (`max_tokens>1`) |
| `mix` | `/v1/completions` | Prefill + generation mixed by weight |

## Presets

```
presets/
  classify.toml              # synthetic 256-768 tok, batch=20
  classify_realworld.toml    # openai/gsm8k
  embed.toml                 # synthetic 256-768 tok, batch=20
  embed_realworld.toml       # openai/gsm8k
  complete_prefill.toml      # prefill-only, synthetic 256-768 tok
  complete_generation.toml   # generation, gsm8k, max_tokens=128, streaming
  complete_generation_syn.toml # generation, synthetic, max_tokens=128, streaming
  prefix_cache.toml          # prefill-only, 3072-token shared prefix
  mix_prefill_gen.toml       # 70% prefill + 30% gen, streaming
```

## Options

### Common

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:8000` | Server base URL |
| `--model` | `test` | Model name in request body |
| `--concurrency` | `1,4,8,16,32,64,128` | Concurrency levels (comma-separated) |
| `--requests` | `200` | Requests per level: single int or comma-separated list matching `--concurrency` |
| `--warmup` | `5` | Warmup requests |
| `--rerun` | `1` | Repeat N times, report mean/std |
| `--seed` | `42` | RNG seed |
| `--timeout` | `120` | Per-request timeout (seconds) |
| `--output` | — | Save results to JSON |

### Data

| Flag | Description |
|------|-------------|
| `--data synthetic` | (default) Random text, no KV reuse |
| `--data realworld` | HuggingFace dataset (needs `question` column) |
| `--dataset` | Dataset name (default: `openai/gsm8k`) |
| `--prompt-tokens N` | Exact N tokens per prompt (needs tokenizer) |
| `--prompt-tokens N-M` | Uniform random in [N, M] (needs tokenizer) |

Omit `--prompt-tokens` to use random chars (no tokenizer needed).

### Task-specific

| Flag | Tasks | Description |
|------|-------|-------------|
| `--batch-size` | classify, embed | Texts per HTTP request (default: 20) |
| `--max-tokens` | complete, mix | Decode length; 1 = prefill-only (default: 1) |
| `--streaming` | complete, mix | SSE mode, collects TTFT/TPOT (needs max_tokens > 1) |
| `--prefill-weight` | mix | Fraction prefill-only (default: 0.7) |
| `--gen-weight` | mix | Fraction generation (default: 0.3) |
| `--prefix` | complete, mix | Use chat API with shared system prompt |
| `--prefix-tokens` | complete, mix | System prompt token count (N or N-M) |
| `--num-unique-prefixes` | complete, mix | Distinct system prompts to rotate (default: 1) |

## TOML Format

```toml
task = "complete"

concurrency = [1, 4, 8, 16, 32, 64, 128]
requests    = [50, 100, 100, 200, 200, 400, 400]
warmup      = 5
max_tokens  = 1
seed        = 42

[data]
type          = "synthetic"
prompt_tokens = "256-768"

[prefix]
enabled       = true
prefix_tokens = 3072
num_unique    = 1
```

All top-level keys map to CLI flags. `[data]` and `[prefix]` are nested sections.

## Output

Summary table printed to stdout:

```
  Conc   Reqs  Fail        RPS    In-tok/s   Out-tok/s    Avg(ms)    P50(ms)    P95(ms)
---------------------------------------------------------------------------------------
     1     50     0      77.09       41292          77      12.88      13.30      14.63
     4    100     0     109.62       59743         110      35.44      34.59      46.38
    64    400     0     165.26       83672         165     360.63     375.45     409.51
```

| Column | Meaning |
|--------|---------|
| `In-tok/s` | Input (prompt) tokens processed per second |
| `Out-tok/s` | Output (completion) tokens generated per second |
| `RPS` | Requests per second |

For classify/embed, `In-tok/s` and `Out-tok/s` are absent (no token-level usage from server).
For streaming runs, per-level output also shows TTFT and TPOT breakdowns.

JSON output (`--output`):

```json
{
  "timestamp": "...",
  "results": [{
    "task": "complete",
    "concurrency": 64,
    "requests": 400,
    "successful": 400,
    "failed": 0,
    "total_time_s": 2.35,
    "throughput_rps": 170.2,
    "input_tokens": 206000,
    "output_tokens": 400,
    "input_tokens_per_s": 87660,
    "output_tokens_per_s": 170,
    "avg_latency_ms": 348.1,
    "p50_ms": 352.6,
    "p95_ms": 435.1,
    "p99_ms": 480.2,
    "extra": {}
  }]
}
```

Streaming runs add `ttft_avg_ms`, `ttft_p50_ms`, `ttft_p95_ms`, `tpot_avg_ms`, `tpot_p50_ms`, `tpot_p95_ms` inside `extra`.
With `--rerun N > 1`, results include `_std` and `_cv` fields for standard deviation and coefficient of variation.

## Server Scripts

```bash
MODEL=Qwen/Qwen3-4B GPU=0 PORT=8000 ./benchmark/prelude/serve_prelude.sh
MODEL=Qwen/Qwen3-4B GPU=0 PORT=8000 ./benchmark/prelude/serve_vllm.sh
MODEL=Qwen/Qwen3-4B GPU=0 PORT=8000 ./benchmark/prelude/serve_sglang.sh
```

| Env var | Default | Description |
|---------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-0.6B` | Model path or HF repo ID |
| `PORT` | `8000` | Server port |
| `GPU` | `0` | `CUDA_VISIBLE_DEVICES` |
| `HOST` | `0.0.0.0` | Bind address |
| `START_TIMEOUT_S` | `180` | Health check timeout |

Extra args: `PRELUDE_EXTRA_ARGS`, `VLLM_EXTRA_ARGS`, `SGLANG_EXTRA_ARGS`.
PID files and logs: `bench_results/runtime/<name>.{pid,log}`.
