# CLI Reference

`prelude-server` starts the Prelude HTTP server. It controls **how the server runs** — which model to load, which device to use, how the scheduler is tuned, and what auth to enforce. Once running, clients interact with it entirely through the [HTTP API](../api/index.md).

<!-- ## CLI vs. API

| Concern | Where it lives |
|---|---|
| Which model to load | CLI (`--model`, `--model-path`, `--task`, `--dtype`) |
| Compute device & memory | CLI (`--device`, `--gpu-memory-utilization`, `--cuda-graph`) |
| Scheduler tuning | CLI (`--max-batch-size`, `--max-running-requests`, …) |
| Auth & CORS | CLI (`--api-key`, `--cors-allow-origin`) |
| Per-request sampling | API (`temperature`, `top_p`, `max_tokens`, …) |
| Prompt / messages | API (request body) |
| Streaming, logprobs | API (request body) |

CLI flags are set once at startup and apply globally. API parameters are set per request and override sampling defaults (see `PRELUDE_DEFAULT_*` in [Configuration](../user_guide/configuration.md#sampling-defaults)). -->

```
prelude-server [OPTIONS]
```

```bash
# Minimal — loads Qwen3-0.6B from HuggingFace Hub on port 8000
./target/release/prelude-server

# Typical
./target/release/prelude-server --model Qwen/Qwen3-4B --port 8000
```

---

## Options

For flag descriptions, defaults, and tuning guidance see [Configuration → CLI Flags](../user_guide/configuration.md#cli-flags). A quick index:

### Model

`--model`, `--model-path`, `--task`, `--dtype`

### Server

`--host`, `--port`, `--api-key`, `--cors-allow-origin`

### Device & Memory

`--device`, `--gpu-memory-utilization`, `--cuda-graph`

### Scheduler

`--max-batch-size`, `--max-batch-wait-ms`, `--max-running-requests`, `--max-prefill-tokens`, `--max-total-tokens`, `--decode-reservation-cap`

### Development / Testing

`--pseudo`, `--pseudo-latency-ms`

---

## Examples

**Local GGUF file:**
```bash
./target/release/prelude-server \
  --model Qwen/Qwen3-4B \
  --model-path /data/models/Qwen3-4B-Q4_K_M.gguf
```

**Specific GPU with auth:**
```bash
CUDA_VISIBLE_DEVICES=1 ./target/release/prelude-server \
  --model meta-llama/Llama-3.2-1B \
  --device cuda:0 \
  --api-key secret123
```

**Mock server for testing (no GPU needed):**
```bash
./target/release/prelude-server --pseudo --pseudo-latency-ms 10
```

For low-latency, high-throughput, and long-context scheduler presets, see [Configuration → Scheduler Examples](../user_guide/configuration.md#examples).

---

## Environment Variables

See [Configuration → Environment Variables](../user_guide/configuration.md#environment-variables) for the full reference.

Key variables at a glance:

| Variable | What it does |
|---|---|
| `PRELUDE_API_KEY` | Merged with `--api-key`; no auth if both are empty |
| `CUDA_VISIBLE_DEVICES` | Standard CUDA device visibility |
| `RUST_LOG` | Log level (default `info`) |
| `PRELUDE_NO_SCHEDULER` | Set `1` to bypass scheduler (single-request mode) |
| `PRELUDE_MOCK` | Set `1` — equivalent to `--pseudo` |

