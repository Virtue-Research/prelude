# E2E Benchmark

Dockerized end-to-end inference benchmark. Each engine runs in its own container.

## Prerequisites

```bash
pip install "genai-bench @ git+https://github.com/rucnyz/genai-bench.git"
```

## Usage

```bash
# All engines, both GPU and CPU
./benchmark/genai-bench/bench.sh all

# Single engine (runs both GPU and CPU)
./benchmark/genai-bench/bench.sh prelude
./benchmark/genai-bench/bench.sh vllm
./benchmark/genai-bench/bench.sh sglang
./benchmark/genai-bench/bench.sh llama.cpp        # auto-converts HF model to GGUF
./benchmark/genai-bench/bench.sh vllm.rs

# GPU or CPU only
./benchmark/genai-bench/bench.sh prelude --cpu
./benchmark/genai-bench/bench.sh prelude --gpu

# Multiple engines
./benchmark/genai-bench/bench.sh prelude vllm sglang

# Custom parameters
INPUT_TOKENS=512 OUTPUT_TOKENS=32 MAX_REQUESTS=100 CONCURRENCY=4 \
  ./benchmark/genai-bench/bench.sh prelude vllm
```

## Engines and Docker Images

| Engine    | Default GPU image                          | GPU            | CPU                     |
|-----------|--------------------------------------------|----------------|-------------------------|
| prelude   | `prelude-dev` (root Dockerfile)            | Y              | Y (`--device cpu`)      |
| vllm      | `vllm/vllm-openai:latest-cu130-ubuntu2404` | Y              | Y (needs `vllm-cpu`)    |
| sglang    | `lmsysorg/sglang:latest-cu130`     | Y              | Y (needs `sglang-cpu`)  |
| llama.cpp | `prelude-others` (Dockerfile.others)       | Y (`-ngl 99`)  | Y (`-ngl 0`)            |
| vllm.rs   | `prelude-others` (Dockerfile.others)       | Y              | N                       |

Images are auto-built on first run and reused afterwards. The cu130
defaults run noticeably faster on CUDA-13 hosts than the engines'
default `:latest` tags (which currently ship a cu128 base); override
via `VLLM_IMAGE` / `SGLANG_IMAGE` if you need a specific version. The
top-level `benchmark/bench.sh` also accepts `--cu12` to swap both back
to the `:latest` images.

## GGUF Auto-Conversion

llama.cpp requires GGUF format. bench.sh handles this automatically:

1. Checks `~/.cache/prelude/gguf/<model>/model-BF16.gguf`
2. If missing, converts from HF cache safetensors via `convert_hf_to_gguf.py`
3. Cached for future runs

Requires llama.cpp's convert script. Auto-searched at common paths, or set `LLAMA_CPP_CONVERT`.

## CPU Image Build (one-time)

vLLM and SGLang CPU images need to be built from upstream source:

```bash
# vLLM CPU
cd /path/to/vllm
docker build -f docker/Dockerfile.cpu -t vllm-cpu --target vllm-openai .

# SGLang CPU
cd /path/to/sglang
docker build -f docker/xeon.Dockerfile -t sglang-cpu .
```

## Environment Variables

| Variable       | Default            | Description                    |
|----------------|--------------------|---------------------------------|
| `MODEL`        | `Qwen/Qwen3-0.6B` | HuggingFace model              |
| `INPUT_TOKENS` | `128`              | Input token count              |
| `OUTPUT_TOKENS`| `1`                | Output token count (1=prefill) |
| `MAX_REQUESTS` | `200`              | Total requests                 |
| `CONCURRENCY`  | `1`                | Concurrent requests            |
| `GPU`          | `0`                | GPU device id                  |
| `PORT`         | `8000`             | Server port                    |
| `HF_TOKEN`     | -                  | HuggingFace token (gated models)|
| `LLAMA_CPP_CONVERT` | (auto-detected) | Path to convert_hf_to_gguf.py |
| `VLLM_IMAGE`   | `vllm/vllm-openai:latest-cu130-ubuntu2404` | vLLM Docker image to pull/run |
| `SGLANG_IMAGE` | `lmsysorg/sglang:latest-cu130`     | SGLang Docker image to pull/run |

## Results

Output to `benchmark/genai-bench/results/`, with a summary table printed after all runs:

```
┌────────────────┬─────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬────────┐
│ Engine         │ Device  │ Start(s) │ TTFT(s)  │ TPOT(s)  │ E2E(s)   │ In t/s   │ Out t/s  │ RPM    │
├────────────────┼─────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────┤
│ prelude        │ GPU     │ 12       │ 0.0234   │ 0.0156   │ 0.5123   │ 1234.5   │ 567.8    │ 45.6   │
│ prelude        │ CPU     │ 5        │ 0.1234   │ 0.0856   │ 2.8123   │ 234.5    │ 67.8     │ 8.2    │
│ vllm           │ GPU     │ 45       │ 0.0345   │ 0.0178   │ 0.6012   │ 1100.2   │ 489.3    │ 38.2   │
│ ...            │         │          │          │          │          │          │          │        │
└────────────────┴─────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────┘
```
