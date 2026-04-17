<p align="center">
  <h1> Prelude: Agent-native inference framework </h1>
</p>
sure
## Quick Start

### Prerequisites

- **Rust** (stable, 1.85+)
- **CUDA Toolkit** (for GPU)
- **CMake** >= 3.18 (for oneDNN CPU backend)

### Build

```bash
# GPU — full stack (recommended): FlashInfer + FA4 + DeepGEMM + oneDNN
cargo build -p prelude-server --release --features flashinfer-v4,onednn,deepgemm

# GPU — FlashInfer only (no FA4)
cargo build -p prelude-server --release --features flashinfer,onednn

# CPU only with oneDNN BF16 GEMM
cargo build -p prelude-server --release --features onednn

# GGUF models (auto-detected, no extra flags needed)
cargo build -p prelude-server --release --features onednn
```

### Run

```bash
# GPU
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server \
  --model Qwen/Qwen3-4B --port 8000

# CPU
PRELUDE_DEVICE=cpu ./target/release/prelude-server \
  --model Qwen/Qwen3-0.6B --port 8000
```

### Query

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

## Supported Models


| Name | Structure | Completion | Classify | Embed | GPU Backend |
|---|---|---|---|---|---|
| Qwen3 (0.6B–32B) | Dense | ✔ | ✔ | ✔ | FlashInfer / FA4 |
| Qwen3-MoE (30B-A3B) | MoE | ✔ | — | — | FlashInfer / FA4 |
| Qwen3-Next (80B-A3B) | Hybrid (DeltaNet + MoE) | ✔ | — | — | FlashInfer / FA4 |
| Qwen3.5 (0.8B–27B) | Hybrid (DeltaNet) | ✔ | — | — | FlashInfer / FA4 |
| Qwen3.5-MoE (35B-A3B) | Hybrid (DeltaNet + MoE) | ✔ | — | — | FlashInfer / FA4 |
| Gemma3 | Dense | ✔ | ✔ | ✔ | FlashInfer / FA4 |
| GGUF (Qwen3, Qwen3.5, LLaMA, Gemma, Phi3) | Quantized | ✔ | — | — | CUDA / CPU |

GGUF models are auto-detected from HuggingFace Hub or local `.gguf` files.


## API

OpenAI-compatible endpoints:


| Endpoint                    | Description     |
| --------------------------- | --------------- |
| `POST /v1/chat/completions` | Chat completion |
| `POST /v1/completions`      | Text completion |
| `POST /v1/embeddings`       | Text embeddings |
| `POST /classify`            | Classification  |
| `GET /v1/models`            | List models     |
| `GET /health`               | Health check    |


Supports `logprobs`, `top_logprobs`, `prompt_logprobs`, `stop` sequences, and `stream` mode. Compatible with OpenAI SDK, vLLM, and SGLang clients.

## Architecture

```
Request -> Continuous Batching Scheduler -> GPU Queue -> GPU Worker -> Response
```

**Attention**: FA4 (prefill + decode) -> FlashInfer (fallback) -> CPU fallback. CUDA graph decode. One file per backend, zero `#[cfg]` in model code.

**GEMM**: CUTLASS (SM80+) / DeepGEMM (SM90+ BF16) / oneDNN (CPU BF16). No cuBLAS dependency.

**Runtime**: Paged KV cache, prefix caching, fused CUDA kernels (QKNorm+RoPE, SiLU*Mul, Add+RMSNorm), pure Rust AVX-512 CPU kernels.

## License

Apache-2.0