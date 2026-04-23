# Prelude Server

## Configuration

### Command Line Arguments

```bash
prelude-server [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--host <HOST>` | `127.0.0.1` | Server bind address |
| `--port <PORT>` | `8000` | Server port |
| `--model <MODEL>` | `Qwen/Qwen3-0.6B` | HuggingFace repo ID or local path to model dir / `.gguf` file |
| `--max-batch-size <N>` | `32` | Max requests per batch |
| `--max-batch-wait-ms <MS>` | `5` | Max wait time before batch dispatch |
| `--max-running-requests <N>` | `8` | Max concurrent running requests |
| `--max-prefill-tokens <N>` | `4096` | Max prefill tokens per step |
| `--max-total-tokens <N>` | `32768` | Max total tokens across requests |
| `--api-key <KEY>` | - | API key for authentication (repeatable). `/v1/*` routes require `Authorization: Bearer <key>` |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRELUDE_DEVICE` | `auto` | Device selection: `auto`, `cpu`, `cuda`, `cuda:N` |
| `PRELUDE_PAGED_BLOCK_SIZE` | `128`* | Block size for paged KV cache (128 with FA3, 16 otherwise) |
| `CUDA_VISIBLE_DEVICES` | all | GPU device selection |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | - | OpenTelemetry collector endpoint (e.g. `http://localhost:4317`) |
| `OTEL_SERVICE_NAME` | `prelude` | Service name reported to OTel collector |
| `RUST_LOG` | `info` | Log level: `debug`, `info`, `warn`, `error` |
| `PRELUDE_MOCK` | - | Enable mock engine (no model needed, dev only) |
| `PRELUDE_MOCK_LATENCY_MS` | `25` | Simulated latency in mock mode (ms) |
| `PRELUDE_NO_SCHEDULER` | - | Disable scheduler, run engine directly (dev only) |
| `PRELUDE_API_KEY` | - | API key (merged with `--api-key` CLI args) |

### Example Startup Commands

**Text Generation (Qwen3-4B)**:
```bash
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server \
  --model Qwen/Qwen3-4B \
  --max-batch-size 32 \
  --port 8000
```

**Classification**:
```bash
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server \
  --model tomaarsen/Qwen3-Reranker-0.6B-seq-cls \
  --max-batch-size 64 \
  --port 8000
```

**Embeddings**:
```bash
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server \
  --model Qwen/Qwen3-Embedding-0.6B \
  --max-batch-size 64 \
  --port 8000
```

---

## REST API Endpoints

### Health Check

```
GET /health
```

**Response**:
```json
{
  "status": "ready",
  "model": "Qwen/Qwen3-4B",
  "uptime_s": 123.45
}
```

---

### List Models

```
GET /v1/models
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-4B",
      "object": "model",
      "created": 1234567890,
      "owned_by": "prelude"
    }
  ]
}
```

---

### Text Completion

```
POST /v1/completions
```

**Request Body**:
```json
{
  "model": "Qwen/Qwen3-4B",
  "prompt": "The capital of France is",
  "max_tokens": 32,
  "temperature": 0.7,
  "top_p": 1.0,
  "stop": ["\n"],
  "stream": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model identifier |
| `prompt` | string | Yes | - | Input text |
| `max_tokens` | int | No | 256 | Max tokens to generate |
| `temperature` | float | No | 1.0 | Sampling temperature |
| `top_p` | float | No | 1.0 | Nucleus sampling threshold |
| `top_k` | int | No | -1 | Top-k sampling (-1 = disabled) |
| `stop` | string[] | No | [] | Stop sequences |
| `stream` | bool | No | false | Enable SSE streaming |

**Response**:
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "Qwen/Qwen3-4B",
  "choices": [
    {
      "text": " Paris, which is also the largest city.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 10,
    "total_tokens": 16
  }
}
```

---

### Chat Completion

```
POST /v1/chat/completions
```

**Request Body**:
```json
{
  "model": "Qwen/Qwen3-4B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 64,
  "temperature": 0.7,
  "stream": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model identifier |
| `messages` | array | Yes | - | Chat messages array |
| `max_tokens` | int | No | 256 | Max tokens to generate |
| `temperature` | float | No | 1.0 | Sampling temperature |
| `top_p` | float | No | 1.0 | Nucleus sampling threshold |
| `stop` | string[] | No | [] | Stop sequences |
| `stream` | bool | No | false | Enable SSE streaming |

**Message Format**:
```json
{
  "role": "user|assistant|system",
  "content": "message text"
}
```

**Response**:
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen/Qwen3-4B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

**Streaming Response** (SSE):
```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"!"}}]}

data: [DONE]
```

---

### Classification

```
POST /v1/classify
```

**Request Body**:
```json
{
  "model": "test",
  "input": ["This is great!", "This is terrible."]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model identifier |
| `input` | string or string[] | Yes | Text(s) to classify |

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "classification",
      "index": 0,
      "label": "positive",
      "probs": [0.95, 0.05],
      "num_classes": 2
    },
    {
      "object": "classification",
      "index": 1,
      "label": "negative",
      "probs": [0.1, 0.9],
      "num_classes": 2
    }
  ],
  "model": "test",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

---

### Embeddings

```
POST /v1/embeddings
```

**Request Body**:
```json
{
  "model": "test",
  "input": ["Hello world", "How are you?"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model identifier |
| `input` | string or string[] | Yes | Text(s) to embed |

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.123, -0.456, ...]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.789, -0.012, ...]
    }
  ],
  "model": "test",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error"
  }
}
```

| HTTP Status | Type | Description |
|-------------|------|-------------|
| 400 | `invalid_request_error` | Malformed request |
| 401 | `authentication_error` | Missing or invalid API key |
| 404 | `not_found` | Model or endpoint not found |
| 500 | `internal_error` | Server error |
| 501 | `not_implemented` | Endpoint not yet implemented |
| 503 | `unavailable` | Model not ready |

---

## Python Client Examples

### Using requests

```python
import requests

BASE_URL = "http://localhost:8000"

# Completion
resp = requests.post(f"{BASE_URL}/v1/completions", json={
    "model": "Qwen/Qwen3-4B",
    "prompt": "Hello",
    "max_tokens": 32
})
print(resp.json())

# Chat
resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Hi!"}]
})
print(resp.json())

# Classification
resp = requests.post(f"{BASE_URL}/v1/classify", json={
    "model": "test",
    "input": ["Good product!", "Bad service."]
})
print(resp.json())

# Embeddings
resp = requests.post(f"{BASE_URL}/v1/embeddings", json={
    "model": "test",
    "input": ["Hello world"]
})
print(resp.json())
```

### Using OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Completion
response = client.completions.create(
    model="Qwen/Qwen3-4B",
    prompt="The capital of France is",
    max_tokens=32
)
print(response.choices[0].text)

# Chat
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Embeddings
response = client.embeddings.create(
    model="test",
    input=["Hello world"]
)
print(response.data[0].embedding[:5])
```

---

## Benchmarking

Use the included benchmark script:

```bash
# Text generation
python benchmark/benchmark_simple.py --mode generate --model Qwen/Qwen3-4B --workers 32

# Classification
python benchmark/benchmark_simple.py --mode classify --workers 32 --requests 200

# Embeddings
python benchmark/benchmark_simple.py --mode embed --workers 32 --requests 200
```

Options:
- `--mode`: `generate`, `classify`, or `embed`
- `--model`: Model name (default: `test`)
- `--url`: Server URL (default: `http://localhost:8000`)
- `--workers`: Concurrent workers (default: 32)
- `--requests`: Total requests (default: 200)
- `--batch-size`: Batch size for classify/embed (default: 20)
- `--max-tokens`: Max tokens for generation (default: 32)
