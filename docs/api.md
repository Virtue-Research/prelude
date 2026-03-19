# Prelude API Reference

Prelude exposes an OpenAI-compatible HTTP API. All endpoints listed below are served by a single `prelude-server` process.

Base URL: `http://<host>:<port>` (default `http://127.0.0.1:8000`)

---

## Authentication

Authentication is optional. When the server is started with `--api-key` or the `PRELUDE_API_KEY` environment variable, all `/v1/*` routes require a Bearer token. The `/health` endpoint is always public.

```
Authorization: Bearer sk-your-key
```

Multiple keys can be provided by repeating `--api-key` on the command line. Any valid key is accepted.

Error response when authentication fails (HTTP 401):

```json
{
  "error": {
    "message": "Missing API key. Expected header: Authorization: Bearer <key>",
    "type": "authentication_error"
  }
}
```

---

## Streaming Format

Endpoints that support streaming (`/v1/completions`, `/v1/chat/completions`) use Server-Sent Events (SSE). Set `"stream": true` in the request body.

Each event is a line prefixed with `data: ` followed by a JSON object. The final event is:

```
data: [DONE]
```

To receive token usage in the stream, set `"stream_options": {"include_usage": true}`. When enabled, a final chunk with an empty `choices` array and populated `usage` field is sent before `[DONE]`.

---

## Endpoints

### GET /health

Returns server health status. Always public (no authentication required).

```bash
curl http://localhost:8000/health
```

**Response** (200 OK):

```json
{
  "status": "ready",
  "model": "Qwen/Qwen3-0.6B",
  "uptime_s": 142.38
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ready"` |
| `model` | string | Loaded model ID |
| `uptime_s` | float | Seconds since server start |

---

### GET /v1/models

List all available models.

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-your-key"
```

**Response** (200 OK):

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-0.6B",
      "object": "model",
      "created": 1710720000,
      "owned_by": "prelude"
    }
  ]
}
```

---

### GET /v1/models/{model}

Retrieve a single model by ID.

```bash
curl http://localhost:8000/v1/models/Qwen/Qwen3-0.6B \
  -H "Authorization: Bearer sk-your-key"
```

**Response** (200 OK):

```json
{
  "id": "Qwen/Qwen3-0.6B",
  "object": "model",
  "created": 1710720000,
  "owned_by": "prelude"
}
```

Returns HTTP 404 if the model ID does not match.

---

### POST /v1/completions

Generate text completions from a prompt. Supports batch prompts and streaming.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "The capital of France is",
    "max_tokens": 32,
    "temperature": 0.0,
    "stream": false
  }'
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | yes | Model ID to use |
| `prompt` | string or string[] | yes | Input text. A string array sends a batch request |
| `max_tokens` | integer | no | Maximum tokens to generate (default: 4096) |
| `temperature` | float | no | Sampling temperature (default: 0.7). Use 0 for greedy |
| `top_p` | float | no | Nucleus sampling threshold (default: 1.0) |
| `stop` | string[] | no | Stop sequences |
| `stream` | boolean | no | Enable SSE streaming (default: false) |
| `stream_options` | object | no | `{"include_usage": true}` to get usage in stream |
| `logprobs` | integer | no | Number of top log probabilities to return per token |
| `seed` | integer | no | Random seed for reproducibility |

**Note:** `n`, `user`, `frequency_penalty`, and `presence_penalty` are accepted but not yet implemented and will return an error if set.

#### Response (non-streaming, 200 OK)

```json
{
  "id": "cmpl-a1b2c3d4e5f6",
  "object": "text_completion",
  "created": 1710720000,
  "model": "Qwen/Qwen3-0.6B",
  "choices": [
    {
      "text": " Paris, which is known for the Eiffel Tower.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 12,
    "total_tokens": 18
  }
}
```

When `logprobs` is set, each choice includes a `logprobs` object:

```json
{
  "logprobs": {
    "tokens": [" Paris"],
    "token_logprobs": [-0.0012],
    "text_offset": [0],
    "top_logprobs": [{"Paris": -0.0012, "London": -8.21}]
  }
}
```

#### Streaming

Streaming is only supported for single prompts (not batch). Each SSE event contains a `CompletionResponse` chunk with incremental text in `choices[0].text`. The final chunk has an empty `text` and a populated `finish_reason`.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Once upon a time",
    "max_tokens": 64,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

```
data: {"id":"cmpl-...","object":"text_completion","created":1710720000,"model":"Qwen/Qwen3-0.6B","choices":[{"text":" there","index":0,"finish_reason":""}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}

data: {"id":"cmpl-...","object":"text_completion","created":1710720000,"model":"Qwen/Qwen3-0.6B","choices":[{"text":" was","index":0,"finish_reason":""}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}

data: {"id":"cmpl-...","object":"text_completion","created":1710720000,"model":"Qwen/Qwen3-0.6B","choices":[{"text":"","index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":64,"total_tokens":68}}

data: [DONE]
```

Usage fields are zero in token chunks unless `stream_options.include_usage` is true, in which case the final chunk(s) contain the totals.

---

### POST /v1/chat/completions

Generate chat completions from a message list. Supports streaming.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 128,
    "temperature": 0.0,
    "stream": false
  }'
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | yes | Model ID to use |
| `messages` | object[] | yes | Array of `{"role": string, "content": string}` objects |
| `max_completion_tokens` | integer | no | Maximum tokens to generate (default: 4096) |
| `max_tokens` | integer | no | Alias for `max_completion_tokens` (deprecated by OpenAI) |
| `temperature` | float | no | Sampling temperature (default: 0.7) |
| `top_p` | float | no | Nucleus sampling threshold (default: 1.0) |
| `stop` | string[] | no | Stop sequences |
| `stream` | boolean | no | Enable SSE streaming (default: false) |
| `stream_options` | object | no | `{"include_usage": true}` to get usage in stream |
| `logprobs` | boolean | no | Return log probabilities (default: false) |
| `top_logprobs` | integer | no | Number of top log probs per token (requires `logprobs: true`) |
| `seed` | integer | no | Random seed for reproducibility |

**Note:** `n`, `user`, `frequency_penalty`, `presence_penalty`, and `response_format` are accepted but not yet implemented and will return an error if set. Only plain-string message content is supported; multimodal and tool-call content will be rejected.

The model's chat template (from `tokenizer_config.json` or `chat_template.jinja`) is used to format messages. If no chat template is available, the endpoint returns HTTP 400.

#### Response (non-streaming, 200 OK)

```json
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1710720000,
  "model": "Qwen/Qwen3-0.6B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 8,
    "total_tokens": 32
  }
}
```

When `logprobs: true`, each choice includes a `logprobs` object with structured per-token entries:

```json
{
  "logprobs": {
    "content": [
      {
        "token": "2",
        "logprob": -0.0015,
        "bytes": [50],
        "top_logprobs": [
          {"token": "2", "logprob": -0.0015, "bytes": [50]},
          {"token": "The", "logprob": -7.12, "bytes": [84, 104, 101]}
        ]
      }
    ]
  }
}
```

#### Streaming

Streaming chunks use `"object": "chat.completion.chunk"` and carry content in `choices[0].delta.content`. The first chunk has an empty `content` with `role: "assistant"`. The final chunk has `finish_reason` set and `content: null`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 64,
    "stream": true
  }'
```

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1710720000,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1710720000,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1710720000,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":"stop"}]}

data: [DONE]
```

---

### POST /v1/embeddings

Generate embeddings for input text(s).

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "input": "The quick brown fox jumps over the lazy dog."
  }'
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | yes | Model ID to use |
| `input` | string, string[], int[], or int[][] | yes | Text or token IDs to embed. Arrays produce batch embeddings |
| `encoding_format` | string | no | `"float"` (default) or `"base64"` |

**Note:** `dimensions` is accepted but not yet implemented and will return an error if set.

#### Response (200 OK)

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023, -0.0091, 0.0152, "..."]
    }
  ],
  "model": "Qwen/Qwen3-0.6B",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

When `encoding_format` is `"base64"`, the `embedding` field is a base64-encoded string of raw little-endian float32 bytes instead of a float array.

---

### POST /v1/classify

Classify input text(s) using a sequence classification model.

```bash
curl http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "input": "How do I bake a chocolate cake?"
  }'
```

#### Request Parameters

Exactly one of `input` or `messages` must be provided.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | yes | Model ID to use |
| `input` | string, string[], int[], or int[][] | no | Text or token IDs to classify. Arrays produce batch classification |
| `messages` | object[] | no | Chat-style messages, formatted as `"role: content\n..."` before classification |

#### Response (200 OK)

```json
{
  "id": "classify-a1b2c3d4e5f6",
  "object": "list",
  "created": 1710720000,
  "model": "Qwen/Qwen3-0.6B",
  "data": [
    {
      "index": 0,
      "label": "safe",
      "probs": [0.98, 0.02],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8,
    "completion_tokens": 0
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `data[].index` | integer | Position in input batch |
| `data[].label` | string or null | Predicted label from the model's `id2label` mapping (argmax of probs). Null if the model has no label mapping |
| `data[].probs` | float[] | Softmax probabilities for each class |
| `data[].num_classes` | integer | Number of output classes |

---

## Not Yet Implemented

The following endpoints are defined but return HTTP 501:

| Endpoint | Status |
|----------|--------|
| `POST /v1/rerank` | Not implemented |
| `POST /v1/score` | Not implemented |
| `POST /v1/tokenize` | Not implemented |
| `POST /v1/detokenize` | Not implemented |

---

## Python Examples

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-your-key",  # or "none" if auth is disabled
)

# Chat completion
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=64,
)
print(response.choices[0].message.content)

# Text completion
response = client.completions.create(
    model="Qwen/Qwen3-0.6B",
    prompt="The capital of France is",
    max_tokens=32,
)
print(response.choices[0].text)

# Embeddings
response = client.embeddings.create(
    model="Qwen/Qwen3-0.6B",
    input=["Hello world"],
)
print(response.data[0].embedding[:5])

# Streaming
stream = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    max_tokens=128,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### requests library

```python
import requests

BASE = "http://localhost:8000"

# Classification (not in OpenAI SDK -- use requests directly)
resp = requests.post(f"{BASE}/v1/classify", json={
    "model": "Qwen/Qwen3-Reranker",
    "input": ["Good product!", "Bad service."]
})
print(resp.json())

# Embeddings
resp = requests.post(f"{BASE}/v1/embeddings", json={
    "model": "Qwen/Qwen3-0.6B",
    "input": ["Hello world"]
})
print(resp.json())
```

---

## Error Format

All error responses use a consistent JSON structure:

```json
{
  "error": {
    "message": "prompt is empty",
    "type": "invalid_request_error"
  }
}
```

Common HTTP status codes:

| Code | Meaning |
|------|---------|
| 400 | Invalid request (bad parameters, empty prompt, missing chat template) |
| 401 | Authentication failed |
| 404 | Model not found (`/v1/models/{model}` only) |
| 500 | Internal engine error |

---

## Compatibility Notes

Prelude's API is designed for drop-in compatibility with OpenAI, vLLM, and SGLang clients.

**OpenAI client libraries:** Set `base_url` to the Prelude server address. The `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and `/v1/models` endpoints follow the OpenAI API spec.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-your-key",  # or "none" if auth is disabled
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=64,
)
```

**vLLM / SGLang compatibility:**

- `/v1/models` returns the same structure used by vLLM and SGLang
- `/health` follows the vLLM health check convention
- Logprobs format matches vLLM/SGLang: completions use the flat-dict format, chat uses the structured list with `bytes`
- Streaming SSE format and `[DONE]` sentinel are identical

**Known differences from OpenAI:**

- `n > 1` (multiple choices per request) is not supported
- `user`, `frequency_penalty`, `presence_penalty`, and `response_format` are parsed but rejected with an error
- `finish_reason` values are `"stop"` and `"length"` (standard OpenAI values)
- The `/v1/classify` endpoint is Prelude-specific and not part of the OpenAI API
- Only plain-string message content is supported in chat; multimodal content arrays are rejected
