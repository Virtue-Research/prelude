# Endpoints

## GET /health

Returns server status. Always public — no authentication required.

```bash
curl http://localhost:8000/health
```

**Response (200 OK):**

```json
{
  "status": "ready",
  "model": "Qwen/Qwen3-4B",
  "uptime_s": 142.38
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ready"` |
| `model` | string | Loaded model ID |
| `uptime_s` | float | Seconds since server start |

---

## GET /v1/models

List all available models.

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-your-key"
```

**Response (200 OK):**

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-4B",
      "object": "model",
      "created": 1710720000,
      "owned_by": "prelude"
    }
  ]
}
```

---

## POST /v1/chat/completions

Generate chat completions from a message list. Requires a model with a chat template (`tokenizer_config.json` or `chat_template.jinja`). Returns HTTP 400 if no chat template is available — use `/v1/completions` instead.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | yes | — | Model ID |
| `messages` | object[] | yes | — | Array of `{"role", "content"}` objects. Roles: `system`, `user`, `assistant` |
| `max_completion_tokens` | int | no | `4096` | Max tokens to generate (preferred name) |
| `max_tokens` | int | no | `4096` | Alias for `max_completion_tokens` (deprecated by OpenAI) |
| `temperature` | float | no | `1.0` | Sampling temperature (0 = greedy) |
| `top_p` | float | no | `1.0` | Nucleus sampling threshold |
| `stop` | string[] | no | — | Stop sequences |
| `stream` | bool | no | `false` | Enable SSE streaming |
| `stream_options.include_usage` | bool | no | `false` | Include usage in final stream chunk |
| `logprobs` | bool | no | `false` | Return logprobs for generated tokens |
| `top_logprobs` | int | no | — | Top-k logprob candidates per token (requires `logprobs: true`) |
| `seed` | int | no | — | Random seed for reproducibility |

**Not supported:** `n > 1`, `frequency_penalty`, `presence_penalty`, `response_format`, `user`, multimodal message content.

### Response (non-streaming, 200 OK)

```json
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1710720000,
  "model": "Qwen/Qwen3-4B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4."
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 8,
    "total_tokens": 32
  }
}
```

When `logprobs: true`, each choice includes a structured `logprobs` object:

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

### Streaming

Each SSE chunk uses `"object": "chat.completion.chunk"` with content in `choices[0].delta.content`. The first chunk has an empty content string with `role: "assistant"`. The final chunk has `finish_reason` set and `content: null`.

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1710720000,"model":"Qwen/Qwen3-4B","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1710720000,"model":"Qwen/Qwen3-4B","choices":[{"index":0,"delta":{"role":"assistant","content":"2 + 2 = 4."},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1710720000,"model":"Qwen/Qwen3-4B","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":"stop"}]}

data: [DONE]
```

---

## POST /v1/completions

Generate text completions from a prompt. Supports batch prompts and streaming.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "prompt": "The capital of France is",
    "max_tokens": 32,
    "temperature": 0.0
  }'
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | yes | — | Model ID |
| `prompt` | string or string[] | yes | — | Input text. A string array sends a batch |
| `max_tokens` | int | no | `4096` | Max tokens to generate |
| `temperature` | float | no | `1.0` | Sampling temperature (0 = greedy) |
| `top_p` | float | no | `1.0` | Nucleus sampling threshold |
| `stop` | string[] | no | — | Stop sequences |
| `stream` | bool | no | `false` | Enable SSE streaming (single prompt only) |
| `stream_options.include_usage` | bool | no | `false` | Include usage in final stream chunk |
| `logprobs` | int | no | — | Return top-N logprobs per generated token |
| `prompt_logprobs` | int | no | — | Return top-N logprobs per prompt token (vLLM extension) |
| `seed` | int | no | — | Random seed |

**Not supported:** `n > 1`, `frequency_penalty`, `presence_penalty`, `user`.

### Response (non-streaming, 200 OK)

```json
{
  "id": "cmpl-a1b2c3d4e5f6",
  "object": "text_completion",
  "created": 1710720000,
  "model": "Qwen/Qwen3-4B",
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

When `logprobs` is set, each choice includes a flat-dict `logprobs` object:

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

When `prompt_logprobs` is set, each choice also includes `prompt_logprobs` (one entry per prompt token, first entry always `null`) and `prompt_token_ids`.

### Streaming

Each SSE chunk carries incremental text in `choices[0].text`. The final chunk has an empty `text` and a populated `finish_reason`.

```
data: {"id":"cmpl-...","object":"text_completion","created":1710720000,"model":"Qwen/Qwen3-4B","choices":[{"text":" Paris","index":0,"finish_reason":""}],...}

data: {"id":"cmpl-...","object":"text_completion","created":1710720000,"model":"Qwen/Qwen3-4B","choices":[{"text":"","index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":6,"completion_tokens":1,"total_tokens":7}}

data: [DONE]
```

---

## POST /v1/embeddings

Generate embeddings for input text or token IDs. Requires an embedding model (e.g. `Qwen/Qwen3-Embedding`).

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "Qwen/Qwen3-Embedding",
    "input": ["Search query: what is deep learning?", "Search query: how to train a model"]
  }'
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | yes | — | Embedding model ID |
| `input` | string, string[], int[], or int[][] | yes | — | Text or pre-tokenized token IDs. Arrays produce batch embeddings |
| `encoding_format` | string | no | `"float"` | `"float"` (array of floats) or `"base64"` (raw little-endian f32 bytes, base64-encoded) |

**Not supported:** `dimensions`.

### Response (200 OK)

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023, -0.0091, 0.0152, "..."]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.0041, -0.0078, 0.0183, "..."]
    }
  ],
  "model": "Qwen/Qwen3-Embedding",
  "usage": {
    "prompt_tokens": 20,
    "total_tokens": 20
  }
}
```

---

## POST /v1/classify

Classify input text(s) using a sequence classification model (e.g. `Qwen/Qwen3-Reranker`). Exactly one of `input` or `messages` must be provided.

```bash
curl http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "Qwen/Qwen3-Reranker",
    "input": ["This is a positive review", "This is a negative review"]
  }'
```

### Request Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Classifier model ID |
| `input` | string, string[], int[], or int[][] | one of | Text or pre-tokenized inputs. Mutually exclusive with `messages` |
| `messages` | object[] | one of | Chat-style `{"role", "content"}` messages, concatenated as `"role: content\n..."` before classification. Mutually exclusive with `input` |

### Response (200 OK)

```json
{
  "id": "classify-a1b2c3d4e5f6",
  "object": "list",
  "created": 1710720000,
  "model": "Qwen/Qwen3-Reranker",
  "data": [
    {
      "index": 0,
      "label": "positive",
      "probs": [0.97, 0.03],
      "num_classes": 2
    },
    {
      "index": 1,
      "label": "negative",
      "probs": [0.08, 0.92],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 0,
    "total_tokens": 16
  }
}
```

| Field | Type | Description |
|---|---|---|
| `data[].index` | int | Position in the input batch |
| `data[].label` | string or null | Predicted label from `id2label` in model config (argmax of probs). `null` if no label mapping |
| `data[].probs` | float[] | Softmax probabilities for each class |
| `data[].num_classes` | int | Number of output classes |
