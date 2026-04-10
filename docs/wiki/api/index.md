# API Reference

AGInfer exposes an OpenAI-compatible HTTP API. You can use any OpenAI-compatible client library by pointing `base_url` at your server.

**Base URL:** `http://<host>:<port>` (default `http://localhost:8000`)

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | [`/health`](endpoints.md#get-health) | Server health check — always public |
| `GET` | [`/v1/models`](endpoints.md#get-v1models) | List loaded models |
| `POST` | [`/v1/chat/completions`](endpoints.md#post-v1chatcompletions) | Chat completion |
| `POST` | [`/v1/completions`](endpoints.md#post-v1completions) | Text completion |
| `POST` | [`/v1/embeddings`](endpoints.md#post-v1embeddings) | Text embeddings |
| `POST` | [`/v1/classify`](endpoints.md#post-v1classify) | Classification / reranking |


## Streaming

Endpoints that support streaming use Server-Sent Events (SSE). Set `"stream": true` in the request body. Each event is a line prefixed `data: ` followed by a JSON chunk. The final event is:

```
data: [DONE]
```

To include token usage in the stream, set `"stream_options": {"include_usage": true}`. A final chunk with an empty `choices` array and populated `usage` field is sent before `[DONE]`.

## Error Format

All errors use a consistent JSON structure:

```json
{
  "error": {
    "message": "prompt is empty",
    "type": "invalid_request_error"
  }
}
```

| HTTP Status | Type | When |
|---|---|---|
| 400 | `invalid_request_error` | Bad parameters, empty prompt, missing chat template |
| 401 | `authentication_error` | Missing or invalid API key |
| 404 | `not_found` | Model not found (`/v1/models/{model}` only) |
| 500 | `internal_error` | Engine error |

## Client Libraries

AGInfer is compatible with the OpenAI SDK, vLLM, and SGLang clients.

=== "Python (OpenAI SDK)"

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="sk-your-key",  # or "EMPTY" if auth is disabled
    )

    # Chat
    response = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
    )
    print(response.choices[0].message.content)

    # Streaming
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "Tell me a joke"}],
        max_tokens=128,
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    ```

=== "Python (requests)"

    ```python
    import requests

    BASE = "http://localhost:8000"
    HEADERS = {"Authorization": "Bearer sk-your-key"}

    # Chat
    resp = requests.post(f"{BASE}/v1/chat/completions", headers=HEADERS, json={
        "model": "Qwen/Qwen3-4B",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
    })
    print(resp.json()["choices"][0]["message"]["content"])

    # Classification (not in OpenAI SDK)
    resp = requests.post(f"{BASE}/v1/classify", headers=HEADERS, json={
        "model": "Qwen/Qwen3-Reranker",
        "input": ["Good product!", "Bad service."],
    })
    print(resp.json())
    ```

### Compatibility Notes

- `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models` follow the OpenAI API spec
- `/health` follows the vLLM health check convention
- Logprob format matches vLLM/SGLang: completions use the flat-dict format, chat uses the structured list with `bytes`
- Streaming SSE format and `[DONE]` sentinel are identical to OpenAI/vLLM

