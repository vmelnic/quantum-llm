# OpenAI-compatible API

Status: implemented bounded compatibility surface, 2026-09-02.

The server implements common OpenAI request/response shapes. It is not the
OpenAI service and does not promise behavior for unlisted fields.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/models` | list the active model |
| `GET` | `/v1/models/{id}` | retrieve the active model |
| `POST` | `/v1/completions` | text completion |
| `POST` | `/v1/chat/completions` | chat, reasoning, tools and artifact-supported image input |
| `POST` | `/v1/responses` | text/function-call Responses shape |

Bearer authentication is required when configured:

```text
Authorization: Bearer <EXPERT_API_KEY>
```

## Chat

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $EXPERT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-fp4",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 128,
    "stream": true,
    "reasoning_effort": "xhigh"
  }'
```

SSE ends with `[DONE]`. Chat deltas expose visible `content`, optional
`reasoning_content` and standard structured tool calls. Usage is returned for
non-streaming responses and when requested for streaming.

## Generation controls

The parser validates bounded temperature, top-p, top-k, min-p, supported
penalties, stop rules, `reasoning_effort` and `enable_thinking`. Artifact
sampling/template/EOS metadata supplies defaults; explicit supported request
fields override them per key.

Tools are passed into the official artifact template. One artifact-declared
parser converts native output into standard events. The client executes tools
and appends their results; the server does not execute arbitrary tools.

## Images

When an artifact declares callable vision operations, OpenAI image parts may
use supported data URLs or bounded public `image_url` values. The service
enforces body, pixel and patch-token budgets and blocks private/reserved remote
targets. Current Qwen supports image understanding. Muse/Mistral auxiliary
vision tensors do not constitute callable support.

This API does not generate images: the active LLMs generate tokens, not pixels.

## Limits and errors

After exact prompt tokenization:

```text
effective output <= min(requested output, maximum context - prompt tokens)
```

This clamp does not preallocate KV. Malformed/unsupported input uses 400,
authentication uses 401, unknown routes/models use 404 and bounded admission
exhaustion uses 503. Do not retry unchanged 400/401 requests or create
unbounded 503 retry loops.
