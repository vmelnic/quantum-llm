# OpenAI-compatible API

Status: implemented compatibility surface, 2026-08-26.

The server implements a bounded subset of common OpenAI request/response
shapes. It is a compatibility adapter, not the OpenAI service and not a claim
of byte-for-byte behavior for unlisted fields.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/models` | list the one active model |
| GET | `/v1/models/{id}` | retrieve the active model |
| POST | `/v1/completions` | legacy text completion |
| POST | `/v1/chat/completions` | chat, reasoning, tools and image input |
| POST | `/v1/responses` | text/function-call Responses shape |

Operational endpoints are documented in [Operations](operations.md).

Bearer authentication is required when configured:

```text
Authorization: Bearer <EXPERT_API_KEY>
```

## Chat example

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

SSE streams end with `[DONE]`. Chat chunks expose visible text as `content`
and model reasoning as `reasoning_content`. Usage is returned for non-streaming
requests and for streaming requests that request usage.

## Sampling and thinking

The request parser supports bounded temperature/top-p/top-k generation and
the artifact's stop rules. `reasoning_effort` accepts the model/template
levels supported by the artifact. `enable_thinking` can explicitly toggle
reasoning; Pi maps its `--thinking` setting to this request control. The
repository default is `xhigh`.

Stochastic sampling disables exact MTP verification until a sampling-aware
verifier preserves the target distribution. This is a generation-policy rule,
not an excuse to allocate unused draft state during prefill.

## Tools

Chat Completions and Responses accept function tool definitions and tool
choice. The official artifact chat template receives those definitions. The
server parses the model's declared assistant codec into structured tool calls.
The client/harness remains responsible for executing tools and appending tool
results to the next request.

## Image input

When the artifact declares vision operations, chat content may contain OpenAI
image parts using a bounded public `image_url` or a supported data URL. The
server decodes through the official image processor, enforces pixel/patch/body
budgets, blocks private/reserved URL targets and sends a typed multimodal packet
to the native provider.

This is image understanding, not image generation. The Qwen text/vision model
produces tokens; it cannot produce diffusion pixels without a separate image
model, which is outside this project's serving goal.

## Limits and errors

Model ID, roles, body size, service output ceiling, media, sampling values and
tools are validated before admission. The service may advertise an absolute
output ceiling of `max_context - 1`; after exact prompt tokenization, the
effective generation ceiling is:

```text
min(requested_output_tokens, max_context - exact_prompt_tokens)
```

This clamp does not preallocate KV. KV pages grow only with populated prompt
and generated positions. Authentication uses 401; malformed/unsupported
requests use 400; queue/slot/KV exhaustion uses 503; unknown model/routes use
404. Clients should not retry 400/401 responses or create unbounded 503 retry
loops.
