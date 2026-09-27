# OpenAI-compatible API

Status: implemented bounded compatibility surface, 2026-09-27.

The service implements selected OpenAI request and response shapes for the one
active model. Unlisted fields are unsupported.

## Endpoints

| Method | Path | Result |
|---|---|---|
| `GET` | `/v1/models` | active model list |
| `GET` | `/v1/models/{id}` | active model metadata |
| `POST` | `/v1/completions` | text completion |
| `POST` | `/v1/chat/completions` | chat, reasoning, tools and supported images |
| `POST` | `/v1/responses` | text and function-call Responses shape |

When configured, every endpoint requires:

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

Chat streaming uses SSE and terminates with `[DONE]`. Deltas may contain
`reasoning_content`, visible `content` and standard tool-call fields.
Non-streaming responses include usage; streaming includes usage when requested.

## Supported controls

- `temperature`, `top_p`, `top_k`, `min_p` and
  `presence_penalty` within service bounds;
- `frequency_penalty=0` and `repetition_penalty=1`;
- stop strings;
- `reasoning_effort` values `low`, `medium` and `xhigh`;
- `chat_template_kwargs.enable_thinking` and `preserve_thinking`;
- `speculative_decoding` as a boolean diagnostic override;
- `stream`, supported usage options and one output candidate;
- function tools with automatic or disabled tool choice.

Artifact sampling profiles supply defaults. A supported request value overrides
the corresponding default. Thinking is disabled with:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

`speculative_decoding=false` forces ordinary target steps for that request.
It does not change target weights, target sampling or the loaded artifact.

Required or named tool choice is rejected because constrained decoding is not
implemented. The server formats tool definitions and parses model output; the
client executes tools.

Log probabilities, logit bias, multiple candidates, echo, audio, prediction,
suffix completion, stored response chaining, automatic truncation and
structured response formats are not implemented.

## Images

Image parts are accepted only when the active artifact declares callable media
operations. Qwen currently exposes image understanding. Muse and Mistral keep
auxiliary vision tensors that are not callable.

Inputs may use bounded data URLs or public `image_url` values. The service
enforces body, pixel and patch-token limits and rejects private or reserved
remote targets. Output is text only.

## Limits and errors

After exact tokenization:

```text
effective output <= min(requested output, service output limit,
                        model context - prompt tokens)
```

Thinking requests may have a smaller artifact-declared ceiling. Context is
allocated progressively.

Malformed or unsupported requests return 400, authentication failures 401,
unknown routes or models 404, and bounded admission exhaustion 503.
