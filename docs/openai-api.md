# OpenAI-compatible HTTP API

This document is the public contract of the local Expert Runtime service. The
implementation targets wire compatibility with OpenAI clients for deterministic
text generation. It does not silently emulate capabilities that the model
runner does not have: unsupported sampling, multimodal, tool, logprob and
server-side state options return a structured `400` error.

The canonical P6 deployment listens on `http://127.0.0.1:8080/v1`. It serves one
model, currently `qwen3-next-80b-a3b-expert-pack-int8`.

## Endpoints

| Method | Path | Contract |
|---|---|---|
| `POST` | `/v1/responses` | Responses API text generation, JSON or typed SSE |
| `POST` | `/v1/chat/completions` | Chat Completions, JSON or chunked SSE |
| `POST` | `/v1/completions` | Legacy Completions, JSON or chunked SSE |
| `GET` | `/v1/models` | OpenAI model list object |
| `GET` | `/v1/models/{model}` | Retrieve the deployed model |
| `GET` | `/health` | Process/worker liveness |
| `GET` | `/ready` | Admission readiness |
| `GET` | `/model-info` | Exact build, container hashes and runtime limits |
| `GET` | `/metrics` | Prometheus text metrics |

Every JSON/SSE response includes `x-request-id`. Errors use the OpenAI shape:

```json
{
  "error": {
    "message": "sampling is not implemented; omit temperature or use 0",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "unsupported_value"
  }
}
```

## Authentication and exposure

Loopback deployment does not require a key. A non-loopback bind is refused by
the launcher unless `EXPERT_API_KEY` is set; clients then send:

```http
Authorization: Bearer <EXPERT_API_KEY>
```

The service has no TLS terminator. Do not expose it directly outside a trusted
host/network; put authentication and TLS at a reverse proxy if remote access is
required.

## OpenAI SDK examples

The normal OpenAI `base_url` override works; any non-empty placeholder key is
enough for the loopback deployment.

For an interactive terminal chat, including an owned SSH tunnel and streaming
output, run from a POSIX control host:

```bash
cp .env.example .env
# Set CHAT_SSH in .env, then:
./ops/chat.sh
```

The `.env` file controls the SSH target, model, service context/output limits,
per-turn chat token limit, ports, readiness timeout, Python executable, base
URL, and optional API key; every variable is listed in `.env.example`. CLI flags remain optional overrides. The client
retains conversation history. Enter `quit` or `exit`, or press Ctrl+C, to close
both the client and the tunnel it created. With an empty `CHAT_SSH`, it connects
directly to `CHAT_BASE_URL`.

With `CHAT_SHOW_STATS=1`, every turn prints prompt/output token counts, TTFT,
total request time, end-to-end tokens/s, post-first-token rate, and finish
reason. `/info` displays the deployed build, placement profile, MTP state,
context/output limits, capacity, current RAM/VRAM expert-cache use, KV pages,
storage format, prefetch state, and worker protocol. `/stats` repeats the last
turn, while `/clear` starts a new conversation without restarting the model.

DeepSeek-V4-Flash does not publish a Transformers `chat_template`. For that
model the service loads the pinned checkpoint's official
`encoding/encoding_dsv4.py` and uses its `encode_messages(...,
thinking_mode="chat")` contract for Chat Completions and Responses. Plain
Completions continues to tokenize the supplied prompt without a chat wrapper.

```python
from openai import OpenAI

client = OpenAI(api_key="local", base_url="http://127.0.0.1:8080/v1")

response = client.responses.create(
    model="qwen3-next-80b-a3b-expert-pack-int8",
    instructions="Answer briefly.",
    input="Why is the sky blue?",
    max_output_tokens=64,
    temperature=0,
)
print(response.output_text)
```

```python
stream = client.chat.completions.create(
    model="qwen3-next-80b-a3b-expert-pack-int8",
    messages=[{"role": "user", "content": "Write one sentence."}],
    max_completion_tokens=64,
    temperature=0,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

The Responses API accepts a string or an array of text messages. Chat accepts
`system`, `developer`, `user` and `assistant`; `developer` is mapped to the
model's system role. Modern text content parts (`text`, `input_text`, and
`output_text`) are normalized before applying the local tokenizer's chat
template.

## Supported request fields

| Field | Support |
|---|---|
| `model` | Must equal the model returned by `/v1/models` |
| `input`, `messages`, `prompt` | Text and token IDs as appropriate for the endpoint |
| `instructions` | Responses API; prepended as a system message |
| `max_output_tokens` | Responses API, bounded by the service limit |
| `max_completion_tokens`, `max_tokens` | Chat/Completions, bounded by the service limit |
| `stream` | JSON response or SSE |
| `stream_options.include_usage` | Final usage chunk for Chat/Completions |
| `stop` | One string or up to four strings; matches may cross token boundaries |
| `temperature` | Omitted or exactly `0` (greedy execution) |
| `top_p` | Omitted or exactly `1` |
| `n`, `best_of` | Omitted or exactly `1` |
| `seed`, `user`, `metadata`, `store`, `service_tier` | Accepted; no remote persistence or tiering is performed |
| `response_format` / `text.format` | Plain `text` only |
| `tools`, `functions` | Empty arrays only |
| `tool_choice` | `none` or `auto` only when no tools are supplied |
| `presence_penalty`, `frequency_penalty` | Omitted or exactly `0` |
| `logprobs`, `top_logprobs` | Disabled (`false`/`0`) only |
| `modalities` | Omitted or `['text']` |
| `truncation` | Omitted or `disabled` |

Unknown harmless metadata fields are tolerated. Known capability fields are
validated so a client never believes that sampling or structured output was
honored when it was not.

## Explicitly unavailable

The current worker returns the selected greedy token, not a logits vector.
Consequently the API rejects nonzero temperature, nucleus sampling, penalties,
logit bias and logprobs. The text-only runner also rejects images, audio, files,
tool/function calling and JSON schema output.

The server keeps KV state only for the lifetime of one request. Responses API
storage, `previous_response_id`, conversations, background mode, WebSocket mode
and retrieve/cancel-by-response-ID are not implemented. A client must resend
conversation history.

These are inference/runtime capability gaps, not merely missing JSON fields.
Adding them correctly requires extending the worker protocol and GPU runner.

## Streaming contracts

Chat/Completions streams use `data: <json>` SSE frames, a final choice carrying
`finish_reason`, an optional usage-only chunk, then `data: [DONE]`.

Responses streams emit typed lifecycle events:

1. `response.created`
2. `response.output_item.added`
3. `response.content_part.added`
4. zero or more `response.output_text.delta`
5. `response.output_text.done`
6. `response.content_part.done`
7. `response.output_item.done`
8. `response.completed`

This follows the official distinction: Chat streaming uses incremental choice
deltas, while Responses streaming uses typed semantic events. A disconnected
client cancels the active worker request before another decode step is queued.

## Limits and operational behavior

Effective limits are returned by `/model-info.runtime_config`. Deployments made
through `ops/model.sh` take them from `MODEL_MAX_CONTEXT` and
`MODEL_MAX_OUTPUT_TOKENS`; the reference `.env.example` enables a 65,536-token
context and at most 8,192 generated tokens. Prompt plus requested output must
fit the context. The request body limit is 1 MiB.

Those configured ceilings are distinct from qualification and checkpoint
metadata. Qwen advertises 262K positions and DeepSeek-V4-Flash advertises 1M;
the only completed long-context correctness gate remains 4,096. The 65,536
DeepSeek setting fits its capacity-one 2 GiB logical KV budget, but long-prompt
quality and performance remain unqualified. Model metadata is therefore not
exposed automatically as an operational API promise. If
concurrent requests exhaust KV page credits, admission returns HTTP `503` with code
`context_capacity_exhausted` before streaming starts.

`/model-info.worker_kv` reports the backend dtype and allocation strategy, page
geometry, total logical page capacity, currently reserved pages, and physical
allocation expressed in the same page geometry. Qwen uses on-demand FP16 pages;
DeepSeek currently uses preallocated BF16 request state.
`/model-info.worker_prefill` reports the backend's causal prefill mode and maximum
tokens per chunk.
`/model-info.worker_execution` distinguishes an authenticated MTP resource from
an active speculative path. `mtp_resource_available` means descriptors exist;
`mtp_runtime_ready` means the worker also loaded the tensor state, compact pack,
shared expert, cache and directory. Either may be true while `mtp_enabled`
remains false unless the operator explicitly enables it; clients must never
infer speculative generation from bundle contents or startup residency alone.
When enabled, the internal protocol may return a verified bonus token, but the
HTTP streaming contracts remain one ordered text delta at a time.

`/model-info.worker_placement` is the authoritative effective placement
contract. It returns the selected `profile`, exact RAM/VRAM cache bytes,
whether speculative prefetch is enabled, and the required recent-observation
count. `/model-info.runtime_config.placement_profile` reports the requested
launcher value; startup fails if the worker reports a different profile or
budget.

Admission is bounded. Overload/drain returns HTTP `503` with code `overloaded`;
it does not create an unbounded queue. Generation timeout returns `504` before
streaming starts, or an `error` event after a Responses stream has started.

`finish_reason` is `stop` for EOS or a requested stop sequence and `length`
when the output-token limit is reached. Usage counts are tokenizer token counts;
the generated token that discovers a stop sequence is included even though the
matched stop text is not returned.

## Upstream references

- [OpenAI Responses API reference](https://developers.openai.com/api/reference/resources/responses)
- [OpenAI streaming Responses guide](https://developers.openai.com/api/docs/guides/streaming-responses)
- [OpenAI migration guide: streaming consumers](https://developers.openai.com/api/docs/guides/migrate-to-responses#7-update-streaming-consumers)
