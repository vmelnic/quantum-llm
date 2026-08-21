# OpenAI-compatible HTTP API

Status: current Qwen3.8 pilot contract as of 2026-08-19.

The service exposes one artifact at a time through the common VM runner on
`http://127.0.0.1:8080/v1`. Request formatting, response parsing, sampling and
tool support are enabled from artifact/tokenizer capabilities; common API code
does not select behavior from a Qwen or DeepSeek model name.

These routes remain available alongside the independent
[Anthropic Messages adapter](anthropic-api.md); enabling that adapter does not
change the OpenAI request or response contract.

## Endpoints

| Method | Path | Contract |
|---|---|---|
| `POST` | `/v1/responses` | Responses API text and function calls, JSON or typed SSE |
| `POST` | `/v1/chat/completions` | Chat Completions, JSON or chunked SSE |
| `POST` | `/v1/completions` | Legacy text Completions, JSON or chunked SSE |
| `GET` | `/v1/models` | Deployed model list |
| `GET` | `/v1/models/{model}` | Deployed model identity |
| `GET` | `/health` | Process and worker liveness |
| `GET` | `/ready` | Admission readiness |
| `GET` | `/model-info` | Artifact, runtime limits and provider capabilities |
| `GET` | `/metrics` | Prometheus metrics |

Every JSON/SSE response includes `x-request-id`. Unsupported values fail with
a structured OpenAI-shaped `400`; the service never silently substitutes a
different execution mode.

## Authentication

Loopback deployment needs no key. A non-loopback bind is rejected unless
`EXPERT_API_KEY` is configured, after which clients send:

```http
Authorization: Bearer <EXPERT_API_KEY>
```

The built-in service is not a public TLS edge. Use a firewall and an
authenticated, rate-limited TLS reverse proxy for non-loopback exposure.

## Qwen3.8 prompt and response contract

The published Qwen3.8 artifact carries the official tokenizer assets. The
server uses the tokenizer's declared chat template and standard Transformers
response parser. It does not invent a second tool or reasoning language.

For Qwen3.8:

- reasoning is enabled by default with `reasoning_effort="xhigh"`;
- hidden reasoning is returned as `reasoning_content` by Chat Completions and
  counted in `usage.completion_tokens_details.reasoning_tokens`;
- `chat_template_kwargs.enable_thinking` and `preserve_thinking` are accepted;
- function definitions are passed to the official template;
- the declared XML response grammar is parsed into standard OpenAI tool calls;
- assistant tool calls and following `tool` results can be sent back as normal
  conversation history.

An artifact without a compatible response grammar rejects non-empty `tools`.
DeepSeek continues to use its pinned official encoder because its checkpoint
does not publish a Transformers chat template.

## Sampling

Qwen3.8 defaults come from its immutable `generation_config.json`:

```text
do_sample=true, temperature=1.0, top_p=0.95, top_k=20
```

The SM86 provider implements token selection from logits with temperature,
top-k, top-p and min-p. `temperature=0` selects greedy decoding. The request
may override:

| Field | Accepted values |
|---|---|
| `temperature` | `0.0` through `2.0` |
| `top_p` | greater than `0.0` through `1.0` |
| `top_k` | non-negative integer; `0` disables the filter |
| `min_p` | `0.0` through `1.0` |
| `seed` | integer |

Sampling is a provider capability. Startup and `/model-info` report the actual
provider contract; a provider that cannot sample fails the request rather than
pretending to honor it. Sampled generation does not accept greedy MTP drafts as
exact matches; MTP state is kept synchronized without changing the requested
distribution.

Presence/frequency penalties, logit bias and logprobs remain unsupported.

## Tool calling

Chat Completions and Responses accept OpenAI function tools. `tool_choice` may
be `auto` or `none`. `required` and named-function forcing are rejected because
the runtime does not yet implement constrained decoding.

```python
from openai import OpenAI

client = OpenAI(api_key="local", base_url="http://127.0.0.1:8080/v1")
model = client.models.list().data[0].id

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is the weather in Chisinau?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls)
```

The real Qwen3.8 gate produced `get_weather` with
`{"city":"Chisinau"}` and, after receiving the tool result, returned the final
natural-language answer. Tools are model-generated calls; their JSON schemas
are prompt guidance, not a constrained grammar guarantee.

## Supported request surface

| Field | Support |
|---|---|
| `model` | Must equal the deployed model |
| `input`, `messages`, `prompt`, `instructions` | Text/token input appropriate to the endpoint |
| `max_output_tokens`, `max_completion_tokens`, `max_tokens` | Bounded by the service limit |
| `stream`, `stream_options.include_usage` | JSON or SSE |
| `stop` | One string or up to four strings |
| sampling fields | As described above |
| `reasoning_effort` / `reasoning.effort` | `xhigh`, `medium`, or `low`; default `xhigh` |
| `tools`, legacy `functions` | Function tools when the artifact declares a response parser |
| `tool_choice` | `auto` or `none` |
| `n`, `best_of` | Omitted or `1` |
| `response_format` / `text.format` | Plain text only |
| `modalities` | Omitted or `['text']` |
| `metadata`, `user`, `seed` | Accepted |

Images, audio, JSON-schema output, logprobs, server-side conversations,
`previous_response_id`, background mode, retrieve/cancel-by-ID and WebSocket
mode are not implemented.

## Sessions and streaming

The server retains bounded worker state for an append-only conversation and
prefills only the new token suffix. The client must still resend history; this
is an internal optimization, not remote conversation storage.

Chat streaming emits ordered choice deltas, including `reasoning_content`,
visible `content`, and complete tool-call deltas. Responses streaming emits
typed lifecycle, text and function-call events. Disconnecting a client cancels
the active worker request before another decode step is admitted.

## Limits and performance metrics

The reference Qwen3.8 profile is:

```text
MODEL_MAX_CONTEXT=262144
MODEL_MAX_OUTPUT_TOKENS=8192
MODEL_MAX_BODY_MIB=16
MODEL_KV_CACHE_MIB=5120
```

Prompt plus requested output must fit the context. KV pages are allocated on
demand; advertising 262,144 does not allocate that context for a short request.

An actual 262,016-token prompt plus 128 generated tokens completed, proving the
full 262,144-token capacity path. It does not prove semantic quality for every
long document or the 30 tok/s objective. See
[Production readiness](production-readiness.md).

For reasoning models, do not derive decode throughput from the first visible
content delta: hidden reasoning may precede it. The authoritative rate uses
server `request_telemetry.ttft_seconds`, which starts at the first generated
token, and `generated_tokens`, which includes hidden reasoning consistently.
