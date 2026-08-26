# Anthropic Messages API

Status: implemented wire adapter with explicit harness limits, 2026-08-26.

## Endpoints

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

The adapter accepts Anthropic `user` and `assistant` message roles, a top-level
system prompt, text/image content, tool definitions/results, streaming,
sampling and output limits. It translates them into the same official
artifact template and native request used by the OpenAI surface.

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H "x-api-key: $EXPERT_API_KEY" \
  -H 'anthropic-version: 2023-06-01' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-fp4",
    "max_tokens": 128,
    "thinking": {"type": "enabled", "budget_tokens": 64},
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

Streaming emits Anthropic message/content block events and reports visible
text, thinking and tool-use blocks separately. `/count_tokens` applies the
same normalization/template path and returns the resulting input token count.

## What compatibility does not mean

This endpoint does not turn Qwen or DeepSeek into Claude. Model behavior,
prompt interpretation, tool reliability, reasoning quality and latency remain
those of the selected local artifact. Claude Code also injects a large system
prompt, repository context and tool schema; a one-word user prompt therefore
does not represent a one-token prefill.

The adapter has passed protocol/unit gates, but Claude Code on the reference
host produced unacceptable first-turn latency and factual behavior. It is not
the recommended production harness. Pi is the maintained local coding-agent
integration; see [Pi CLI](pi-cli.md).

## Authentication and limits

The configured secret is accepted as `x-api-key` or bearer authentication.
Messages obey the same request-body, context, output, image, queue, timeout and
KV admission limits as the OpenAI API. Unsupported roles/blocks and invalid
thinking/tool combinations fail with Anthropic-shaped errors rather than being
silently discarded.
