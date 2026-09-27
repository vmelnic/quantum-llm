# Anthropic Messages API

Status: implemented wire adapter, 2026-09-27.

## Endpoints

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

The adapter accepts Anthropic user and assistant messages, top-level system
text, supported text and image blocks, tools and tool results, streaming,
sampling and output limits. It normalizes them into the same artifact template,
admission path and native request used by the OpenAI-compatible surface.

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

Streaming emits Anthropic message and content-block events for thinking, text
and tool use. `count_tokens` uses the same message normalization, media path
and chat template as generation.

The adapter shares the service's model, context, body, image, queue, timeout
and authentication limits. Invalid role, content block, thinking or tool
combinations return Anthropic-shaped errors.

This is protocol compatibility with the active artifact. Pi is the maintained
coding harness. The repository's Claude configuration remains a compatibility
path and does not define a separate runtime or model behavior.
