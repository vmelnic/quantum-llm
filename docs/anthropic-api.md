# Anthropic Messages API

Status: implemented wire adapter; Pi remains the maintained harness,
2026-09-12.

## Endpoints

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

The adapter accepts Anthropic user/assistant messages, top-level system text,
supported text/image blocks, tools/tool results, streaming, sampling and output
limits. It normalizes them into the same artifact template and native request
used by the OpenAI surface.

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

Streaming emits Anthropic message/content-block events and separates visible
text, thinking and tool use. `count_tokens` uses the same normalization and
template path.

## Boundary

Wire compatibility does not turn Qwen or DeepSeek into Claude. Model quality,
tool reliability and latency remain those of the selected artifact. Claude
Code injects a large Claude-oriented system/tool prompt; historical project
gates sent roughly 29K input tokens before a one-word request and had
unacceptable first-turn latency and factual behavior. Claude Code is therefore
not the recommended production harness.

The repository `.claude/settings.json` and `ops/claude-api-key.sh` remain a
bounded compatibility configuration. The helper reads `EXPERT_API_KEY` from
ignored `.env`; no secret is committed. Pi is the maintained local agent path.

The adapter obeys the same context, body, output, image, queue and timeout
limits as the OpenAI API. Invalid role/block/thinking/tool combinations fail
explicitly with Anthropic-shaped errors.
