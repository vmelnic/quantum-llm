# Anthropic Messages API and Claude Code

Status: protocol-compatible gateway; Qwen3.8 backend rejected for production
Claude Code use on 2026-08-21.

The common HTTP service exposes an Anthropic-compatible adapter in parallel
with the existing OpenAI-compatible endpoints. Both protocols use the same
artifact-selected tokenizer, prompt template, admission control, retained KV
sessions and CUDA worker. The adapter does not add another model runner.

## Endpoints

| Method | Path | Contract |
|---|---|---|
| `POST` | `/v1/messages` | Anthropic Messages JSON or named SSE events |
| `POST` | `/v1/messages/count_tokens` | Token count after the real artifact chat template |

Supported message blocks are `text`, assistant `thinking`, `tool_use`, and
user `tool_result`. Client tools use Anthropic `input_schema` definitions and
are translated to the tokenizer's declared function-tool contract. Images,
documents, server-side Anthropic tools, structured output and constrained
`any`/named tool choice are rejected rather than silently substituted.

Prompt-cache `cache_control` annotations do not claim Anthropic cache
accounting. Claude Code's inline system/cache boundary is translated into a
worker KV checkpoint before the dynamic reminder, allowing later turns to
prefill only their conversation delta. Runtime retained-prefix reuse remains
the service's actual cache mechanism. Qwen reasoning stays hidden from the
Messages response because this local model cannot produce Anthropic-signed
thinking blocks.

Streaming Messages emit Anthropic `ping` events during long prefill and
buffered tool-call generation so clients do not mistake active local inference
for an idle or failed connection.

The OpenAI endpoints remain available and unchanged at `/v1/responses`,
`/v1/chat/completions`, and `/v1/completions`.

Protocol compatibility is not a model-quality claim. The real Qwen3.8 FP4
qualification sent about 29.5K prompt tokens for a cold Claude request, took
about 83 seconds to first token, fabricated basic facts in a simple poem task,
and entered multi-thousand-token reasoning/tool loops. The service is therefore
not an approved Claude Code backend despite accepting the API contract.

## Private-LAN deployment

The default bind remains loopback. To make the service reachable through the
Windows host's private IP, configure the control-host `.env`:

```dotenv
MODEL_HOST=0.0.0.0
EXPERT_API_KEY=<private-lan-token>
```

`model.sh start` passes both values into the one common scheduled task and its
readiness check. A non-loopback bind without a key fails closed. Limit the
Windows firewall rule to the private subnet or, preferably, the exact client
address. The built-in service is plain HTTP and must not be exposed directly
to the public internet.

## Claude Code

Pin every Claude Code model role to the single deployed artifact. Otherwise
background/Haiku calls may send an undeployed Anthropic model ID and correctly
receive `unknown model`:

```bash
export ANTHROPIC_BASE_URL=http://10.10.88.4:8080
export ANTHROPIC_AUTH_TOKEN=<private-lan-token>
export ANTHROPIC_MODEL=qwen3.8-27b-fp4
export ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-27b-fp4
export ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-27b-fp4
export ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-27b-fp4
export CLAUDE_CODE_SUBAGENT_MODEL=qwen3.8-27b-fp4

claude --model qwen3.8-27b-fp4
```

`ANTHROPIC_AUTH_TOKEN` is sent as a bearer token. The server also accepts the
Anthropic SDK's `x-api-key` header. See the upstream
[gateway configuration](https://docs.anthropic.com/en/docs/claude-code/llm-gateway)
and [model configuration](https://code.claude.com/docs/en/model-config) for the
client-side behavior of these variables.
