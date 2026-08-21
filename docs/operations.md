# Operations runbook

Status: current day-two lifecycle as of 2026-08-19. The implementation resume
point is [the MoE VM handoff](moe-vm-next.md).

Use [Install, configure and use](deployment.md) for initial setup. This page is
the short day-two runbook.

## Lifecycle

```bash
./ops/model.sh install            # after clone or requirements changes
./ops/model.sh config             # resolved non-secret local configuration
./ops/model.sh start              # CHAT_MODEL from .env
./ops/model.sh start qwen         # Qwen FP4 pack (ABI 3)
./ops/model.sh start deepseek
./ops/model.sh status
./ops/model.sh chat               # resolves the actually deployed model
./ops/model.sh stop
./ops/model.sh stop all           # release every model process and VRAM
./ops/model.sh sync               # source/docs only, without restart
```

`start` synchronizes by default, performs a read-only dependency preflight
before disrupting the current service, stops the common task and any retired
task names, installs the selected artifact, waits for readiness and verifies
model/context/output/deadline.

Only one model owns the GPU and loopback API port. Task Scheduler is a pilot
supervisor. The repository stop command kills the complete Python/worker
descendant tree; stopping only the visible task can leave CUDA children alive.

## Health and identity

- `/health`: worker process is alive;
- `/ready`: healthy and admitting, not necessarily warm;
- `/model-info`: model/build/artifact identity, configured limits, placement,
  KV geometry, capacity and active request count;
- `/metrics`: bounded service counters and latency summaries.

After every deploy, require `model.sh start` to return without mismatches and
confirm `model.sh status`. A listening port alone is not proof of identity.

## Limits and capacity

`.env` is the deployment authority:

| Variable | Reference value | Meaning |
|---|---:|---|
| `MODEL_MAX_CONTEXT` | `262144` | instructions + history + input + requested output |
| `MODEL_MAX_OUTPUT_TOKENS` | `8192` | server ceiling for one response |
| `CHAT_MAX_TOKENS` | `8192` | terminal client request ceiling |
| `MODEL_GENERATION_TIMEOUT_SECONDS` | `14400` | request execution deadline |
| `MODEL_READY_TIMEOUT` | `600` | lifecycle wait deadline |
| `MODEL_VRAM_CACHE_GIB` | `12` | common routed-VRAM budget validated on the 24 GiB RTX 3090 |
| `MODEL_KV_CACHE_DTYPE` | `artifact` | provider-declared default; `fp16` requests exact target KV semantics from a capable provider |

The current universal `.env` profile uses one worker slot and four queued
requests for every artifact. KV credits are aggregate.
Overload or insufficient context credits returns bounded HTTP errors rather
than allocating unbounded memory.

The 262K capacity path has completed, but both the original FP4-KV service and
the later FP8-KV experiment failed production qualification. See
[Performance evidence](benchmarks.md) and
[Production readiness](production-readiness.md).

## Logs and diagnosis

The service JSONL is under the ignored remote `logs/` directory. It contains
identity/readiness, HTTP summaries, startup failures, request failures and
worker stderr without prompt content or API keys.

Diagnostic order:

1. `./ops/model.sh status`;
2. confirm task state and last result;
3. inspect the latest `service_start_failed`, `request_preprocessing_failed`,
   `request_failed` or CUDA worker event;
4. run the same launcher in foreground only when Task Scheduler hid a native
   exception;
5. fix the contract/dependency; do not extend timeouts blindly.

`model.sh install` reconciles pinned server requirements. `start` only checks
imports/versions and fails before stopping the currently running model.

## Rollback

1. Stop all model processes.
2. Deploy a previously validated commit to a clean directory.
3. Reconcile pinned requirements and build/test the native runtime.
4. Validate the immutable pack/bundle independently.
5. Start and require `/model-info` identity plus one real chat request.

Code rollback never mutates model artifacts. Keep the original checkpoint or
an independently validated immutable pack with hashes.

## Exposure and security

The default loopback bind plus SSH tunnel is the safest pilot configuration.
`MODEL_HOST` can explicitly enable a private-LAN bind for the
[Anthropic/Claude Code path](anthropic-api.md), but the service refuses that
bind without `EXPERT_API_KEY`. The built-in server is not an internet edge.
Restrict direct access with a private firewall allowlist; use a
TLS/authenticated/rate-limited reverse proxy for broader exposure. Rotate keys
outside the repository and restrict health/metrics/model-info separately.

## Performance reporting

Never publish one unqualified “tok/s” number. Record:

- cold or warm and how warming was obtained;
- single-stream or aggregate;
- prompt/history and output token counts;
- TTFT, post-first-token and end-to-end rates;
- cache/storage/transfer state when available.

The corrected real F16 Qwen3.8 `model.sh chat` gate measured about 30.40 tok/s
after the first generated token for `hi` and 29.26 tok/s for the four-stanza
prompt. These are short-context results. The terminal client must not compute
post-first throughput from total output tokens when its timestamp is the first
visible content after hidden reasoning. The historical direct `hi` gate
measured about 32.0 tok/s after the
first generated token. It did not predict Claude Code behavior: a cold Claude
`hi` carried 29,487 prompt tokens and took 82.875 seconds to first token. The
FP8-KV maximum-context gate populated all 262,144 positions but decoded at only
3.064 tok/s after a 1,168.594-second prefill. Keep direct, harness and saturated
context workloads separate. The old client `135.91 tok/s` value is invalid
because it combined hidden reasoning token counts with first-visible-content
timing.

The 2026-08-21 common-runner regression used the validated 12 GiB routed-VRAM
budget. Qwen3.8 F16 answered `hi`; server telemetry measured 53 prompt tokens,
30 generated tokens, 1.703 s TTFT, 2.672 s wall and 29.93 tok/s after the first
generated token. DeepSeek-V4-Flash also answered `hi`; it used 5 prompt tokens,
10 generated tokens, 3.579 s TTFT and 10.735 s wall. DeepSeek advertises
`session_retention=false` because its callable provider has no exact
checkpoint/rewind implementation. The 13 GiB cache profile failed the RTX 3090
preflight once fixed allocations and the 1 GiB reserve were included; do not
restore it as the common default without new capacity evidence.
