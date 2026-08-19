# Operations runbook

Status: current day-two lifecycle as of 2026-08-11. The implementation resume
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
./ops/model.sh start lfm2-8b-a1b-expert-pack-fp4
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
| `MODEL_MAX_CONTEXT` | `65536` | instructions + history + input + requested output |
| `MODEL_MAX_OUTPUT_TOKENS` | `8192` | server ceiling for one response |
| `CHAT_MAX_TOKENS` | `8192` | terminal client request ceiling |
| `MODEL_GENERATION_TIMEOUT_SECONDS` | `600` | request execution deadline |
| `MODEL_READY_TIMEOUT` | `600` | lifecycle wait deadline |

The current universal `.env` profile uses one worker slot and four queued
requests for every artifact. KV credits are aggregate.
Overload or insufficient context credits returns bounded HTTP errors rather
than allocating unbounded memory.

The 65K ceiling is enabled but unqualified. See
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

The default loopback bind plus SSH tunnel is the supported pilot configuration.
The built-in server is not an internet edge. For broader access, add a private
firewall allowlist and a TLS/authenticated/rate-limited reverse proxy. Rotate
keys outside the repository and restrict health/metrics/model-info separately.

## Performance reporting

Never publish one unqualified “tok/s” number. Record:

- cold or warm and how warming was obtained;
- single-stream or aggregate;
- prompt/history and output token counts;
- TTFT, post-first-token and end-to-end rates;
- cache/storage/transfer state when available.

An identical repeated Qwen prompt can reach 27–29 tok/s after the first token
(39.6–45.6 with the FP4 pack), while changing multi-turn chat remains
storage-bound. Those are dated performance-suite results, not the latest
common-path `hi` gate. The VM gate proved lifecycle compatibility for Qwen,
DeepSeek and LFM but did not demonstrate a speed increase. Both facts must
remain visible.
