# Operations

Status: current operator runbook, 2026-09-02.

## Health and status

```bash
./ops/model.sh status
```

| Endpoint | Meaning |
|---|---|
| `GET /health` | Python/native process liveness |
| `GET /ready` | liveness plus admission/not-draining state |
| `GET /model-info` | active artifact, limits, provider, KV and capabilities |
| `GET /v1/models` | advertised active model |
| `GET /metrics` | cached Prometheus snapshot |

These endpoints must not wait for a long prefill. `ready=true` does not imply a
warm cache, unused queue or achieved throughput target.

The primary host log is `logs/expert-server.jsonl`. Startup phase events
separate artifact loading, provider construction, program preparation and
worker readiness. `request_telemetry` is the authoritative request record.

## Diagnose latency

Report before changing code:

1. exact prompt/prefill tokens and whether a prefix was reused;
2. TTFT, first visible text and total wall time;
3. generated, reasoning and visible/useful output tokens;
4. KV dtype, populated pages/bytes, mirror/staging and restore traffic;
5. routed VRAM/RAM hits, SSD misses, reread bytes, upload bytes and wait time;
6. CPU/GPU expert decisions, queue/admission state and cancellation outcome.

For DeepSeek, distinguish novel from settled routes. For any 262K claim, prove
actual populated positions. GPU phase profiling is off by default because CUDA
event attribution can synchronize the hot path.

## Sessions and cancellation

Retaining providers key sessions by exact prompt/media prefix. Matching resume
feeds only the suffix; an unchanged prefix is zero-delta. Failed suffix feed
rewinds to the committed prompt. Client cancellation never commits partial
assistant output.

Admission accounts real parked bytes. Several clients may advertise 262K, but
several populated 262K F16 histories cannot be inferred safe from that ceiling.
Execution remains serialized through one hot provider slot.

DeepSeek does not implement exact checkpoint/rewind/session retention. A long
DeepSeek cancellation can leave the worker unhealthy. Check `/ready`; if it is
not healthy, stop and restart before another request. Do not claim automatic
recovery.

## Failure handling

- `401`: client and service keys differ or are absent;
- `400`: unsupported model/role/media/sampling/tool field; do not retry unchanged;
- `503`: slot, queue, KV or cache admission exhausted; use bounded retry;
- readiness timeout: inspect task state and server log, then stop before retry;
- worker/protocol failure: discard in-process state and replay from a
  client-owned prefix after restart;
- artifact failure: stop, validate the known rollback, promote it, start and
  run the real smoke.

Do not locally patch the same failed mechanism more than once. After a second
failure, stop editing and reassess the complete path.

## Cleanup

```bash
./ops/model.sh stop all
./ops/model.sh status
```

Verify no Quantum LLM worker remains and model VRAM returned to the Windows
desktop baseline. `work/`, `out/`, `logs/` and local artifacts are generated,
but remove them only after durable results are recorded. Published artifacts
below `MODEL_ROOT` and Hugging Face caches are not normal scratch cleanup.
