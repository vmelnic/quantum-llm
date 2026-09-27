# Operations

Status: current operator runbook, 2026-09-27.

## Health

```bash
./ops/model.sh status
```

| Endpoint | Meaning |
|---|---|
| `GET /health` | Python process and native worker liveness |
| `GET /ready` | liveness and admission while the service is not draining |
| `GET /model-info` | active artifact, limits, formats, provider and capabilities |
| `GET /v1/models` | advertised active model |
| `GET /metrics` | cached Prometheus snapshot |

These endpoints use bounded snapshots and must remain responsive during long
prefill. Readiness does not imply a warm cache, an empty queue or an achieved
latency target.

The primary log is `logs/expert-server.jsonl`. Startup events separate
artifact load, provider creation, program preparation, provider finalization
and worker readiness. `request_telemetry` is the request-level evidence
record.

## Diagnose a request

Record these facts before changing code:

1. model, artifact hash, KV policy, MTP state and sampling;
2. actual prompt tokens and provider prefill tokens;
3. cold, live-resumed or disk-restored prefix state;
4. TTFT, first visible text, decode wall and total wall;
5. generated, reasoning and visible tokens;
6. populated KV pages, mirror/staging bytes and park/restore traffic;
7. routed VRAM/RAM hits, storage misses, bytes read and H2D bytes;
8. storage/upload wait, expert execution decisions and cancellation outcome.

A configured context ceiling is not evidence of populated context. Process I/O
counters are not SSD-only traffic. GPU phase profiling is disabled in normal
service because event collection can synchronize the hot path.

## Sessions and cancellation

Retaining providers key state by exact prompt and media identity. A matching
request resumes from the retained prefix and feeds only the suffix. A failed
suffix feed preserves or abandons the old state explicitly. Cancellation can
rewind to the declared prompt checkpoint; partial assistant output is not
silently committed.

Parked sessions own real host bytes. Persistent Qwen snapshots additionally
own bytes below `${MODEL_ROOT}/.session-cache/<artifact>`. Clear only the
selected artifact cache with:

```bash
./ops/model.sh clear-cache qwen
```

Several clients may queue or hold parked sessions. The current hot provider
slot still executes one batch at a time.

## Failure handling

| Symptom | Action |
|---|---|
| HTTP 400 | fix the unsupported request field or malformed input; do not retry unchanged |
| HTTP 401 | fix the client/service credential mismatch |
| HTTP 404 | verify route and active advertised model |
| HTTP 503 | inspect queue, slot, KV and cache admission; retry only with a bound |
| readiness timeout | inspect task and startup events, then stop before retry |
| worker/protocol failure | stop, discard in-process state and replay a client-owned prefix |
| artifact validation failure | stop, validate the known rollback, promote it and run the real smoke |
| repeated inference failure after one local correction | stop editing and reassess the end-to-end path |

After a header, provider ABI, wire ABI or launch-structure change, reproduce a
native failure once on the canonical clean build before editing model logic.

## Cleanup

```bash
./ops/model.sh stop all
./ops/model.sh status
```

Verify that the scheduled task and native worker are gone, the service port is
closed and GPU memory returned to the desktop baseline. Generated results may
be removed only after durable evidence is recorded. Published artifacts and
Hugging Face caches are not routine scratch cleanup.
