# Operations

Status: current operator runbook, 2026-08-26.

## Observe without blocking inference

```bash
./ops/model.sh status
```

The service exposes authenticated endpoints:

- `GET /health`: process and worker liveness;
- `GET /ready`: liveness plus admission/not-draining state;
- `GET /model-info`: artifact, limits, provider, session and KV geometry;
- `GET /v1/models`: advertised model identity;
- `GET /metrics`: cached Prometheus text snapshot.

These endpoints use bounded cached worker state and must not wait for a long
prefill. `ready=true` does not mean that caches are warm or a throughput target
has been met.

## Diagnose a slow request

Separate the phases and rates before changing code:

1. prompt tokens, prefill time and time to first visible token;
2. generated target tokens and post-first-token rate;
3. reasoning tokens versus visible answer tokens;
4. Qwen KV pages/bytes populated, staged and restored;
5. DeepSeek VRAM/RAM/storage hits, reread bytes and storage/H2D wait;
6. cancellation, queue and capacity outcomes.

For DeepSeek, compare novel and settled routes separately. For Qwen, report
actual populated context and KV dtype. Never infer a saturated-context result
from a configured maximum.

GPU phase profiling is diagnostic and off by default; enabling it changes the
hot path through CUDA event collection. Keep it out of production results
unless the result explicitly measures instrumentation overhead.

## Sessions and cancellation

Retained sessions are keyed by the exact prompt/media prefix. A valid resume
feeds only the suffix; an unchanged prefix performs a zero-delta resume. A
failed suffix prefill rolls back to the committed checkpoint. Client
cancellation preserves only the prompt that the client can echo on its next
turn and discards partial assistant output.

Admission is based on real parked bytes as well as slots. A session whose
exact F16 KV grows near 262K consumes about 16 GiB before continuation state;
several such sessions cannot be admitted merely because each declares the
same maximum.

## Failure handling

- HTTP 401: use the same `EXPERT_API_KEY` in the service and client.
- HTTP 503 overloaded: all request slots, KV capacity or queue capacity are
  occupied; do not retry in an unbounded loop.
- readiness timeout: inspect scheduled-task state and server logs, then run
  `model.sh stop all` before restarting.
- worker failure: do not reuse its in-process KV/session state; restart the
  common service and replay from a client-owned prefix.
- artifact failure: stop, restore the timestamped rollback directory, validate
  it, then start and smoke it through the public API.

## Cleanup

`work/`, `artifacts/`, `logs/`, `out/` and Python caches are generated and
ignored. They may be removed only after any durable measurement has been
summarized in [Benchmarks](benchmarks.md) or
[Research decisions](research-decisions.md). Published model artifacts below
`MODEL_ROOT` are not scratch space and are never deleted by normal lifecycle
commands.

After any gate:

```bash
./ops/model.sh stop all
./ops/model.sh status
```

Then inspect `nvidia-smi` on the execution host if GPU cleanup is material to
the gate.
