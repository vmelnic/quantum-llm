# Roadmap

This is the current backlog. Completed experiments and rejected approaches are
kept in [Engineering history](history.md); measured results are in
[Performance evidence](benchmarks.md).

## P0 — truthful chat behavior

1. Expose Qwen request-level expert telemetry through `/model-info`, logs and
   metrics: SSD bytes, RAM/VRAM hits, H2D bytes, CPU/GPU selections and time.
2. Distinguish lifecycle states `ready`, `warming` and workload-qualified `hot`.
   Readiness alone must never imply the throughput SLO.
3. Add a repeatable chat workload with changing prompts and growing history.
   Report cold/warm, TTFT, post-first-token and end-to-end rates separately.
4. Add retained conversation state or prefix/KV reuse so every turn does not
   re-prefill the entire unchanged history.
5. Build workload-aware expert placement/prefetch for route diversity. A single
   repeated warm-up prompt is not an acceptance test.

Acceptance: at least 30 useful output tok/s for a documented single-chat
workload, with bounded memory and no omitted experts. Aggregate throughput is a
separate SLO.

## P1 — DeepSeek working-set performance

1. Attribute every token wall-time segment to dense/attention, routing,
   acquisition, storage, H2D, expert compute, aggregation and output head.
2. Keep routed FP4 records compact through storage and residency; create only
   bounded compute-ready tiles/slots.
3. Replace per-expert launch/compute paths with grouped or fused SM86 kernels
   where measurements show a win.
4. Overlap cold acquisition with ready work without blocking or evicting live
   request leases.
5. Use CPU expert compute only when measured completion beats local GPU load
   plus compute; the tested forced desktop-CPU split was slower.
6. Evaluate MTP by accepted useful output tokens/s, including verification and
   rejected drafts, never by draft rows alone.

Acceptance: representative DeepSeek multi-turn output improves monotonically
with exact token equality; no synthetic microbenchmark may replace API evidence.

## P2 — long-context qualification

The API ceiling is currently configurable to 65,536, but only 4,096 has passed
the historical qualification gates.

1. Decouple causal prefill chunk size from request concurrency.
2. Add adaptive/grouped prefill and an efficient full-attention kernel.
3. Gate 8K, 16K, 32K and 64K sequentially for numerical equality, memory high
   water, cancellation, TTFT and decode.
4. Test mixed context lengths against the aggregate KV credit budget.
5. Advertise a qualified context separately from the configured hard ceiling.

## P3 — service hardening

1. Native Windows service or equivalent non-interactive supervisor.
2. Log rotation, external metrics, request correlation, tracing and alerts.
3. TLS/auth/rate-limiting reference edge; keep the built-in server loopback.
4. 24-hour soak plus cancellation, overload, short-read, corrupt-pack, disk-full,
   CUDA OOM/reset and worker-restart campaigns.
5. Self-hosted SM86 CI, pinned toolchain/driver and signed release artifacts.

## P4 — platform-neutral core and Metal

Keep manifests, request state, placement, scheduler and cache policy independent
of CUDA/Windows. Add macOS asynchronous storage and a Metal backend after the
single-host contracts are stable. Unified memory removes some copies but does
not turn another Mac's RAM into local memory, and fanless/thermal constraints
remain explicit.

RTX 3090/CUDA remains the first performance backend. Metal is an adaptation,
not a reason to weaken the current acceptance gates.

## P5 — distributed expert workers

Do not stream whole expert weights over ordinary network links per token. Move
activation compute to the node that already owns the expert:

```text
coordinator: dense + attention + exact router
       │ activations, expert IDs, routing weights
       ▼
expert worker: local RAM/SSD/accelerator → gate/up/down
       │ strict outputs or weighted partials
       ▼
coordinator: stable aggregation
```

The future application protocol includes:

- persistent TCP first, including IP over Thunderbolt Bridge;
- separate control and data planes;
- binary framing with bounded payloads and checksums;
- model manifest and quant/kernel ABI negotiation;
- credits for queue, staging, compute rows and cache bytes;
- operation IDs, connection epochs, cancellation and idempotent retry;
- exact failure when any selected expert has no local/remote replica.

Use standard transport reliability and cryptography; do not invent TCP or TLS.
Remote placement is admitted only when measured remote completion beats local
load plus compute.

## P6 — hundreds of billions to one trillion parameters

- shard packs/indexes across storage and workers;
- bound metadata independently of total parameter count;
- window the device directory and compute-ready slots;
- replicate hot experts based on observed demand;
- optimize single-stream and aggregate workloads separately;
- preserve exact routing and explicit failures under partial node loss.

The durable rule is: weights remain near their storage/compute tier;
activations move instead of whole experts per token.
