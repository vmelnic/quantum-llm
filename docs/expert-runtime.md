# Expert Runtime contract

Status: current VM, worker and placement contract, 2026-08-26.

## Program and provider negotiation

Every service artifact publishes `runtime-model.tsv`. The common native runner
parses the program, validates geometry/encodings and asks registered providers
for the declared kernel capabilities. Common code does not branch on Qwen,
Muse, Ornith, DeepSeek, layer count or upstream tensor names.

A new checkpoint using existing operations requires a strict source adapter
and a new artifact. New mathematics, encoding or geometry requires a provider
implementation plus an independent numerical oracle. It does not require a
new HTTP service, scheduled task or `model.sh` case.

## Worker protocol

The server starts one native worker and performs a handshake that advertises
model identity, context/output limits, provider capabilities, KV policy,
session retention and telemetry schema. Commands cover:

- request begin/resume and bounded prompt feed;
- token step/generation and streaming output;
- commit/checkpoint, rewind, retain/park, restore and drop;
- cancellation and shutdown;
- cached status/telemetry snapshots.

Request state transitions are transactional. Resume does not consume the
parked state until rebind and suffix feed succeed. A zero-length suffix is
valid. Cancellation rewinds to the last client-echoable prompt checkpoint;
worker/protocol errors fail the request rather than silently rebuilding a
different state.

## Dense placement

The dense/hybrid FP4 provider currently owns one hot CUDA execution slot:

- FP4 matrix weights are resident in VRAM;
- activation tiles stay device-resident across the operation program;
- recurrent state remains hot while a request executes;
- exact target F16 KV grows according to artifact-declared global or sliding
  geometry; Qwen uses request-owned 256-token pinned-host pages;
- bounded global-attention KV spans are staged to the GPU; artifact-declared
  sliding windows remain exact cyclic windows;
- inactive sessions retain populated pages and a compact continuation blob.

MTP/draft state is allocated only when the generation policy can actually use
exact verification. Stochastic sampling currently uses target-only decode;
paying draft prefill/KV in that mode is forbidden.

## Sparse placement

DeepSeek experts use immutable keys and a single state machine:

```text
absent -> SSD loading -> RAM ready -> GPU upload -> VRAM ready
```

Demand, prediction and warm work are separate priorities. Demand owns enough
staging capacity to make progress; speculative work is bounded and
cancellable. Protected/probationary host retention and transient/protected
VRAM keep one-shot routes from evicting established hot records. Leases pin
records until consuming CUDA work completes.

## Capacity and accounting

Worker slots, queue entries, RAM pages, VRAM pages, staging buffers and output
tokens are independent credits. Parking admission accounts the real stored
bytes, including fixed continuation state, rather than deriving capacity only
from KV page count. Status reports actual allocated/populated bytes and never
speculatively reserves a request's declared maximum.

Provider telemetry separates the copied continuation blob
(`provider_parked_request_bytes`) from total retained-session ownership,
including authoritative host KV (`provider_parked_session_bytes`). Both are
instantaneous gauges, not request-attributed traffic counters.

The current Python command channel/provider mutex serializes dense execution.
`MODEL_WORKER_CAPACITY>1` can retain/admit more state but does not create
parallel GPU kernels. Multi-device execution requires the allocator/shard/P2P
work in [Roadmap](roadmap.md).

## Correctness boundaries

- unknown operation, encoding, record version or missing provider fails start;
- missing routed experts fail the request; they are never treated as zero;
- exact target F16 KV cannot be substituted with FP8/Q4 by policy;
- sampling, reasoning, tool and image behavior use artifact template metadata;
- profiling is capability-driven and disabled unless explicitly requested;
- all model-specific source knowledge ends before the common program/runtime
  boundary.
