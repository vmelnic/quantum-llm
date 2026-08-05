# Expert Runtime v1 contract

This document defines invariants between the portable core, Windows storage,
heterogeneous cache/scheduler, CUDA backend, model runner, and HTTP service.
The current backend target is Windows + RTX 3090 (SM86).

## Module boundary

```text
manifest/index → feasibility/admission → request state
                                         │
exact router → device dispatch plan → expert-centric scheduler
                                         │
           ┌─────────────────────────────┼────────────────────────────┐
           ▼                             ▼                            ▼
     VRAM resident                 RAM → H2D slab               RAM CPU lease
     grouped CUDA                  transient CUDA               CPU executor
           └─────────────────────────────┬────────────────────────────┘
                                         ▼
                              stable weighted aggregation
```

Core contracts expose IDs, byte budgets, state and interfaces. Win32 handles
and CUDA types remain behind backend boundaries.

## Container lifecycle

```text
UNOPENED → MANIFEST_VALID → PACKS_VALID → PLANNED → READY
    └────────────── any validation/planning error ─────────► FAILED
READY → DRAINING → CLOSED
```

`READY` requires exact schema/ABI support, valid hashes, compatible SM target,
and feasible declared budgets. No best-effort integrity mode exists.

## Expert lifecycle

```text
ABSENT → SSD_LOADING → RAM_READY → GPU_UPLOADING → VRAM_READY
   ▲          └──────────── failure ───────────────────────► FAILED
   └──────────────── safe eviction after leases/events ──────────────┘
```

Rules:

- one load and one upload in flight per expert key;
- waiters share the same operation;
- RAM publishes only after complete read and checksum;
- VRAM publishes only after copy completion event;
- leases, scheduler reservations and incomplete CUDA events prevent eviction;
- credits are acquired before allocation and returned exactly once;
- cancellation removes a waiter, not data still required by another request;
- short read, checksum or CUDA error fails all dependent requests;
- RAM cache, pinned staging, VRAM resident/transient, workspace and KV budgets
  are independent.

`stored_bytes` and `device_bytes` are separate capacity claims. Zero
`device_bytes` preserves the legacy equal-size path. Expanding admissions such
as compact DeepSeek FP4 → SM86 INT8 must declare the exact hot allocation before
I/O begins; the cache reserves that larger value and refuses work that cannot
fit. Upload completion may shrink a reservation but may never exceed it.

The key is `(model_content_hash, layer, expert, quant_abi)`.

## Heterogeneous execution

An `ExpertWorkGroup` represents one expert and every microbatch row selecting
it. Executors are:

- `cuda_resident`: published device directory entry;
- `cuda_slab`: bounded RAM→pinned→device transient execution;
- `cpu_local`: host lease and bounded CPU pool;
- future `remote_worker`: same semantic output from another node.

Every executor writes a per-selection output. Device aggregation applies
routing weights in stable `(request, row, top-k slot)` order. Completion order
must not alter numerical order.

CPU execution uses compact result slots. Each CPU work group retains the global
selection ID for input-row lookup and maps it to a unique compact output slot.
Only `cpu_selection_count × hidden × sizeof(float)` is copied to the device;
the aggregation kernel resolves non-GPU selections through a bounded
`cpu_slot_by_selection` table. The mapping changes storage only, never routing
or aggregation order.

Planner decisions use measured queue, transfer and compute costs. A missing
expert is never treated as zero and a timeout never reduces top-k.

`HybridDispatchPlanner` receives one unique candidate per routed expert. It
keeps resident GPU experts fixed, assigns forced paths explicitly, then greedily
balances flexible RAM misses by projected critical path. Current synchronous
H2D is modeled as serialized before GPU expert compute; CPU compute may overlap
that GPU path. Stable ties remain on CPU to avoid unnecessary placement
mutation. Invalid, duplicate, unavailable, or over-bound plans fail closed.

CPU/GPU/H2D cost EWMAs are bounded state. The newest 256 per-expert reason/cost
snapshots form a circular diagnostic trace; aggregate counters are emitted for
benchmark windows. The trace affects neither routing nor aggregation order.

The CPU executor owns a maximum worker pool but activates only its calibrated
subset. Lazy calibration tests physical-core and logical-thread estimates with
two bounded gate/down tiles on the first real batch. Persistent scratch,
precomputed row pointers, multi-row weight reuse, and AVX2 software prefetch do
not change FP32 activation arithmetic or compact output-slot semantics.
Telemetry reports calibration cost, chosen threads/tiles, selections, logical
weight bytes, and effective bytes/s; the latter is not raw DRAM bandwidth.

## Placement policy

The cache uses bounded frequency/reuse evidence with aging. RAM→VRAM promotion
must outperform the conservative H2D cost and displace only a strictly colder
entry. Current execution continues on an available path while asynchronous
promotion can benefit future tokens.

Cache temperature also includes exact selected-router score mass and peak
score, encoded at Q20 precision. Score evidence ages with frequency and affects
victim/admission ordering only; it never changes router top-k, routing weights,
or aggregation. Non-finite feedback is ignored and all arithmetic saturates.
The entry snapshot exposes frequency, score evidence, and final temperature for
diagnostics.

At warmup/request barriers, admitted promotions drain and placement freezes.
The measured frozen epoch performs no promotion, expert H2D, or policy mutation.

The promotion predictor retains at most 4,096 histories, requires two recent
observations, expires them after 192 routed-layer epochs, and permits one
in-flight promotion. Score-weighted saved CPU debt determines priority only
after current work is planned. Stale pending work is cancelled. A completed
promotion is useful only when a later route finds it GPU-resident; selection
after eviction or TTL expiry is charged as wasted bytes. Prefetch never weakens
strictly-colder admission or current-route references.

Operators select a policy goal with `-PlacementProfile` (PowerShell) or
`--placement-profile` (HTTP server):

| Profile | Policy | Evidence boundary |
|---|---|---|
| `latency` | Admit prefetch after one recent observation and remove the extra admission margin. | More aggressive warming; not yet an independent throughput SLO. |
| `balanced` | Require two recent observations and use measured CPU/H2D/GPU critical-path cost. | Qualified default for the reference 80B deployment. |
| `capacity` | Disable speculative prefetch and opportunistic RAM→VRAM execution; retain mandatory cold fallback. | Minimizes churn for larger working sets; not a promise that CPU execution is faster. |

These are policy goals, not expert percentages. Resident experts stay on the
GPU, unavailable host experts still take the required load path, and all three
profiles preserve exact routing and stable aggregation. RAM and VRAM budgets
remain independent operator inputs.

## Windows storage

- pack files are opened read-only with overlapped I/O;
- unbuffered I/O is used only when file offset, length and buffer alignment
  meet effective volume constraints;
- one bounded IOCP/pool serves reads—never one blocking thread per expert;
- staging buffers have explicit owner/generation and fixed capacity;
- durable RAM cache is pageable so pinned staging cannot be exhausted by
  retention;
- useful, requested, physical-read and overfetch bytes are measured separately;
- EOF, short completion, checksum mismatch and device removal fail closed.

## CUDA ABI

`expert-pack-sm86-int8-row-v1` consumes symmetric per-row INT8 expert weights
and FP32 scales. Expert function:

```text
down(silu(gate(x)) * up(x))
```

Resident/transient descriptors contain only device pointers, declared geometry,
dtype, ABI, workspace and completion event. Host pointers are never published
in the device directory. Pointer generations prevent ABA reuse after eviction.

The straightforward FP32-activation kernel is the numerical oracle, not the
performance contract. Optimized kernels may change internal tiling/dtype only
behind a versioned kernel ABI and correctness tolerance.

## Qwen3-Next runtime

The runner implements full-attention GQA with partial RoPE/output gate and
Gated DeltaNet with persistent Conv/recurrent state. Dense projection, routed
expert grouping and aggregation operate on a microbatch. KV/Conv/DeltaNet state
is isolated by worker slot; weights/cache are shared.

The KV implementation uses on-demand FP16 pages:

```text
logical FP16 KV bytes = 24 KiB × admitted context tokens
```

KV is reserved in 256-token, 6 MiB superpages spanning all twelve full-attention
layers. Page credits are acquired per request, physical pages are allocated on
first use, and released pages enter a bounded reuse pool. Online-softmax
attention does not allocate a score array proportional to context length.

Only 4096 context with capacity four is certified. The model's 262K position
metadata remains outside the runtime guarantee until efficient prefill and
long-context gates pass.

## Worker protocol

The local line-framed protocol supports:

- startup `ready` with protocol/capacity, causal prefill chunk size, and KV page
  geometry;
- protocol-v3 `BEGIN` with an exact context reservation, then position-ordered
  causal prefill in chunks no larger than the advertised worker capacity;
- `STEP` to decode several active request IDs together;
- `STATS` for current KV page allocation/reservation;
- `END` to release/cancel request state;
- `SHUTDOWN` for orderly worker exit.

Unexpected message type, duplicate ID, invalid capacity or mismatched response
is fatal to the affected control flow. The front-end serializes worker commands
and continuously batches compatible decode waiters within a bounded window.

## Request lifecycle

```text
QUEUED → ADMITTED → PREFILL → DECODE → STREAMING → COMPLETE
             └──────── failure/cancel ───────────► FAILED/CANCELLED
```

Admission reserves capacity before streaming headers. Queue and worker-slot
waits have deadlines. Client disconnect is checked before scheduling another
decode step; cancellation always closes the generator and sends `END` when the
worker state is active.

## Telemetry

The runtime distinguishes SSD misses, RAM hits, VRAM hits, useful/read/uploaded
bytes, cache high-water marks, executor time, batch rows, TTFT and inter-token
latency. Metrics windows are bounded. A benchmark must identify cold/warm/frozen
placement and cannot infer hot-path throughput from configuration alone.

Expert-lane telemetry distinguishes CPU and resident-GPU selections and time,
reports nanoseconds per selection, and accounts separately for compact CPU
results and selection-map H2D bytes. GPU time uses CUDA events. Observed
CPU/GPU overlap is the positive difference between the sum of both lane times
and the enclosing expert-phase wall time; because transfer and aggregation are
inside that enclosing phase, this is a lower bound rather than a complete
critical-path trace.

Chunked prefill shares the normal transformer microbatch implementation. The
full-model gate requires its generated sequence to equal scalar prefill plus
decode exactly. The implementation does not claim FlashAttention or an
efficient 262K path.
