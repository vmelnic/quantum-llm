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
- each active device route owns an independent bounded pin token, so one
  request waiting on expert readiness does not invalidate or serialize another;
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

`source_abi` and the key's target `quant_abi` form a fail-closed pair. Expert
Pack v1 records continue through their existing validator and uploader.
DeepSeek compact records are SHA-256 checked as complete 13,369,344-byte
staging payloads, decoded only inside the CUDA uploader, and published as exact
25,198,592-byte SM86 slots. Compact source bytes are never exposed through the
host INT8 executor.

FP8 shared experts use a separate source ABI with 128×128 block scales but
converge on the same SM86 slot. Source ABI therefore selects decoding semantics;
target ABI alone never guesses how bytes should be interpreted. Shared experts
are integrity-loaded through the cache, then planner-pinned as always-active
model state rather than admitted by routed frequency.

`ResidentExpertSet` owns those long-lived leases. Its startup path is serial by
design: one pinned source slot bounds host memory while VRAM accumulates only
the declared final allocations. Duplicate keys fail before I/O. A load failure
releases the partial set and trims its now-unreferenced cache entries.

Dense block-scaled FP8 matrices use `DeepSeekDenseMatrix`, not a synthetic
expert key. Admission validates 128×128 source geometry, rejects E4M3FN/UE8M0
NaNs, derives a row-INT8 matrix, and exposes the existing `Int8Matrix` GEMV ABI.
Source and device byte claims remain explicit; dense allocations are model
state rather than router-evictable cache entries.

`DeepSeekDenseSet` validates the complete name/shape/byte plan, loads matrices
serially through one maximum-sized pinned slot, and publishes the collection
only after all matrices succeed. Hot paths bind matrix pointers once during
model construction; string lookup is not part of token execution.

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

DeepSeek layer execution exposes cache misses as a resumable state-machine
boundary. Attention/router results remain in bounded request state while the
outer scheduler acquires the exact missing experts. Resume replans the complete
top-k, executes only when every dependency is available, and releases the
request's directory pin after the FFN stream completes.

The immutable DeepSeek routed catalog contains exactly 43 × 256 records in
layer-major order. Each record reconstructs one `PayloadRecord` from six
exact-cover SafeTensors extents and declares both compact FP4 source bytes and
expanded SM86 bytes. Lookup does not parse files, allocate, or hash in the hot
path. Catalog generation hashes the authoritative source once; normal cache
admission verifies the selected expert again before publication.

The DeepSeek outer scheduler converts controller misses into catalog-backed
`ExpertCache::acquire` handles. Its independent credits bound active requests,
cache waiters, and layer advances per event-loop poll. Completed acquisitions
become request-owned leases; a request becomes runnable only when its entire
exact top-6 route is present. Other runnable requests continue while a route is
on SSD/I/O/H2D. Shared expert 256 is not silently loaded from the routed catalog:
it must be held by the resident model set, otherwise the request fails closed.

DeepSeek request admission also budgets the complete output surface: HC-head
workspace, 129,280 FP32 logits, and one sampled token. Embedding and head
bindings point into the immutable typed model state. The runtime rounds at the
same BF16 boundaries as the reference before final normalization/projection;
greedy argmax is device-side, so ordinary decode does not copy full logits to
the host.

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

DeepSeek ingestion uses an exact-cover gather descriptor. Six buffered,
overlapped SafeTensors reads target disjoint offsets in one pinned staging
buffer, then a whole-payload checksum gates CUDA admission. Buffered children
are intentional because tensor offsets need not satisfy sector alignment;
unbuffered aligned reads remain the contiguous-pack path. No full checkpoint or
per-expert payload copy is created.

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
