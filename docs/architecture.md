# Architecture

## Objective

Run sparse MoE checkpoints larger than local VRAM and RAM without transferring
all model weights for every token. Dense tensors remain GPU-resident. Only the
experts selected by the router enter the placement and execution pipeline.

## System view

```text
                       compile time
SafeTensors ───────────────────────────────────────────────────────────┐
    │ strict architecture adapter                                     │
    │ row-streaming INT8 quantization                                  │
    ▼                                                                 │
Qwen Expert Pack / DeepSeek compact bundle                            │
    ├── manifest + checksums                                           │
    ├── dense/shared state ────────────────────────┐                   │
    └── indexed expert records ────────┐           │                   │
                                        │          │                   │
                       inference time   ▼          ▼                   │
OpenAI HTTP ─► admission ─► tokenizer ─► request state ─► dense CUDA  │
                                              │              │         │
                                              ▼              ▼         │
                                        exact router ─► expert plan    │
                                                           │           │
                    ┌──────────────────────────────────────┼────────┐  │
                    ▼                                      ▼        ▼  │
               VRAM resident                       RAM resident    SSD │
               grouped CUDA                    CPU or H2D slab     IOCP│
                    └───────────────────┬───────────────────────┘      │
                                        ▼                              │
                              stable weighted aggregation ─► logits ──┘
```

## Components

### Compiler

The Python compiler reads SafeTensors through read-only mappings and converts
one tensor/row at a time. Architecture adapters classify every tensor; unknown
or missing tensors fail conversion. Output is committed atomically only after
independent validation.

### Core

The portable C++ core owns strict JSON parsing, checksums, feasibility, byte
budgets, and admission decisions. It has no Win32 or CUDA types in its public
contracts.

### Storage and cache

Windows IOCP performs bounded asynchronous reads. Expert entries move through:

```text
ABSENT → SSD_LOADING → RAM_READY → GPU_UPLOADING → VRAM_READY
   └──────────────────────── failure ───────────────────────► FAILED
```

One in-flight load/upload exists per key. Leases and CUDA events prevent reuse
or eviction while data is referenced. RAM, pinned staging, VRAM resident, VRAM
transient, KV, and workspace budgets are independent.

Device-directory protection is request-scoped rather than global. Each route
receives a bounded pin token covering its ready unique experts; overlapping
requests may hold independent tokens, including overlapping expert sets. A
token is released exactly once after its dependent CUDA work completes. The
directory serializes only the small planning/release control operation, not the
lifetimes of active request routes.

### Scheduler and placement

The router is backend-exact. Qwen uses its declared softmax/top-k behavior;
DeepSeek uses `sqrt(softplus)`, token-table selection on hash layers or
score-plus-bias selection on learned layers, then normalizes the unbiased
selected weights. Work is grouped by `(layer, expert, quant ABI)` across the
active microbatch. Ready GPU work proceeds while cache misses are loaded or
executed on CPU. Weighted aggregation uses a stable order.

CPU expert results return in a compact selection buffer rather than a dense
`rows × top_k` buffer. A device-side selection map restores the original stable
top-k order during weighted aggregation.

Adaptive placement observes reuse and measured execution cost. Placement is
frozen at a safe request/warmup boundary for deterministic latency measurement;
the frozen hot path performs no demand expert-weight H2D on resident routes.
While frozen, a bounded re-promotion may still upload RAM-resident hot experts
to VRAM, gated by a stale-victim byte budget so a repeating route whose working
set exceeds the VRAM budget cannot ping-pong.

The runner measures resident GPU expert work with per-layer CUDA events and
CPU expert work with the host steady clock. It publishes selection-normalized
lane costs and a conservative overlap lower bound. These measurements form the
cost basis for dynamic placement; they do not prescribe a fixed CPU share.

The portable hybrid dispatcher makes a bounded plan only when a layer has a
non-resident expert. GPU-resident work is fixed. A RAM-resident miss may execute
locally or upload only if measured CPU versus serialized-H2D-plus-GPU cost
reduces the projected layer critical path and cache admission protects hotter
or in-use residents. CPU/GPU compute may overlap; the planner models an H2D
upload conservatively as serialized before GPU expert execution, even though
the uploader itself completes through per-upload CUDA events on a dedicated
completion thread rather than stream-wide host syncs. Decision reasons,
alternative costs, and EWMAs are retained in bounded telemetry.

During unfrozen placement, exact selected routing weights feed cache
temperature. Frequency, accumulated score mass, and peak score are stored as
bounded fixed-point values and aged together. Eviction compares this composite
temperature before deterministic pressure/recency tie breaks. Frozen epochs do
not copy scores or mutate this evidence.

Sequence-local prefetch promotes a repeatedly CPU-executed RAM expert only
after measured saved CPU debt amortizes upload. Candidate history, in-flight
credits, score EWMA, and TTL are bounded. Current route references are
protected before admission, and only a strictly colder victim is eligible.
Router observations distinguish a GPU-resident useful prediction from a
prefetch that was evicted before use; useful, wasted, stale, and credit metrics
remain explicit.

The CPU executor keeps FP32 activations and the INT8-per-row weight ABI. Its
worker pool is persistent; per-batch intermediate/offset/validation scratch is
reused. AVX2 kernels reuse each weight vector across routed rows and issue
bounded software prefetches. A first-real-batch calibration chooses active
workers and gate/down tiles under a time limit, then publishes the selected
configuration and effective weight traversal rate.

DeepSeek placement now crosses the real decode scheduler boundary. For every
suspended layer, the scheduler gives the bounded hybrid planner the exact
routed union and the current host/device residency. A CPU decision is resolved
as a typed host lease; all remaining selections are resolved as typed device
leases. Both lease classes stay alive until the layer finishes, so the
controller receives stable compact-record spans and the CUDA directory cannot
evict an in-flight selection. If a host copy is evicted between inspection and
lease acquisition, that selection safely falls back to the device path.

The controller owns one reusable hybrid FFN workspace per active request. It
only executes the placement supplied by the scheduler and aggregates CPU and
GPU selections in stable route order; it does not infer placement from pointer
location or silently promote a host record. The current qualification uses
forced planner costs to exercise one CPU plus five GPU experts exactly. Real
serving policy must instead be seeded and updated by measured CPU, GPU, H2D,
storage, and queue costs.

Completed scheduler routes feed a fixed-cardinality route census before their
leases are released. Its lazy-decayed heat, lifetime count, and consecutive
reuse evidence produce a deterministic warm set under global and per-layer
bounds. Two independently authenticated generations survive torn writes and
are rejected when the model hash or quantization ABI changes. The complete
DeepSeek table is about 518 KiB and contains no prompt, token, activation, or
request identity.

Planner seeds come from a measured placement profile, not family defaults.
Qualification times only routed CUDA selections, all-core compact CPU experts,
and pinned H2D transfer, then requires nonzero sample evidence for each. A
separate fail-closed solver subtracts explicit fixed allocations and emergency
headroom from operator-approved host/device envelopes and returns whole compact
expert slots plus the matching memory-governor configuration. Required minima
are never silently reduced.

DeepSeek dense, typed, and always-resident shared descriptors are parsed by one
runtime loader used by both qualification and the production worker. It binds
every payload to its declared SHA-256, canonical extent count, dtype, geometry,
and checkpoint root before any model allocation begins. This prevents the
worker and benchmark tools from developing separate metadata interpretations.

### Native model backends

The Qwen3-Next backend implements alternating full attention and Gated
DeltaNet, output-gated attention, partial RoPE, shared and routed experts, and
isolated KV/Conv/DeltaNet state per slot; its routed experts are consumed as
INT8 per-row records (quant ABI 1) or FP4-E2M1/UE8M0 block-32 records (quant
ABI 3) through packed `__dp4a` kernels. The DeepSeek-V4-Flash backend
implements its native attention/CSA/HCA, routing, shared/routed FFN, compact
FP4 expert path, and persistent request state. CUDA is compiled for SM86.

### HTTP service

The Python front-end owns tokenization, bounded admission, continuous decode
batching, cancellation, SSE, metrics, and the OpenAI-compatible wire contract.
The C++ worker owns model state and token selection. Their local protocol is
line-framed and versioned. Protocol v5 supports exact context reservations, KV
telemetry, batched `STEP`, explicit prefetch state, one-or-more returned
tokens for the speculative-execution boundary, and retained conversation
sessions (`BEGIN … RESUME` / `END … RETAIN` / `DROP`) so multi-turn chat
prefills only the delta tokens of a turn. The front-end matches retained
sessions by token-prefix and stays stateless for clients and workers without
retention support.

## Memory placement

For the tested Qwen3-Next container:

- dense pack: approximately 3.90 GiB and persistent on the GPU;
- experts: approximately 72.4 GiB total, cached across SSD/RAM/VRAM;
- active experts: ten per routed layer and token;
- VRAM expert cache: operator-budgeted, 18 GiB in the tested profile;
- RAM expert cache: 48 GiB in the tested profile.

Cache budgets are upper bounds, not startup reservations. A cold server uses
roughly dense + runtime state; repeated requests heat expert residency until
the budget is reached.

## Serving memory hierarchy (measured)

The serving design is a three-tier hierarchy. Measured on the RTX 3090 host
(2026-08-10, docs/benchmarks.md §Host bandwidth):

```text
+---------------------------------------------------------------+
| VRAM 24 GB (RTX 3090) — the only memory the GPU computes from  |
|                                                               |
|  - hot routed experts: operator-budgeted cache (12-18 GiB)    |
|    FP4 slots: 12.75 MiB (DeepSeek) / ~1.9 MB (Qwen) per       |
|    expert — roughly twice as many residents as int8           |
|  - shared experts + dense pack (~4-5 GiB)                     |
|  - conversation KV cache + workspaces                         |
|  - transfer cost: none — native speed (40-47 tok/s hot Qwen)  |
+------------------------------^--------------------------------+
                               | PCIe Gen3 x16: 12.46 GiB/s measured
                               | every expert missing from VRAM
                               | crosses this link, no exceptions
+------------------------------|--------------------------------+
| RAM 64 GB (expert budget 40-48 GiB)                           |
|  - warm experts that did not fit VRAM; staging only — the     |
|    GPU cannot compute from RAM, entries upload on demand      |
|  - Qwen FP4 pack (45.36 GB) fits entirely: disk leaves the    |
|    steady state after warm-up                                 |
|  - DeepSeek pack (147 GB) fits only ~27%: most cold experts   |
|    still fall through to disk                                 |
+------------------------------^--------------------------------+
                               | SATA sequential: 0.47 GiB/s measured
                               | (26x slower than PCIe)
+------------------------------|--------------------------------+
| DISK: the immutable expert packs (Qwen 45-72 GB, DeepSeek     |
| 147 GB). Cold experts live here; every read from this tier    |
| dominates request wall time (S1-DeepSeek probe: storage wait  |
| was larger than the entire decode wall on settled turns).     |
+---------------------------------------------------------------+
```

Per-token flow for one routed layer: the directory checks VRAM — a hit
computes immediately at zero transfer cost; a miss uploads from RAM at PCIe
speed if RAM-resident, otherwise reads from disk first (SATA-bound) and then
uploads. Eviction never selects an entry pinned by an in-flight route
(structural `route_pinned` skip), so supply cannot deadlock against the
forward pass that triggered it.

Two accounting facts decide how much each tier holds: VRAM residency doubled
when slots became FP4 (12.75 MiB) instead of int8 (25.2 MB), and the tier
capacities follow from the pack sizes above — Qwen FP4 eliminates the disk
tier after warm-up, DeepSeek cannot on 64 GB RAM. Hardware levers act on
exactly one term each: an NVMe pack tier attacks the SATA miss path
(~10x), RAM >= 192 GB removes the disk tier for DeepSeek entirely, and VRAM
is fixed at 24 GB on this host (which is why FP4 slot size matters).

## Context memory

The model advertises 262,144 positions. Qwen3-Next has twelve full-attention
layers. The runner stores K/V in FP16 superpages shared across those layers:

```text
page tokens                    256
one layer K + V                512 KiB/page
twelve full-attention layers     6 MiB/page
logical KV per token            24 KiB/request
```

At admission, each request reserves `ceil((prompt + output) / 256)` page
credits. Physical CUDA pages are allocated only when prefill/decode first
enters them, then retained in a reusable high-water pool. A released request
returns its pages and credits. The historically qualified 4096 × four-slot
profile needs at most 64 pages, or 384 MiB—not 768 MiB reserved at startup.
The reference lifecycle currently advertises 65,536 tokens, but on-demand
allocation does not make that larger limit correctness- or latency-qualified.

Attention uses online softmax and constant shared memory instead of storing one
score per context token. This removes the previous kernel launch ceiling for
large contexts. Prefill runs in causal chunks whose size is an independent
worker setting (`--worker-prefill-chunk-tokens`, default 256), decoupled from
the decode batch capacity. Full-attention cache writes and DeltaNet state
updates remain position ordered, while projections and MoE work reuse the
microbatch path. This is bounded chunked prefill, not FlashAttention: full
attention remains quadratic. Prefill-specific workspaces/kernels, RoPE
validation, and staged SLO/correctness gates are required before qualifying
beyond 4096 tokens.

## Failure model

- malformed or mismatched model data prevents readiness;
- no request runs with a partial top-k;
- bounded credits prevent unbounded RAM, VRAM, staging, or queue growth;
- client disconnect cancels the C++ request before another decode step;
- worker failure fails dependent requests explicitly;
- deployment identity includes build ID and manifest/index hashes;
- source checkpoints are never deleted unless an operator passes the explicit
  destructive conversion confirmation.

## Extension boundary

The core is intended to remain platform-neutral. Future backends fit behind the
same resource interface:

```text
core/        manifest, cache policy, scheduler, request state
backends/    cuda, future metal
storage/     windows IOCP, future macOS async storage
distributed coordinator, expert worker, placement protocol
```

Distributed execution will move activations to the node that owns an expert,
not pretend that remote RAM is local memory.

DeepSeek's compact-source and derived SM86-cache boundary is specified by the
[DeepSeek compact pack](deepseek-compact-pack-v1.md) and
[runtime contract](expert-runtime.md). It deliberately does not reinterpret
the Expert Pack v1 ABI used by the Qwen/OLMoE backend.
Its resident dense and dtype-preserving tensors are published as one model
transaction. Layer construction resolves and geometry-checks names once, then
execution consumes stable pointer bindings rather than performing string
lookups or per-layer uploads in the hot path.

The external-knowledge research path lives in the standalone public
`memory-expert` project (extracted from this repository; see
[Engineering history](history.md#memory-expert-extraction)). It attaches
bounded retrieved records as a prefilled K/V prefix read by the frozen
model's native attention; it does not place source text in the request
context and does not alter MoE expert placement.

One `DeepSeekRequestState` owns the mutable attention, FFN, HC-head/logits, and
four-stream state for all 43 layers and retains the immutable model object that
its bindings reference. The
factory first computes the exact complete CUDA footprint, checks a caller-owned
per-request budget, binds every layer, and only then publishes a fully built
request. Partial allocation or a missing layer never becomes schedulable. The
compression schedule is explicit: three pure sliding-window layers, twenty
ratio-four layers, and twenty ratio-128 layers.

`DeepSeekDecodeController` advances one layer at a time. It composes attention,
exact routing, directory planning, routed/shared execution, and pin release. A
cold route returns its exact missing expert IDs while preserving the computed
attention/router state. The scheduler can run other requests while the cache
publishes those experts, then call `advance()` again to replan and resume the
same FFN. Cancellation and every failure path release any active pin token.

`DeepSeekExpertCatalog` is the immutable source directory behind that boundary.
It resolves every main-model `(layer, expert)` in O(1) to six SafeTensors
extents and the exact source/device ABI geometry. The catalog contains metadata
and hashes only; the original checkpoint remains authoritative and recoverable.
This keeps startup metadata bounded while allowing future placement planners to
point the same logical expert at local SSD, RAM, VRAM, or a remote worker.

`DeepSeekDecodeScheduler` is a single-owner event-loop component above the
controllers. Each `poll()` first consumes already-completed cache futures, then
advances a bounded number of runnable layers, and finally fills a bounded global
acquisition window. It never blocks on I/O. Request and acquisition queues are
round-robin; the cache still deduplicates identical expert loads across
requests. Acquired leases live through FFN resume, while cancellation and every
terminal failure unwind pending waiters, leases, and controller pins.

The scheduler submits a bounded union of `(layer, expert)` records to
`IExpertStore`, rather than exposing individual cache futures as its placement
contract. Every member names an explicit target (`device` or `host_ready`) and
the result owns the corresponding device or validated compact-host lease. The
store publishes either the complete ordered placement set or one explicit
failure; a host target never silently promotes or falls back. The local store
still deduplicates immutable records through `ExpertCache`, while remote-owner
stores can implement the same batch boundary without changing request state.
Global credits count experts, not batch objects, so batching cannot bypass I/O
or memory bounds.

`MemoryResourceGovernor` is the single admission authority above these tiers.
Dense state, KV pages, request workspaces, and network staging are explicit
reservations. Host and device caches implement `ITrimmableMemoryTier`; before a
reservation is admitted, lower-priority tiers are trimmed toward declared
protected floors. If live leases still exceed the physical budget, admission
fails with backpressure instead of relying on pagefile or CUDA OOM behavior.

The same request owns the model edges: BF16 embedding gather expands directly
into its four input streams; after layer 42, HC-head collapse, final RMSNorm,
the untied BF16 vocabulary projection, and greedy argmax write into a fixed
615,460-byte I/O allocation. The 2+ GiB embedding/head weights remain part of
the immutable typed model state and are never copied per request.
