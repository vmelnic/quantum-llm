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
Expert Pack v1                                                        │
    ├── manifest + checksums                                           │
    ├── dense.qpack ───────────────────────────────┐                   │
    └── experts-*.qpack ────────────────┐          │                   │
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

### Scheduler and placement

The router is exact: it performs global softmax, top-k selection, and selected
weight renormalization. Work is grouped by `(layer, expert, quant ABI)` across
the active microbatch. Ready GPU work proceeds while cache misses are loaded or
executed on CPU. Weighted aggregation uses a stable order.

Adaptive placement observes reuse and measured execution cost. Placement is
frozen at a safe request/warmup boundary for deterministic latency measurement;
the frozen hot path performs no expert-weight H2D.

### Qwen3-Next backend

The current production candidate implements Qwen3-Next's alternating full
attention and Gated DeltaNet, output-gated attention, partial RoPE, shared
expert, routed experts, and isolated KV/Conv/DeltaNet state per slot. CUDA is
compiled for SM86.

### HTTP service

The Python front-end owns tokenization, bounded admission, continuous decode
batching, cancellation, SSE, metrics, and the OpenAI-compatible wire contract.
The C++ worker owns model state and token selection. Their local protocol is
line-framed and versioned; protocol v2 supports multiple active request slots
and batched `STEP`.

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

## Context memory

The model advertises 262,144 positions. The current runner preallocates FP32 KV
for every configured slot. Qwen3-Next has twelve full-attention layers, so KV
cost is:

```text
48 KiB × maximum_context × worker_capacity
```

The certified profile is four slots × 4096 tokens (about 768 MiB KV). Larger
values are configurable but are not certified. Paged, on-demand FP16/BF16 KV
and chunked prefill are required before long context becomes a production
claim.

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
