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
line-framed and versioned; protocol v3 supports multiple active request slots,
exact context reservations, KV telemetry, and batched `STEP`.

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
returns its pages and credits. The default 4096 × four-slot profile needs at
most 64 pages, or 384 MiB—not 768 MiB reserved at startup.

Attention uses online softmax and constant shared memory instead of storing one
score per context token. This removes the previous kernel launch ceiling for
large contexts. Prefill now groups up to four consecutive tokens from one
request into a causal microbatch. Full-attention cache writes and DeltaNet state
updates remain position ordered, while projections and MoE work reuse the
microbatch path. This is bounded chunked prefill, not FlashAttention: full
attention remains quadratic and the chunk is tied to worker capacity. Larger
chunks, prefill-specific workspaces/kernels, RoPE validation, and staged
SLO/correctness gates are required before raising the certified 4096 limit.

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
