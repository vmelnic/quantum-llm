# Roadmap

This roadmap records direction, not a compatibility promise.

## DeepSeek-V4-Flash backend

1. ~~Strict source contract, compact expert ABI, resident dense/typed/shared
   state, HCA, CSA cache/index primitives.~~ Implemented and independently
   qualified on the pinned checkpoint.
2. ~~Transactional model publication and the first complete ratio-four
   attention decode sublayer.~~ Implemented for token zero with an independent
   full-graph oracle.
3. ~~Sequential token-three gate that emits and consumes the first compressed
   slot through the composed executor.~~ Implemented with real RoPE and an
   independent four-token oracle.
4. Optimize the measured layer workload: tiled/vectorized dense GEMV, fused
   normalization/RoPE/QAT, grouped output kernels, batching, and parallel top-k.
   Aligned INT8 vector loads are complete; the end-to-end layer improved 6.6%.
   Standalone activation-INT8 DP4A was measured, regressed, and removed.
   The eight `wo_a` groups now execute in one launch without changing results.
5. ~~Compose routing, resident shared expert, streamed routed experts, FFN HCA,
   and attention into the first complete transformer block.~~ Qualified on the
   real layer-2 compressed token with an independent full-block oracle.
6. ~~Extend ownership/state across 43 layers.~~ A preflight-budgeted request
   transaction now retains the immutable model plus all 43 attention/FFN
   states. Concurrent directory pin tokens and the suspend/resume layer control
   loop are complete.
7. ~~Index every main-model routed expert without copying the checkpoint.~~ The
   atomic catalog contains 11,008 records and 66,048 source extents; the runtime
   parser and real first/last-expert cache/CUDA boundary are complete.
8. ~~Add the outer async scheduler.~~ The non-blocking round-robin loop bounds
   concurrent requests, layer advances, and cache acquisitions; it retains
   leases across suspended layers and was qualified with six catalog-backed
   cold routed experts under a two-acquire limit.
9. ~~Add embedding, output head, and greedy sampling.~~ The request-owned I/O
   state reuses resident typed weights, is included in preflight accounting,
   and matches an independent full-vocabulary oracle.
10. ~~Add official tokenizer integration and a token-level `[0,43)`
    controller.~~ A five-token real chat prompt now preserves state across all
    layers and produces the expected decoded response boundary. Layer-major
    prefill reduced that prompt from 19.91 to 3.74 seconds without changing the
    output. Multi-token autoregressive state is now functional; its first real
    decode measurement was 0.261 tok/s because all 258 routed selections miss
    the 64-slot global cache. Parallel routed down-selection plus stable
    aggregation is complete and preserved a three-token sequence; grouped
    prefill rows, Tensor Core weight kernels, and the persistent HTTP worker
    remain.
11. ~~Publish a resumable compact routed-expert pack.~~ The 147.17 GB artifact
    contains 43 aligned layer shards and preserves all 11,008 authenticated
    FP4 records. It reduced a warm five-token prompt from 3.74 to 3.61 seconds,
    but the cold run still read 12.57 GB and took 29.05 seconds. Next replace
    transient allocation with persistent device slots, retain a learned active
    set across decode tokens, and use grouped/Tensor Core execution. Bounded
    persistent device slots are complete (795 reuses from 107 allocations in
    the real prompt); active-set placement and grouped compute remain. Compact
    storage is not itself the compute-ready representation.

## Phase 1 — long-context and cold-path production work

1. ~~Paged, on-demand FP16 KV cache with per-request credits.~~ Implemented in
   worker protocol v3.
2. ~~Mixed context lengths without reserving maximum context for every slot.~~
   Implemented through page-credit admission.
3. ~~Bounded causal chunked prefill vertical slice.~~ Implemented with a
   four-token current profile and scalar/full-model equality gate.
4. Decouple prefill chunk size from request concurrency; add adaptive chunks
   and FlashAttention-class full-attention kernels.
5. Certified 8K → 16K → 32K → 64K gates.
6. Workload-trained warm expert sets and non-blocking prefetch.
7. End-to-end SLOs for TTFT, inter-token latency, throughput, and memory.

## Phase 2 — service hardening

1. Native Windows service/supervisor and non-interactive identity.
2. External TLS/auth/rate limiting reference deployment.
3. Log rotation, traces, alerts, dashboards, and capacity reporting.
4. 24-hour soak, failure injection, CUDA/OOM and disk-corruption recovery.
5. Self-hosted SM86 CI and signed release artifacts.

## Phase 3 — platform-neutral core and Metal

Keep manifests, cache policy, scheduler, request state, and placement interfaces
platform-neutral. Add macOS async storage and a Metal backend. An Apple M4 host
can become a coordinator because unified memory avoids host→VRAM copies, but its
24 GB total memory and thermal limits remain explicit constraints.

## Phase 4 — distributed expert workers

Two machines do not expose one transparent RAM pool. The coordinator moves
compute to the node that owns an expert:

```text
coordinator: dense + attention + exact router
       │ activation, expert IDs, routing weights
       ▼
expert worker: local RAM/SSD/accelerator → gate/up/down
       │ weighted partial or strict per-expert output
       ▼
coordinator: stable aggregation
```

The protocol will use persistent TCP initially (including IP over Thunderbolt
Bridge between Macs), with separate control and data planes, binary framing,
manifest/quant ABI negotiation, credits, cancellation, epochs, idempotent
operation IDs, and explicit retry/failure semantics. It will not invent a new
transport reliability or cryptography layer.

Placement may choose local VRAM, local RAM, local SSD, remote worker memory, or
a replica set. Remote compute is admitted only when measured completion time is
better than local load plus compute.

## Phase 5 — larger models

For hundreds-of-billions to trillion-parameter MoE models:

- window the device expert directory;
- shard packs and indexes across storage/nodes;
- keep metadata bounded independently of weight count;
- use replica-aware placement for hot experts;
- optimize aggregate throughput across concurrent requests;
- preserve exact routing and explicit failure when any selected expert is
  unavailable.

The durable principle is: **weights remain near their storage/compute tier;
activations move, not whole experts per token.**
