# Roadmap

This roadmap records direction, not a compatibility promise.

## DeepSeek-V4-Flash backend

1. ~~Strict source contract, compact expert ABI, resident dense/typed/shared
   state, HCA, CSA cache/index primitives.~~ Implemented and independently
   qualified on the pinned checkpoint.
2. ~~Transactional model publication and the first complete ratio-four
   attention decode sublayer.~~ Implemented for token zero with an independent
   full-graph oracle.
3. Sequential token-three gate that emits and consumes the first compressed
   slot through the composed executor.
4. Optimize the measured layer workload: tiled/vectorized dense GEMV, fused
   normalization/RoPE/QAT, grouped output kernels, batching, and parallel top-k.
5. Compose routing, resident shared expert, streamed routed experts, FFN HCA,
   and attention into the first complete transformer block.
6. Extend ownership/state across 43 layers, then add embedding, head,
   tokenizer, sampling, and the HTTP worker boundary.

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
