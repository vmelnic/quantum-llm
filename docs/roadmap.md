# Roadmap

Status: active dependency-ordered work, 2026-08-25. This is the only current
backlog; completed and rejected experiments are not hidden roadmap items.

## 1. Re-establish a clean baseline

The 2026-08-25 baseline is Windows Release/CUDA build, 5/5 CTest and the
canonical 93-test compiler/server contract. After the single-parser service
change, the local canonical suite discovered 93 tests, passed 85 and skipped 8
optional NumPy cases; the server subset passed 56/56. Real common service
regressions passed at 23.85 tok/s after first visible for Qwen `chat hi`, 15.15
tok/s for the latest Ornith chat ledger entry, and 1.24 tok/s for DeepSeek
`chat hi` before cancellation. After the cancelled Pi prefill and clean
restart, DeepSeek passed again at 1.37 tok/s after first visible. Hold these as short-context regression gates; they are not maximum
context results or claimed q-star improvements.

## 2. Lock numerical and harness correctness

- preserve the official-checkpoint FP4 source gate;
- extend the bounded official-reference comparison to stochastic sampling and
  visual inputs;
- qualify long tool histories and explicit truncation through Pi;
- preserve the one-parser streaming contract across every artifact protocol;
- fault-inject cancellation, timeout, worker restart and KV exhaustion at a
  large populated prefix;
- never exchange exact F16 KV, context, top-k or `xhigh` reasoning for speed
  without explicit approval.

## 3. Extend the qualified universal hybrid sparse path

The pinned source is `ornith-ai/Ornith-1.5-35B-A3B` revision
`10fbf86fed7ecee4a061f8b499a618f46001cac1`. The source adapter is named
`hybrid_delta`; the official `qwen3_5*` strings remain source metadata only.
There is no Ornith runner, layer-count branch or deployment stack.

Capacity and traffic prerequisites for its declared 40-layer, 256-expert,
top-8 geometry are:

```text
routed disk/RAM records
  40 * 256 * 1,675,264 = 17,154,703,360 bytes = 15.9765625 GiB

routed CUDA payload touched by one scalar target call
  40 * 8 * 1,671,168 = 534,773,760 bytes = 0.498046875 GiB

exact F16 KV at 262,144 populated positions
  10 * 2(K,V) * 2 KV heads * 256 values * 2 bytes * 262,144
  = 5,368,709,120 bytes = 5 GiB

worst-case all-miss expert upload at 15 target calls/s
  0.498046875 * 15 = 7.470703125 GiB/s

scalar dense/shared/router/head payload scan
  1,108,002,880 bytes = 1.031908095 GiB/call

scalar weights including selected experts
  1,108,002,880 + 534,773,760
  = 1,642,776,640 bytes = 1.529954970 GiB/call

maximum-context scalar hot-byte lower bound
  1.529954970 GiB weights + 5 GiB exact KV
  = 6.529954970 GiB/call; at 15 calls/s = 97.94932455 GiB/s

recurrent request state
  30 * ((8,192 * 4) + (32 * 128 * 128)) * 4
  = 66,846,720 bytes = 63.75 MiB active mathematics
  = 191.25 MiB in the current state/checkpoint/retention allocation

draft/MTP state
  source tensors are preserved as qualified auxiliary records, but the MoE
  artifact advertises no exact MTP verifier; runtime draft state = 0 bytes
```

The 5 GiB exact KV fits progressively on the RTX 3090 beside bounded dense
state and the common 12 GiB routed-VRAM cache. The whole routed pack fits in
the 48 GiB host budget, but not in routed VRAM. The PCIe inequality is only a
necessary bound; it does not prove service throughput or small-GEMV efficiency.
After the fixed 12 GiB expert tier, 5 GiB KV, recurrent copies and 1 GiB device
reserve, startup must prove that resident dense/vision tensors, staged weights,
attention scores/probabilities and allocator overhead fit the remaining device
capacity. Those allocations are a startup/service gate, not assumed free.

The common routed provider has the prerequisites needed for this gate: a
generic standard/compact FP4 CPU ABI, all-layer cache identity, a monotonic
host bank when the pool fits, and exact adaptive CPU/GPU merge.

The 2026-08-25 qualification is complete:

- the official Xet snapshot and all 1,811 SafeTensors entries were validated;
- the `hybrid_delta` adapter compiled 10,240 fused logical FP4 experts and
  1,731 dense records with complete source-region coverage;
- source-to-artifact numerical qualification passed before transactional
  publication below `MODEL_ROOT`;
- the real `model.sh chat` `hi` gate passed for Ornith at 15.15 tok/s after
  first visible, then for Qwen at 25.05 and DeepSeek at 1.41;
- Ornith telemetry proved routed RAM/VRAM execution with 4,046 RAM hits and
  2,005,401,600 uploaded bytes, without a family-specific runner or service.

The gate does not establish maximum-context speed or broad Ornith behavioral
quality. Those remain numerical/harness work under sections 2 and 5.

Only after that real chat gate, consider the remaining FreeToken mechanisms in
this order:

- pin the fully populated final host bank instead of pre-faulting an empty
  pinned allocation, gated by startup latency and identical record ownership;
- move cache deduplication/victim selection/valid counts into the captured
  device path, gated by removed host synchronization and unchanged routes;
- rebuild the expert/KV split only at scheduler safe points, while preserving
  the validated 12 GiB DeepSeek profile and exact admission accounting;
- add full-layer prefill double buffering only when two artifact-declared
  layers fit and a complete populated prefill block proves that expert
  movement, rather than dense causal work, is the active bottleneck.

## 4. Keep rejected model integrations out of deployment

- Nemotron-H from BF16 through the generic FP4 profile is rejected: its
  corrected common service path still failed the real semantic gate;
- no Nemotron alias, source adapter or qpack is part of the supported model
  set; its HF cache is retained to avoid destructive redownloads;
- a future attempt requires explicit approval for the official NVFP4 format,
  W4A16 group-16 and FP8 Mamba ABIs, followed by numerical and real-chat gates;
- older Qwen3.5 and Qwen3-Coder deployments remain out of scope.

## 5. Close production operations

- authenticated TLS/rate-limited edge while the service remains loopback;
- durable service supervision, log rotation, metric retention and alerts;
- bounded session-RAM admission in a single budget shared with retained expert
  pages, based on actual artifact and parked-state bytes;
- documented backup/rollback for immutable artifacts;
- release gate covering build, source quality, behavior, Qwen/DeepSeek chat,
  harness, cleanup and rollback.

## Completion conditions

Qwen is complete only when 262,144 positions are populated with exact F16 KV
and the real coding harness sustains approximately 15 useful output tok/s with
acceptable TTFT/reuse behavior. DeepSeek is complete only when its exact routed
path achieves the accepted novel/settled targets with measured tier traffic.
Neither is complete because the API accepts the request or a microkernel is
fast.
