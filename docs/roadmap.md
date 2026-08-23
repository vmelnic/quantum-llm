# Roadmap

Status: active dependency-ordered work, 2026-08-23. This is the only current
backlog; completed and rejected experiments are not hidden roadmap items.

## 1. Re-establish a clean baseline

- pass the local Python/compiler/server suite after repository cleanup;
- build the Windows Release/CUDA target;
- run the same `hi` lifecycle through `model.sh chat` for Qwen and DeepSeek;
- stop the service and prove process/GPU cleanup;
- record only changed evidence in [Benchmarks](benchmarks.md).

## 2. Lock numerical and harness correctness

- preserve the official-checkpoint FP4 source gate;
- extend the bounded official-reference comparison to stochastic sampling and
  visual inputs;
- qualify long tool histories and explicit truncation through Pi;
- fault-inject cancellation, timeout, worker restart and KV exhaustion at a
  large populated prefix;
- never exchange exact F16 KV, context, top-k or `xhigh` reasoning for speed
  without explicit approval.

## 3. Add a universal multi-device substrate

- enumerate CUDA devices and publish compute capability, VRAM, bandwidth and
  P2P topology;
- replace the single-device global state with per-device allocators, streams,
  caches and request-state ownership;
- compile provider modules for declared architectures rather than one global
  `CUDA_ARCHITECTURES 86` binary;
- add artifact shard semantics for expert, KV-head, row/column and
  non-shardable tensors;
- keep model/service selection capability-driven; no P100/P40/Qwen branch in
  common code;
- account every peer/host transfer and synchronization per request.

## 4. Gate P100 exact-F16 KV attention

- verify `nvidia-smi topo -m`, negotiated PCIe width/generation and CUDA P2P;
- implement only the exact Qwen full-attention operation for SM60;
- shard KV heads evenly across the three P100s and keep attention compute with
  each shard;
- compare numerically with the existing exact F16 provider;
- include QKV/output movement and PLX/P2P latency in effective bandwidth;
- stop below the 150-200 GB/s gate; proceed toward 15 tok/s only if it passes.

## 5. Integrate heterogeneous Qwen execution

- let the artifact planner bind full-attention KV-head shards to P100 and
  dense/recurrent organs to the RTX 3090;
- preserve progressive allocation, parking, checkpoint, rewind and zero-delta
  resume per shard;
- keep sampling policy and MTP allocation semantics unchanged;
- run populated 262K prefill, multi-turn reuse and sustained decode through
  the real Pi harness;
- accept only measured useful output throughput with exact F16 KV.

## 6. Gate a Pascal DeepSeek expert provider

- test the P40 SM61 packed expert ABI independently against the SM86 numerical
  oracle;
- if it passes, give each device an owned expert cache and execute selected
  experts locally;
- transfer only activations/results and preserve exact stable aggregation;
- use P100 for DeepSeek only if a separate numeric/performance gate proves an
  appropriate FP16 path; do not assume its HBM bandwidth compensates for the
  missing packed instruction path;
- measure novel and settled routes separately.

## 7. Close production operations

- authenticated TLS/rate-limited edge while the service remains loopback;
- durable service supervision, log rotation, metric retention and alerts;
- bounded session-RAM admission based on actual parked bytes;
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
