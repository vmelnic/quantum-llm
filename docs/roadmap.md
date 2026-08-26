# Roadmap

Status: active dependency-ordered work, 2026-08-26. Completed measurements
belong in `benchmarks.md`; rejected directions belong in
`research-decisions.md`. This file contains only work that remains active.

## Baseline that every change must preserve

- one artifact-driven service/VM lifecycle for Qwen, Muse, Ornith and
  DeepSeek;
- Qwen, Muse and Ornith matrix/expert payloads execute through the declared
  FP4 ABI; DeepSeek keeps its separately authenticated compact physical ABI;
- exact F16 target KV for promoted dense/hybrid service paths;
- exact routed top-k and stable aggregation for sparse paths;
- the validated 12 GiB routed-VRAM profile on the single RTX 3090;
- `MODEL_ROOT` as the only host-dependent artifact root;
- default `xhigh` reasoning, with explicit per-request control;
- the current short-chat regression results recorded in `benchmarks.md`.

## 1. Close correctness and recovery qualification

This work precedes throughput claims because an optimization is irrelevant if
the public service cannot preserve exact request state and model behavior.

- extend official-reference comparisons to stochastic sampling and Qwen
  visual inputs;
- extend Muse's passing bounded prefix comparison to complete text/tool cases;
- qualify Ornith on representative coding/tool histories rather than only its
  current bounded Pi loop;
- fault-inject large-prefix cancellation, timeout, worker restart, KV
  exhaustion and retained-session rollback;
- make DeepSeek cancellation leave the common service healthy or restart the
  failed worker automatically without advertising stale readiness;
- keep the single artifact-declared streaming parser authoritative from first
  delta through finalized tools/EOS.

Acceptance: failures are explicit, committed prefixes survive exactly where
the provider advertises retention, non-retaining providers receive no session
commands, and repeated public requests need no manual process cleanup.

## 2. Qwen exact-F16 262K throughput research

The active target remains one self-contained RTX 3090, 262,144 actually
populated positions, exact F16 target KV and approximately 15 useful output
tok/s on the real Pi coding harness. Cold prefill, prefix reuse and saturated
decode are independent gates.

The current scalar-call lower bound is:

```text
hot target weights              13,625,700,352 bytes
exact F16 target KV             17,179,869,184 bytes
subtotal                        30,805,569,536 bytes
RTX 3090 physical capacity      25,769,803,776 bytes
measured pinned link                         12.46 GiB/s
host-KV transfer floor                        1.284 s/call
```

Ordinary offload, dense layer streaming and another wrapper cannot satisfy
the target. Before runtime infrastructure is added, a new mechanism must prove
one of these prerequisites on real Qwen state:

- enough additional compute-local high-bandwidth capacity for exact weights,
  KV, recurrent/draft state, workspaces and reserve; or
- lossless residency below the practical bit threshold recorded in
  `research-decisions.md`; or
- at least 20 useful exact accepted tokens per amortized full-KV traffic unit,
  before verifier compute, with all proposer/verifier state included.

Any proposal that fails its prerequisite is recorded and stopped before a
maximum-context service run. A passing prerequisite must then pass, in order:
bounded numerical parity, complete populated prefill, saturated decode and the
real Pi tool-loop gate.

## 3. DeepSeek novel and settled throughput

The exact model may exceed RAM+VRAM and use local NVMe. Approximately 1 tok/s
on novel routes is acceptable; the useful settled target remains 10--15
tok/s. Measurements must report storage, RAM, VRAM, CPU/GPU decisions, reread
bytes and H2D bytes separately.

- first close the cancellation/recovery defect from section 1;
- preserve demand priority, bounded speculative staging, protected/transient
  cache classes and exact CPU/GPU merge;
- attribute why the current short gate is near 1.38 tok/s after first visible
  while historical settled prompts reached 6.54--6.69 tok/s;
- optimize only a confirmed remaining critical-path byte or synchronization
  cost; keep or revert each mechanism by the real common-chat result;
- do not raise routed VRAM above 12 GiB without Qwen and DeepSeek preflight,
  chat and cleanup gates.

FreeToken-derived device cache control, safe-point expert/KV rebalance,
post-population pinning and full-layer prefill double buffering remain research
candidates, not implemented claims. Each requires its quantitative gate from
`research-decisions.md` before code is added.

## 4. Complete Muse and Ornith service qualification

Muse:

- preserve the pinned `meta-models/Muse-Glimmer-30B` source and published
  `muse-glimmer-30b-fp4` artifact;
- keep exact artifact-declared global/sliding F16 KV and the common FP4
  provider;
- run a populated-context gate before making any 131K latency claim;
- keep vision auxiliary-only until a generic vision operation passes source,
  numerical, API and real-image gates.

Ornith:

- preserve the pinned `ornith-ai/Ornith-1.5-35B-A3B` source, standard FP4
  expert pages and generic routed provider;
- qualify long-context capacity/traffic and representative coding quality;
- do not add a family runner, task, fixed layer count or deployment branch.

## 5. Close production operations

- enforce one admission budget across parked KV/session bytes and retained
  expert RAM;
- add durable supervision, bounded log/metric retention and actionable health
  reporting;
- qualify authenticated edge deployment with TLS and rate limiting while the
  native service remains self-contained;
- document and test immutable-artifact backup, transactional rollback and
  failed-start recovery;
- define one release command/gate covering source quality, build/tests, all
  four public chat smokes, selected harness gates and final GPU/process cleanup.

## Optional hardware path

Multi-GPU organ placement is not active on the current one-RTX-3090 host. It
becomes active only after compatible hardware exists. Then compute must follow
state: KV is sharded beside attention compute and expert pages beside expert
compute; no design may depend on a remote owner or treat PCIe transfer as
VRAM bandwidth.

## Completion conditions

Qwen is complete only when the real service populates 262,144 exact-F16
positions and the real coding harness sustains approximately 15 useful output
tok/s with acceptable cold/reused-prefix behavior. DeepSeek is complete only
when exact novel and settled routes meet their accepted targets with measured
tier traffic and recovery. Muse and Ornith are complete only at their stated
source, numerical, populated-context and harness gates. API acceptance, a
metadata smoke or a microkernel ceiling satisfies none of these conditions.
