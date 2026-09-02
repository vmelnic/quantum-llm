# Roadmap

Status: active dependency order, 2026-09-02. Completed results belong in
`benchmarks.md`; rejected ideas belong in `research-decisions.md`.

## Baseline to preserve

- one self-contained RTX 3090 host, one service task and one artifact-driven VM;
- six active aliases: Qwen, Qwen Flash, Mistral, Muse, Ornith and DeepSeek;
- declared FP4/NVFP4/BF16 formats, exact target KV semantics and exact MoE
  top-k/stable aggregation;
- current common routed-VRAM ceiling of 12 GiB; DeepSeek 13 GiB is not valid;
- `MODEL_ROOT` as the only host-dependent model root;
- default `xhigh` thinking with explicit request control;
- no external owner, family runner or model-specific service branch.

## 1. Close correctness and recovery

- fault-inject long-prefix cancellation, timeout, worker restart, KV exhaustion
  and retained-session rollback;
- make DeepSeek cancellation recover automatically or fail readiness and restart
  cleanly; never advertise stale health;
- extend official-reference behavior to representative stochastic sampling,
  Qwen image input, Muse text/tools and Ornith coding/tool histories;
- preserve one artifact-declared streaming parser from first delta through EOS
  and final structured tools.

Acceptance: explicit failure, exact committed-state behavior where advertised,
no unsupported DeepSeek session commands and no manual orphan cleanup.

## 2. Qwen3.8-27B maximum-context target

Target: 262,144 populated exact-F16 positions and approximately 15 useful
output tok/s in a real coding harness. Cold prefill, reused-prefix prefill,
saturated decode and complete tool-loop wall time are separate gates.

Current prerequisite fails:

```text
hot weights + exact 262K KV = 30,805,569,536 bytes
RTX 3090 physical           = 25,769,803,776 bytes
host-KV scan floor          = 1.284 s/scalar call
```

Do not add ordinary offload, another KV codec or another rejected proposer. A
new mechanism must first pass one quantitative prerequisite from
[Research decisions](research-decisions.md), then numerical parity, populated
prefill, saturated decode and the real Pi gate.

## 3. DeepSeek settled throughput

Novel routes near 1 tok/s are acceptable. The useful settled target remains
10-15 tok/s.

- first close cancellation/recovery;
- attribute current novel and settled storage/RAM/VRAM/CPU/GPU traffic;
- preserve demand priority, retention classes, exact CPU/GPU merge and 12 GiB
  common preflight;
- optimize only a measured critical-path byte or synchronization cost;
- require the real common chat result to keep or revert each mechanism.

Do not reintroduce layer-major prefill, row-width tuning or shared block-128
INT8 without the fail-fast gates in the decision ledger.

## 4. Qualify the additional artifacts

Qwen Flash:

- preserve exact QSA and exact top-10 routing;
- obtain operation-level official parity beyond the current QSA CUDA gate;
- treat routed MoE as the measured short-context bottleneck;
- require a representative route/transfer trace before another throughput edit;
- do not claim coding readiness until a real Pi task makes correct edits.

Mistral:

- preserve source-native NVFP4 and BF16 latent KV;
- improve only a measured TTFT/critical-path blocker;
- qualify coding behavior separately from valid tool transport;
- do not advertise auxiliary multimodal tensors as callable support.

Muse and Ornith:

- run populated-context capacity/latency gates at their declared geometry;
- qualify representative coding/tool histories;
- keep Muse vision auxiliary-only until processor, operation, numerical and
  real-image gates pass.

## 5. Production operations

- unify admission across retained KV/session bytes and routed RAM ownership;
- add durable supervision, bounded log/metric retention and actionable health;
- qualify immutable-artifact backup, rollback and failed-start recovery;
- qualify a trusted TLS/rate-limited edge;
- provide one release gate covering source quality, clean build/tests, required
  chats, selected Pi gates and final process/GPU cleanup.

## Completion

Qwen completes only with the populated exact-F16 262K real-harness target.
DeepSeek completes only when exact novel/settled traffic and recovery meet their
accepted rates. Additional models complete only at their stated numerical,
populated-context and harness gates. API compatibility, `hi`, configured limits
or microbenchmarks do not satisfy these conditions.
