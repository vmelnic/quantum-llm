# Roadmap

Status: active dependency order, 2026-09-12. Measurements belong in
[Benchmarks](benchmarks.md); failed ideas belong in
[Research decisions](research-decisions.md).

## Baseline to preserve

- one self-contained Windows host, one primary RTX 3090, one service task and
  one artifact-driven VM;
- six artifacts and eight user-facing aliases from `ops/model-aliases.tsv`;
- exact declared routing/top-k/aggregation and honest FP4/NVFP4/BF16/F16 names;
- 12 GiB common fixed routed-VRAM ceiling; 13 GiB fails DeepSeek preflight;
- `MODEL_ROOT` as the only host-dependent model root;
- request-selectable thinking, default `xhigh`;
- no external model-state owner or family-specific service runner;
- installed P100s are optional capacity experiments, not a performance
  dependency or default.

## 1. Correctness and recovery

- fault-inject long-prefix cancellation, timeout, KV exhaustion and restart;
- make unhealthy DeepSeek cancellation fail readiness and restart cleanly;
- verify committed retained-state rollback for providers that advertise it;
- retain one artifact-declared streaming parser from first delta through final
  text/tool extraction;
- extend independent reference behavior to stochastic sampling, Qwen images,
  Muse tools and Ornith coding histories.

Acceptance: explicit failure states, no orphaned workers, no partial assistant
commit and no unsupported session commands.

## 2. Qwen maximum context

Target: 262,144 populated exact-F16 positions and about 15 useful output tok/s
in a real coding harness. Cold prefill, reused prefill, saturated decode and
tool-loop wall time are separate gates.

Current exact-F16 prerequisite fails:

```text
hot weights + exact 262K KV = 30,805,569,536 bytes
RTX 3090 physical           = 25,769,803,776 bytes
host-KV scan floor          = 1.284 s/scalar call
```

K1 fits and measured 14.39 tok/s after first token, but it is lossy. The
matched 262,016-token case took 546.173 s under K1 and 2,278.082 s under F16.
Do not add another codec until K1 receives a broader retrieval/reasoning/coding
quality decision. If K1 is accepted, optimize only the measured full-shape
dominants: 250.365 s attention, 178.069 s dense FFN and 116.299 s recurrent.

Do not reopen ordinary offload, rejected speculative proposers or P100 KV/
dense sharding without a new prerequisite that changes their failed math.

## 3. DeepSeek settled throughput

Novel routes near 1 tok/s are acceptable; settled target remains 10-15 tok/s.

- close cancellation/recovery first;
- measure novel and settled NVMe/RAM/VRAM traffic on the primary path;
- preserve demand priority, exact router/top-k/merge and 12 GiB preflight;
- optimize only the critical-path byte transfer or synchronization identified
  by request telemetry;
- keep a change only if real direct chat improves without breaking Qwen and
  cleanup.

The best recorded settled primary path is 6.54-6.69 tok/s. The P100 path
reached 3.18 tok/s and is closed as a speed direction; do not patch its Pascal
kernel again without a complete design that first proves <=100 ms/token.

## 4. Qualify callable artifacts

| Artifact | Next meaningful gate |
|---|---|
| Qwen Flash | official operation parity, then representative coding; P100 route is not an accelerator |
| Mistral | identify and reduce measured TTFT before another coding run |
| Muse | populated 131K text and representative tools; vision remains auxiliary |
| Ornith | long-context K1/F16 quality and correct coding completion without routed churn |

Minimal `hi` and valid tool transport do not satisfy these gates.

## 5. Production operations

- unify admission for active/retained KV, sessions and routed RAM ownership;
- add durable supervision, bounded logs/metrics and actionable readiness;
- verify immutable-artifact backup, rollback and failed-start recovery;
- put TLS and rate limiting in a trusted edge before public exposure;
- provide one release command for clean build/tests, required chats, selected
  Pi gates and final process/GPU cleanup.

## Completion

Qwen completes only at the populated exact-F16 real-harness target. DeepSeek
completes only when exact novel/settled traffic and recovery meet their accepted
rates. Other models complete at their numerical, populated-context and harness
gates—not at configured context, parser smoke or `hi`.
