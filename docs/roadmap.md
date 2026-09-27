# Roadmap

Status: active dependency order, 2026-09-27.

Measurements belong in [Benchmarks](benchmarks.md). Rejected mechanisms and
reopening gates belong in [Research decisions](research-decisions.md).

## Baseline to preserve

- one self-contained Windows host and one primary RTX 3090;
- one common service task and one artifact-driven VM;
- the six model IDs in `.pi/settings.json`;
- exact declared routing, top-k and aggregation;
- honest FP4, NVFP4, BF16, F16 and compact-KV labels;
- 12 GiB fixed routed-VRAM ceiling and 1 GiB emergency reserve;
- `MODEL_ROOT` as the only host-dependent model-storage root;
- request-selectable thinking with `xhigh` as the harness default;
- no external model-state owner or family-specific service launcher;
- P100 devices disabled as a performance dependency.

## 1. Long-request correctness and recovery

Diagnose the Pi call that stalled after 58,664 populated prompt tokens while
the GPU was idle. Add a deterministic recovery gate for the identified
boundary before rerunning the long benchmark.

Exercise cancellation, timeout, KV exhaustion, worker failure and restart with
a large prefix. Verify:

- bounded health and readiness;
- no orphaned process or GPU allocation;
- no partial assistant commit;
- exact retained-state rollback where the provider advertises it;
- one response parser from first delta through final tool extraction.

## 2. Operational Qwen long-context gate

Rerun the gradual Pi dialogue only after the stall boundary is understood.
Record every model call at its real populated prompt size. Completion requires:

- one prompt from 255,000 through 255,999 positions;
- cold and reused-prefix prefill reported separately;
- TTFT, generated/reasoning/visible tokens and tool wall time;
- Q4H and MTP state confirmed from provider telemetry;
- language, coding and long-range recall review;
- final process, port and GPU cleanup.

This gate qualifies the operational lossy path. It does not satisfy the
exact-F16 product goal.

## 3. Exact-F16 Qwen target

Target:

```text
262,144 populated positions
exact IEEE F16 target KV
about 15 useful output tok/s
real xhigh coding harness
one RTX 3090
```

The current capacity and host-scan inequalities fail. Do not implement another
offload, codec or proposer until it passes the reopening gate in
[Research decisions](research-decisions.md). Cold prefill, reused-prefix
prefill, saturated decode and complete tool-loop wall time are independent
acceptance results.

## 4. Fidelity diagnosis

The Romanian defect survived F16 target KV, scalar decode, sampling changes and
multiple weight/activation candidates. The next candidate must start from the
exact-token logit prerequisite in the decision ledger. Local reconstruction
error alone is insufficient.

Promotion requires:

- independent source/reference validation;
- finite independent QPack and source logits on the same exact prefix;
- a causal tensor-role selection result;
- the fixed Romanian service gate;
- at least 49.06 generated tok/s;
- no alias update until all gates pass.

## 5. Other model qualification

| Model | Next gate |
|---|---|
| Qwen Abliterated | representative behavioral and coding comparison against official Qwen |
| Qwen Flash | official operation parity, practical TTFT and representative coding |
| Mistral | isolate and reduce measured TTFT, then representative coding |
| Muse | populated 131K text plus tools; vision remains auxiliary |
| Ornith | publish the validated sampling policy, long-context quality and a complete coding fixture |

Minimal `hi` and valid tool transport do not complete these gates.

## 6. Production boundary

After model correctness and recovery:

- unify admission across active KV, parked sessions and routed caches;
- qualify competing sessions under the one-slot execution contract;
- add durable supervision and bounded log/metric retention;
- test artifact rollback and failed-start recovery;
- place TLS and public rate limiting at a trusted edge;
- provide one release command for clean build, tests, required chats, selected
  Pi gates and final cleanup.

## Completion

Qwen completes only at the populated exact-F16 real-harness target. Each other
model completes at its own numerical, populated-context and representative
harness gates.
