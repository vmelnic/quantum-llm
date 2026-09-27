# Research decisions

Status: active decision ledger, 2026-09-27.

This file records durable mechanism decisions. Measurements belong in
[Benchmarks](benchmarks.md). Reopen a rejected direction only when new evidence
changes its failed capacity, traffic, latency, acceptance or fidelity bound.

## Fixed Qwen target

```text
model       Qwen3.8-27B
hardware    one RTX 3090
context     262,144 populated positions
KV          exact IEEE F16
harness     real coding agent, xhigh
throughput  about 15 useful output tok/s
```

The current exact-F16 inequality is negative:

```text
hot target weights             13,625,700,352 bytes
exact F16 target KV            17,179,869,184 bytes
subtotal                       30,805,569,536 bytes
RTX 3090 physical              25,769,803,776 bytes
pinned PCIe link                    12.46 GiB/s
full host-KV scan floor              1.284 s/call
```

A proposal must first prove enough compute-local capacity or enough exact
accepted positions per complete KV/weight traffic unit. Moving hot state to
RAM or NVMe is a capacity mechanism and cannot meet this throughput target.

## Accepted mechanisms

| Mechanism | Accepted boundary |
|---|---|
| artifact-driven VM | source adapters own upstream names; common runtime binds declared capabilities and ABIs |
| progressive exact-F16 pages and mirror | preserves exact values for prefixes that fit; host staging remains slow at maximum context |
| exact sparse-expert paging | immutable selected pages may move among NVMe, RAM and VRAM without dropping top-k |
| lazy dense and logits workspaces | allocate only requested rows; released about 994 MiB in the measured Qwen path |
| exact hierarchical top-k | preserves values, token IDs, tie order and NaN exclusion; reduced worker sampling to 0.258 ms/call |
| grouped standard-FP4 routed prefill | groups rows by selected expert while preserving route and aggregation |
| bounded layer-major Qwen Flash prefill | valid only when complete declared streams fit with the 1 GiB reserve |
| exact-decode ABI 2 / MTP-4 | Q8 proposer state, batch-five target verification and exact sampled rejection |
| persistent Qwen continuation | exact-prefix RAM parking plus transactional lazy NVMe restart/restore |
| Q4H operational policy | explicitly approved lossy default after live 32K and 128K speed gates; separate from the exact-F16 target |

## Q4H boundary

Q4H stores one F16 scale for each complete 256-value signed-Q4 K or V head.
The target cache is 4.0625 GiB at 262,144 positions; Q8 MTP state raises it to
4.566406 GiB. Packed attention reads this representation directly.

The independent layout and attention gates passed, and live decode reached
54.54 tok/s at 32K and 47.16 tok/s at 128K. This admits Q4H as an operational
performance policy. It does not establish F16 fidelity, a live 262K result or
reliable long-agent behavior.

K1 and the block-floating Q4 variants remain implemented diagnostics. No
active primary alias selects them.

## Exact-F16 reopening gate

Resident F16 attention measured 91.832 ms across the 16 full-attention layers
at 262K before target matrices, proposer, head or sampling. The older 40 tok/s
packed-KV experiment allowed 70.56 ms for a complete cycle at its measured
acceptance and therefore did not admit this kernel. The product target is
about 15 useful tok/s and requires its own complete-cycle calculation.

New exact-F16 work must satisfy both:

1. the complete resident capacity including weights, recurrent state, draft
   state, workspaces and 1 GiB reserve;
2. a measured or bounded complete-cycle cost below 66.667 ms per useful token
   for the actual accepted positions per call.

A codec ratio measured only at rest is insufficient.

## Rejected general mechanisms

| Direction | Failed prerequisite |
|---|---|
| FP8 target KV | wrong fidelity; 262K prefill 1,168.594 s, decode 3.064 tok/s and factual failure |
| Q4 target KV as exact result | lossy by construction and material quality degradation observed |
| dense layer streaming | 12.690 GiB hot weights / 12.46 GiB/s gives at least 1.018 s/call |
| static lossless weight+KV coding | ideal estimate still exceeded VRAM by 1,123,844,072 bytes before runtime state |
| reversible F16 transforms | 13.265926 bits/value measured; 11.617612 required |
| Huffman/Zstd/zlib/LZ4 | best held-out Huffman result was 13.719539 bits/value |
| exact offloaded MTP tree | optimistic ceiling 2.8 tok/s |
| block-Jacobi proposer | corrected ceiling 1.55 tok/s |
| prompt n-gram/copy proposer | 3.8485 useful positions/cycle; 24 required |
| exact page refinement | 2,029.014 MiB/call; 272.822 MiB budget |
| staged single-GPU exact attention | 91.763 ms for attention before other work |
| CPU F16 V-tail | 226.279 ms/call |
| all-K-resident/selective-V | about 25,922 MiB before V staging |
| P100 exact attention | 77.101 ms compute-only; required at most 32 ms |
| P100 compact dense shard | 35.939 ms/card before integration; required at most 20 ms |
| P100 lossless F16 MLP shards | 307.7/311.0 GB/s; required at least 500 GB/s/card |
| P100 Qwen Flash experts | 0.20 tok/s versus 0.37 tok/s on the primary path |

The P100 executor may remain as an optional exact capacity tier for compatible
routed pages. It is not a performance dependency.

## MTP decisions

The original sampled MTP-4 implementation reran target work after speculative
rejection and reached only 15.67 tok/s at 32K and 8.59 tok/s at 128K. It was
reopened after profiling proved the redundant target execution.

The accepted implementation exports recurrent checkpoints for every verified
row and restores the selected row without rereading target weights or KV.
Historical K1 measurements improved to 26.19 tok/s at 32K and 23.89 tok/s at
128K. Direct packed Q4H then moved attention and the complete cycle to the
current 54.54/47.16 tok/s results.

Acceptance alone is not a tuning target. A speculative proposal must account
for proposer, verifier, target matrix, attention, head and sampling cost, then
report accepted positions per complete target call.

## Long-agent sampling

Qwen Abliterated entered lexical loops at 133,610 and 148,441 prompt tokens.
The first mitigation, `presence_penalty=1.5`, produced visible Romanian
lexical and grammatical degradation and is rejected.

Current Qwen artifacts use `presence_penalty=0.5` and a 32,768-token thinking
circuit breaker. This is an operational bound, not a fidelity fix. A coherent
23,599-token tool-producing turn showed that a 16K ceiling would reject valid
observed work.

## Romanian fidelity diagnosis

A controlled matrix showed malformed Romanian with:

- official and Abliterated weights;
- Q4H and F16 target KV;
- MTP-4 and scalar target decode;
- presence penalties 0.5 and 0.0;
- temperatures 1.0 and 0.6.

Do not continue KV, MTP or sampler tuning as the primary diagnosis.

The following weight/activation candidates also failed:

| Candidate | Useful result | Failure |
|---|---|---|
| selective embedding/head MXFP6 | relative L2 0.05634 | 39.13 tok/s and malformed Romanian |
| source-BF16 activation rounding | exact artifact capability | 45.78 tok/s and malformed Romanian |
| hybrid block-32 Q8 activations | 19.26x local activation-SSE reduction | 41.97 tok/s and malformed Romanian |
| activation-aware scale selection | holdout SSE improved 24.12% | 51.32 tok/s; Romanian still failed |
| adjacent FP4 codes | holdout SSE improved 41.93% | 50.48 tok/s; Romanian still failed |
| recurrent-A MXFP6 | third-holdout SSE improved 62.84% | 49.47 tok/s; Romanian still failed |
| BF16 dense operands | local kernel gate passed | live upper bound below 48.55 tok/s |
| exact recurrent convolutions | removed 100% of selected source error | 49.70 tok/s; Romanian still failed |

Local tensor or projection residual has repeatedly failed to predict language
quality. Another mixed-precision or weight change requires a validated
logit-directed prerequisite on an exact service token prefix.

## Rejected offline quantizers

| Candidate | Result | Decision |
|---|---|---|
| exhaustive UE8M0 scale search | 0% beyond MSE-v2 | current sampled MSE-v2 scales already optimal for that objective |
| RMSNorm diagonal equalization | all 128 groups selected identity | no candidate artifact |
| unrestricted FP4 coordinate correction | holdout regressed 25.08% | overfit |
| guarded coordinate correction | holdout +0.0127%; 9/12 role regressions | failed gate |
| sparse F16 residual, K=8 | holdout +21.40%; required 25% | failed byte/quality trade |
| matrix-specific Q4 codebook | holdout +0.6114%; one role regression | failed gate |
| block-Hessian FP4 GPTQ | holdout -130.53%; 12/12 regressions | broad overfit |
| one-byte U6M2 scale | holdout +29.97%, source SSE only +8.44%; required 30% source | failed source-fidelity gate |

Do not tune the same search bounds on the same captures. A future GPTQ-style
attempt requires a materially larger corpus and a full-network/logit gate.

## Current fidelity prerequisite

The next diagnosis must:

1. retain exact service token IDs and the effective target sampling
   distribution at a known malformed continuation;
2. produce finite, validated source-BF16 and independent QPack logits on the
   same exact prefix without exceeding host memory;
3. separate stored-weight error from provider arithmetic;
4. rank candidate tensor roles from that logit difference;
5. pass an independent source-quality gate;
6. retain at least 49.06 generated tok/s on the fixed live request;
7. pass the Romanian service gate before promotion.

An earlier gradient/sensitivity analyzer emitted NaNs and values near
`1e77` while incorrectly marking itself passed. Its ranking is invalid and
must not select a candidate.

## Proposal gate

Before runtime implementation, record the exact model, populated context,
formats, hardware and harness, then calculate:

```text
weights + recurrent state + exact/packed KV + draft state + workspaces + reserve
bytes touched per target call and per accepted token
GPU, RAM, PCIe and NVMe lower bounds
speculative acceptance and complete proposer/verifier cost
```

Stop before implementation when the required compression, traffic, latency,
acceptance or fidelity inequality fails.
