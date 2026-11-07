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

## External coding-model screening, 2026-10-02

Scope: one RTX 3090, Pi on real repositories, 262,144 populated positions,
default xhigh, user-authorized alternative models and weight quantization.
F16 KV remains the capacity reference; weight-quantization permission does
not establish approval or quality for a new KV codec. Nemotron was explicitly
removed from the shortlist by the user. No model download, build, service
operation or local model benchmark was performed for this screening.
The later user clarification recorded below supersedes this screening's F16
selection filter and permits considering lossy weight/KV formats and small
models. Historical exact-F16 calculations remain evidence, not a selection
requirement for the clarified request.

### Coding evidence

Scores below are external measurements, not local qualification. Compare
models within the same evaluation pipeline; do not rank percentages from
different harnesses as though they were controlled comparisons.

| Candidate | Published coding evidence | Evidence boundary |
|---|---|---|
| Qwen3.8-27B GSQ-RCO | LiveCodeBench v6 IQ3_S 85.71 vs BF16 85.71; IQ3_XXS 84.57 | controlled quantization comparison; no published full SWE-bench or populated-256K Pi/3090 gate identified |
| Qwen3.6-35B-A3B | official SWE-bench Verified 73.4, Pro 49.5 | original checkpoint and upstream harness, not our quantized Pi path |
| KAT-Coder-V2.5-Dev | Verified 69.40 vs base Qwen3.6 64.40; Pro 45.96 vs 40.63; Terminal-Bench 2.1 average 41.02 vs 34.84 | Kwaipilot re-evaluated public checkpoints under one pipeline; no local QPack result |
| Occamy-1.0 | Terminal-Bench 2.1 59.00 vs Qwen3.6 49.50 | publisher evaluation; no full SWE-bench repository score identified in its card |
| Tiel-Coder | 12/25 SWE-bench-Live issues vs stock Qwen3.6 8/25 and Qwen3.8 16/25 | publisher's small single-run screen; tested UD-Q4_K_XL, not the smaller IQ4_XS tier |
| Bonsai 2 27B | HumanEval+ 95.12, MBPP+ 83.07, LiveCodeBench 90.07 | Prism's thinking-mode EvalScope/vLLM H100 evaluation; current card has no SWE-bench score |

Sources: [GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF),
[Qwen](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
[KAT](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev),
[Occamy](https://huggingface.co/Accio-Lab/occamy-1.0),
[Tiel](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF),
[Bonsai](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

Occamy additionally publishes a bounded quantization screen on B200:
IQ4_XS and BF16 each passed 23/24 HumanEval+ cases, and 6/6 native-tool
cases. This is neither a full coding-agent gate nor a 3090 performance result;
the report also documents tokenizer metadata corrections after measurement.
See its [validation report](https://huggingface.co/Accio-Lab/occamy-1.0-GGUF/blob/main/VALIDATION.md).

### Corrected speed and context claims

Giles Thomas measured Qwen3.6-35B-A3B UD-IQ4_NL_XL at about 140 generated
tok/s fully on a 3090, and about 89 tok/s with ten FFN layers on CPU. The
configured windows were 89,600 and 262,144 respectively, but the benchmark
populated only 2,457 prompt tokens and generated up to 6,144. These numbers
are not 256K-populated throughput or coding results.
[Original experiment](https://www.gilesthomas.com/2026/07/benchmarking-qwen-3-6-35b-moe-rtx-3090).

An independent Bonsai PTQ1_0 comparison reports median decode 60.77 vs
44.17 tok/s for Qwen3.8 Q4, and 3/7 vs 7/7 fixed probes. Its raw results
and summary explicitly show short prompts with 131K/252K configured windows,
not populated long-context retrieval. The coding probe also had a 256-token
generation cap, and the commands used raw completion/no chat conversation.
This is a bounded negative screening signal, not a definitive ternary coding
quality result. Its F16 262K boundary allocated about 22,270 MB on the GPU;
that is allocation evidence, not a populated-context latency/quality gate.
[Summary and raw artifacts](https://github.com/ippocra/bonsai-qwen-rtx3090-benchmark/blob/main/data/runs/20260918-bonsai2-27b-vs-qwen38-27b/SUMMARY.md).

### Capacity and traffic prerequisites

For Qwen3.6, KAT and Occamy text geometry, and the inherited Tiel backbone:

```text
C = 262,144; full-attention layers = 10; KV heads = 2; head width = 256
F16 KV = 2 * C * 10 * 2 * 256 * 2 = 5 GiB
core recurrent state = 30 * 32 * 128 * 128 * 4 = 60 MiB
VRAM = routed weights + dense/shared matrices + embeddings/head + routers
       + norms + recurrent/conv state + KV + draft state + workspaces + reserve
```

The following are storage-based subtotals for text-only scalar decode with
no draft, 5 GiB F16 KV, core recurrent state and the mandatory 1 GiB reserve.
They exclude convolution state, additional runtime allocations and workspaces;
file bytes are a planning proxy, not measured runtime residency.

| Artifact | File GiB | Subtotal GiB | Remaining below 24 GiB |
|---|---:|---:|---:|
| KAT IQ4_XS | 17.515 | 23.573 | 0.427 |
| KAT Q4_K_M | 19.922 | 25.981 | negative |
| Tiel UD-IQ4_XS, no MTP | 16.513 | 22.571 | 1.429 |
| Tiel benchmarked UD-Q4_K_XL, no MTP | 20.825 | 26.883 | negative |

Exact file sizes were read from publisher repository metadata:
[KAT files](https://huggingface.co/bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF/tree/main),
[Tiel files](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF/tree/main).
Native mathematical geometry comes from the upstream config files. Passing
the smaller tier's capacity bound cannot inherit the larger tier's quality.

For standard block-32 FP4 QPack, routed weights alone would require:

```text
40 * 256 * 3 * 2048 * 512 * (0.5 + 1/32) = 15.9375 GiB
selected routed bytes / scalar target call = (8/256) * 15.9375 = 0.49805 GiB
bytes / scalar token >= hot dense/shared/head/router bytes + 0.49805 GiB
                       + 5 GiB KV + recurrent traffic + intermediates
```

This exceeds the current 12 GiB expert-cache ceiling even before counting
non-routed state. A QPack conversion alone cannot establish full residency
or faster prefill. Do not raise the ceiling based on this arithmetic: the
existing real-chat and cleanup requirements still apply.

For any candidate, the bandwidth prerequisite is:

```text
t_call >= max(B_GPU / BW_GPU, B_RAM / BW_RAM,
              B_PCIe / 12.46 GiB/s, B_NVMe / BW_NVMe)
t_useful = t_complete_cycle / accepted_positions_per_cycle
```

The 3090's theoretical GPU bandwidth is about 936 GB/s; sustained kernel,
RAM and NVMe bandwidth for these candidates remains unmeasured. With scalar
decode, accepted positions/call = 1 and proposer cost/state = 0. Scanning
5 GiB KV through the measured PCIe link alone would cost at least 401 ms;
resident GPU attention is therefore a prerequisite for speed. Speculation
requires independent acceptance, proposer, verifier and memory measurements;
do not import another runtime's MTP or DFlash speedup. KAT and the main
Occamy checkpoint declare zero MTP layers; Tiel has a separate MTP release.

GSQ-RCO retains Qwen3.8's dense 27B geometry and 16 GiB F16 KV at 262K.
GSQ learns scalar quantization assignments/scales; RCO chooses precision per
tensor under a file-size budget. The output is standard mixed-precision GGUF,
not ternary weights. Exact repository metadata gives IQ3_S 11,771,546,784
bytes (10.963 GiB), IQ3_XXS 10,094,357,632 bytes (9.401 GiB), and IQ2_XS
8,422,841,472 bytes (7.844 GiB). Thus even IQ2_XS + 16 GiB F16 KV +
1 GiB reserve exceeds 24 GiB before recurrent state and workspaces. Stop
the fully resident exact-F16 proposal at this failed capacity prerequisite.

Conditional Q8_0 KV (34 bytes per 32 values) would occupy 8.5 GiB and
give an IQ3_S + KV + reserve subtotal of 20.463 GiB. The currently approved
Q4H geometry instead gives 16.026 GiB. These are capacity estimates, not
codec interoperability, quality or performance results; no new KV fidelity
change was approved or implemented. Optional IQ3_S MTP weights add about
0.325 GiB plus separately budgeted draft KV/state/workspaces. Scalar traffic
still includes all hot dense weights and retained full-attention KV per call;
no primary 3090 populated-256K throughput result was identified. MTP needs
measured accepted positions/cycle and full proposer/verifier cost before a
speed claim. RAM/NVMe/PCIe offload of these hot tensors cannot establish fast
decode. The GGUF IQ scalar codebooks and per-tensor allocation must survive
any QPack integration: re-quantizing them into existing FP4 does not inherit
the published quality result. "Task-lossless" means matching these benchmark
scores, not exact numerical fidelity or proven coding-agent equivalence.
[GSQ-RCO release and reproducibility artifacts](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF),
[GSQ implementation](https://github.com/IST-DASLab/GSQ).

Bonsai retains the dense 27B attention geometry: F16 KV at 262K is 16 GiB.
Its PTQ1_0 file is about 5.54 GiB, giving at least 22.54 GiB with the 1 GiB
reserve before recurrent state/workspaces. PQ2_0 is about 6.71 GiB and leaves
less headroom. Dense projection weights and all retained KV are hot on each
scalar call. Bonsai also requires a new ternary encoding and matching
Hadamard activation operation; ordinary FP4 repacking cannot inherit its
file size or numerical contract. The publisher requires its custom runtime.

Other screens: Qwen3-Coder-30B needs 24 GiB F16 KV alone at 262K;
Devstral Small 2 needs 40 GiB; Laguna XS.2 needs 10 GiB global F16 KV
plus sliding-window state. These are poor exact-F16/full-residency choices
with their 4-bit weights on 24 GiB. GLM-4.7-Flash declares only 202,752
positions. LFM2-24B-A2B declares 32K and explicitly discourages coding;
LFM2.5-8B-A1B advertises about 128K and discourages heavy programming.
[Laguna config](https://huggingface.co/poolside/Laguna-XS.2/blob/main/config.json),
[GLM config](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json),
[LFM2](https://huggingface.co/LiquidAI/LFM2-24B-A2B),
[LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B).

Next qualification order: GSQ-RCO IQ3_S is the first controlled weight-quality
candidate for retaining Qwen3.8, conditional on an explicitly qualified KV
format; KAT and Qwen3.6 are the first MoE repository-coding/speed candidates.
Occamy/Tiel remain explicitly bounded alternatives. First
resolve exact artifact residency and source fidelity; then native numerical
quality, direct chat, streaming/tools and the real Pi task at actual populated
context. Bonsai is a capacity candidate with unresolved agent quality. No
candidate in this screening has proved fast, correct Pi coding at 256K
populated tokens on our single 3090.

### User clarification: coding quality and useful speed, any model/precision

The user explicitly rejected using F16 preservation as the selection objective
and requested coding quality and speed even with 4B models, Q4/Q2 weights and
quantized KV. Research invariants now are one RTX 3090, real Pi coding,
262,144 populated positions, default xhigh, any useful model size and supported
precision, with coding task outcomes and elapsed execution time as the quality
and speed criteria. This is research authorization, not a service deployment.
No runtime/provider edits, downloads or new local inference tests were made.

Primary-source additions:

| Candidate | Coding evidence | Speed/evidence boundary |
|---|---|---|
| Qwen3.5-4B | official LiveCodeBench v6 55.8, BFCL-v4 50.3 | submitted 3090 UD-Q4_K_XL run: 138 decode tok/s, 4,899 prefill tok/s at p512/tg128, not long-context Pi |
| Qwen3.5-9B | official LiveCodeBench v6 65.6, BFCL-v4 66.1; Comet paper's controlled baseline resolves 178/731 SWE-bench Pro tasks (24.35%) | no verified single-3090 populated-256K coding speed identified in this screen |
| Qwopus3.5-4B-Coder | author BugFind 71 vs base 52; suite average 82 vs 74 | small benchlocal suite, up to three answer attempts, Apple setup; no full SWE-bench qualification |
| Swift 1.5 Qwen3.8-27B | publisher's same-protocol BF16 xhigh evaluation: LiveCodeBench 81.71 vs base 76.76; Terminal-Bench 2.1 72.13 vs 69.21 | five seeds/trials; coding mean token reduction 24.5%, Terminal mean 16.3%; GSQ files only have 512-context KLD screens, not these task scores |
| Comet-9B | 223/731 SWE-bench Pro tasks (30.51%) vs base 178/731 (24.35%) in the same mini-SWE-agent pipeline | paper published 2026-09-28; no identifiable released checkpoint/runtime speed found, so not a deployable recommendation |

[Qwen 4B](https://huggingface.co/Qwen/Qwen3.5-4B),
[Qwen 9B](https://huggingface.co/Qwen/Qwen3.5-9B),
[3090 4B run and reproduction](https://llamabench.ai/r/952),
[Qwopus 4B](https://huggingface.co/Jackrong/Qwopus3.5-4B-Coder-MTP-GGUF),
[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b),
[Swift GSQ](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF),
[Comet paper](https://arxiv.org/html/2609.34359v1).

An independent single-card matched Q4_K_M comparison of Swift 1.0 against
base Qwen3.8 reports 66.7% vs 69.4% on 36 Aider exercises, while summed
per-exercise elapsed execution drops 43% and median reasoning/request drops
57%. This supports investigating reasoning-efficient weights. It does not
validate Swift 1.5, its GSQ conversion, a full repository agent or 256K.
[Matched experiment](https://alexander-ollman.github.io/qwen3.8-on-rtx3090/results.html).

GSQ's controlled precision ladder is LiveCodeBench v6 85.71 (IQ3_S), 84.57
(IQ3_XXS), 82.29 (IQ2_S), 76.57 (IQ2_XS), against BF16 85.71. Low-bit
formats are eligible; smaller bytes alone do not prove faster execution.
The Swift 1.5 GSQ release reuses ISTA allocations with Swift-specific
refinement and reports held-out KLD at context 512, explicitly not coding-task
accuracy or long-context quality. Do not transfer base GSQ's coding scores to
Swift GSQ or use a demonstration's 9.18x gain as a universal speedup.

Local Model Benchmarks publishes additional quantized coding screens on one
RTX 6000. At Q4_K_M with thinking off, Qwen3.5-4B scores 76.83% HumanEval+
and 49% BigCodeBench; Qwopus 4B scores 66.46% and 43%; LFM2.5-8B-A1B
scores 73.78% and 34%. These data do not establish a Qwopus improvement or
justify preferring LFM for coding. They are not 3090 speed or populated-context
results. Source snapshot 2026-09-09, configured context 65,536.
[Dashboard](https://localmodelbenchmarks.com/),
[Raw measurements](https://localmodelbenchmarks.com/data.json).

Revised capacity prerequisites use supported KV block encodings rather than
rejecting candidates for failing the superseded F16 filter:

```text
C = 262,144
B_KV_Q8_0 = number_of_K_and_V_elements * 34/32
B_KV_Q4_0 = number_of_K_and_V_elements * 18/32
Qwen3.5 4B/9B: 8 full-attention layers, 4 KV heads, head width 256
  Q8_0 KV = 4.25 GiB; Q4_0 KV = 2.25 GiB
  core recurrent state = 24 * 32 * 128 * 128 * 4 = 48 MiB
KAT/Qwen3.6: Q8_0 KV = 2.65625 GiB; Q4_0 KV = 1.40625 GiB
Qwen3.8/Swift/Bonsai: Q8_0 KV = 8.5 GiB; Q4_0 KV = 4.5 GiB
residency = weights + recurrent/conv + KV + draft + workspace + reserve
```

KAT IQ4_XS + Q4_0 KV + core recurrent + 1 GiB reserve is approximately
19.980 GiB before conv state/workspaces. GSQ IQ3_S + Q8_0 KV + 1 GiB
reserve is 20.463 GiB before recurrent/draft/workspace; GSQ IQ3_XXS gives
18.901 GiB. Codec provider capability, actual residency and populated-context
quality remain separate gates. No CPU/NVMe offload is needed by these
subtotals, and no hot-weight offload speed claim is made.

For scalar decode, all dense weights and retained full-attention KV remain
hot; unrouted MoE experts remain cold. The bandwidth and speculation equations
above still apply, now with the selected quantized KV bytes. Baseline draft
cost/state = 0 and accepted positions/call = 1; MTP must measure acceptance
and verifier/proposer cost before recommendation as a speed improvement.
The existing 12 GiB routed QPack budget remains unchanged. New numerical
operations or IQ encodings require capabilities, not model-family runners.

Revised shortlist: KAT/Qwen3.6 MoE with low-bit weights and quantized KV for
repository coding and speed; Swift 1.5 and base GSQ IQ3_S/IQ3_XXS for a
reasoning-efficient/compact Qwen alternative; Qwen3.5-9B and 4B as smaller
quality/speed baselines; Bonsai as a low-memory candidate. No small-model,
ternary, Q2 or MTP candidate is excluded by precision alone. Acceptance is
successful real coding tasks per elapsed execution time at the actual
populated context, with independent quality validation before promotion.

### Selected candidates and host preparation

The user selected this order: Swift 1.5 GSQ-RCO IQ3_XXS first, then
KAT-Coder-V2.5-Dev IQ4_XS. Standalone Qwen3.5 models are no longer candidates.
The user requested stopping further benchmark research and proceeding with
practical preparation. No benchmark suite was started.

Hardware correction: the A520M/RTX 3090 machine and X99E/three Tesla P100
16 GiB machine are independent rigs. Their VRAM is not pooled. A read-only
inventory of the configured 3090 host confirms Ryzen 5 5600, 63.93 GiB RAM,
one RTX 3090 24 GiB, and the existing llama.cpp 0.5.0-dev binary (7fe450e).
The P100 sizes/topology are user-provided, not newly measured; no serving
dependency on the P100 rig was introduced.

The selected Swift repository is public and ungated according to current Hub
metadata, despite stale private-access wording in its card. Identity:

```text
repository: ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF
revision: d74895bbe5db4bec1e0024e7cc87d59c02d7631a
file: Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf
bytes: 10,094,357,792
sha256: 86969b8bde72e602bfb42deb83eb8bb3706c8f14250641f6444dd2355f934ac2
optional MTP file: 10,442,827,776 bytes; not independently evaluated
```

The published allocation contains 851 tensors and 12 encodings, including
IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_XS, Q2_K,
Q4_K, BF16 and F32. The IQ3_XXS tier name is a size profile, not a uniform
tensor encoding. Existing QPack encoding ABIs do not preserve that allocation.
The unmodified GGUF is the selected reference artifact; no FP4 re-quantization
or new native provider was implemented.

Source geometry confirms 64 layers, 48 recurrent and 16 full-attention layers,
4 KV heads of width 256, native 262,144 positions, and one optional MTP layer.
Core recurrent state is 144 MiB. Without MTP, the file-size planning subtotal
with Q8_0 KV and 1 GiB reserve is 19.042 GiB; with Q4_0 KV it is 15.042 GiB,
before convolution state and workspaces. This does not impose F16 fidelity.
Native template supports xhigh/medium/low and structured tool calls.

At 2026-10-02 10:36 UTC, the canonical Start-HuggingFaceModelDownload.ps1
started the pinned exact-file Xet transfer under MODEL_ROOT/.hf-cache on the
3090 host. On the next check, the worker had exited successfully, the full
10,094,357,792-byte snapshot existed, and an independent Windows Get-FileHash
matched the pinned SHA256 above. Get-HuggingFaceModelDownload.ps1 still
reported completed=false because the cache lacked its expected tree metadata;
this was a status-validation limitation, not a failed transfer. No partial file
was promoted and Pi was not pointed at it.

At 2026-10-02 10:58 UTC, the same canonical Xet downloader started the second
user-selected artifact on the independent 3090 host:

```text
repository: bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF
revision: d8f684f08d2950ea9d2db6a35ef7dada0707858b
file: Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
bytes: 18,806,446,496
sha256: e35e23219a81590b9d4174eea4717d716dd62676c8c434f6b708f49a07310e1a
```

The download operation stored the published GGUF unchanged under
MODEL_ROOT/.hf-cache; it did not deploy a runtime, change Pi, or impose an
F16 fidelity requirement. On the subsequent user-requested serving setup,
the full KAT size, GGUF v3 header and SHA256 above passed the independent
Windows publisher check. The downloader's completed=false status again came
from missing cache tree metadata, despite successful worker exit and a valid
complete file.

The user explicitly requested a separate llama.cpp operations script to
preserve QPack behavior. Swift and KAT now have transactional GGUF publication
and a generic llama.cpp task through ops/llama.sh; ops/model.sh, its aliases,
ops/pi.sh, QPack code and existing Pi providers/defaults were not changed.
The llama-3090 provider was added to local user models.json and user/project
enabledModels. Both models passed local Pi streaming reasoning, a real read
tool execution, tool-result continuation and an exact final marker. Swift's
first tool probe used 965 populated prompt tokens; KAT's used 927. These are
integration probes, not repository-coding or populated-256K quality/speed gates.

Both startup contracts reported one slot and configured context 262,144,
with full GPU layer placement requested and automatic fit disabled. Observed
total GPU usage was 15,800 MiB for Swift and 20,508 MiB for KAT; idle was about
426-428 MiB. The selected GGUF weights remain unchanged; target K/V is Q4_0,
and there is no draft/MTP model. Native llama.cpp code/binary was not modified.

The initial stop implementation terminated the server but then queried CIM
identity again during termination and falsely reported an identity change.
A retry-only patch did not resolve the canonical stop gate. The revised
lifecycle validates identity before stopping, captures a process handle and
waits for that process to exit. The canonical start/stop/status gate then
passed, and GPU usage returned to 428 MiB. Detailed commands and boundaries
are in docs/llama-cli.md. Reasoning blocks were parsed by Pi, but usage reported
zero reasoning tokens, so those counters are not an independent reasoning-
token measurement.

### Swift live-use speed question: research only

The user reports roughly 16 GB VRAM and 30 tok/s in the active local Pi
session and explicitly forbids reading logs. No logs, active-process telemetry,
requests, restarts or runtime configuration changes were performed for this
screen. The reported rate is not a controlled decode measurement; actual
populated context and Pi's rate accounting are unknown.

Invariants for a prospective candidate: same Swift 1.5 GSQ-RCO IQ3_XXS
backbone, published matching integrated-MTP package, 262,144 configured
positions, Q4_0 target K/V, one single-GPU Windows llama.cpp service and Pi
coding/tool transport. This is a llama.cpp-specific operations candidate, not
a universal VM claim. Its numerical/tool-loop quality is unqualified.

The publisher states that the MTP package retains the matching main-model
tensors and adds 15 head tensors. Its size is 10,442,827,776 bytes versus
10,094,357,792 bytes without MTP: a 348,469,984-byte (0.324538 GiB) increment.
In pinned llama.cpp 7fe450e, common/speculative.cpp creates the embedded MTP
context against model_tgt when no separate draft file is supplied. The memory
filter in src/llama-model.cpp includes only the nextn layer for this hybrid
architecture. Consequently, the candidate replaces the standard GGUF with
the integrated file; it must not load two complete target copies.

Planning equations, assuming the documented one-layer/4-head/256-width MTP
geometry is validated against the actual candidate:

```text
C = 262,144
Q8_0 MTP KV = 2 * C * 1 * 4 * 256 * (34/32) = 0.53125 GiB
target weights/file + MTP weights/file + target KV + recurrent + draft KV
  + workspaces/rollback + reserve
  = 9.401 + 0.325 + 4.500 + 0.141 + 0.531 + unknown + 1 GiB
  = approximately 15.898 GiB + unknown
```

The user-reported current residency plus head/cache increments suggests a
capacity margin, not measured candidate residency. Verification scratch and
recurrent rollback states remain explicit unknowns; preserve 1 GiB headroom
and reject any automatic CPU offload or lowered context.

For D proposed tokens, let A be average accepted output positions including
the target token. Scalar accepted positions/call = 1; the MTP candidate has:

```text
T_cycle = T_proposer(D) + T_verifier(D+1) + T_rollback/sampling
B_GPU_cycle = target hot dense/KV traffic during verification
              + D * (MTP hot head/KV traffic) + state/intermediate traffic
B_GPU_per_accepted = B_GPU_cycle / A
T_cycle >= max(B_GPU_cycle/BW_GPU, B_RAM_cycle/BW_RAM,
               B_PCIe_cycle/BW_PCIe, B_NVMe_cycle/BW_NVMe)
T_per_accepted = T_cycle/A
beat the user-reported 30 tok/s only if T_cycle/A < 33.333 ms
```

936 GB/s theoretical GPU bandwidth and the historical 12.46 GiB/s PCIe
measurement are bounds, not sustained Swift measurements. Dense weights and
retained KV stay GPU-hot. Planned RAM/PCIe/NVMe hot-state paging is zero;
host sampling/control transfers and checkpoint/cache traffic are separate
and unmeasured. RAM/NVMe sustained bandwidth is unknown. D=2 is a candidate
starting depth, not an acceptance or throughput result. Stop the candidate if
the complete-cycle inequality or real Pi quality/tool gate fails.

Increasing batch/ubatch primarily targets prefill; GPU backend sampling is
another conditional candidate, with no quantified gain absent profiling.
More VRAM occupancy, Q8 target KV, forced cuBLAS or a different weight tier
does not by itself establish faster scalar generation. No speculative speed
result from another GPU/model is promoted to this user's active workload.

Sources: [Swift release](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF),
[pinned speculative implementation](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/common/speculative.cpp),
[pinned memory filter](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/src/llama-model.cpp),
[CUDA build options](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

### Authorized Swift MTP-10 deployment preparation

The user explicitly requested integrated MTP with a maximum of 10 proposals,
deployment to the 3090 host, and no server start/stop/restart or agent-run
inference tests. The existing no-log-reading restriction remains in force.
The later question about 50 proposals is informational and does not replace
the selected limit of 10. Startup and real Pi quality/speed gates are deferred
to the user's restart/test; preparation must not be described as passing them.

Invariants: published Swift IQ3_XXS matching-MTP package; no re-quantization;
262,144 configured positions; Q4_0 target KV; Q8_0 single-layer MTP KV;
one RTX 3090; integrated weights shared with the target; xhigh; separate
llama.cpp operations path. No native runtime/provider/ABI change is required.
The exact pinned candidate is 10,442,827,776 bytes at revision
d74895bbe5db4bec1e0024e7cc87d59c02d7631a, SHA256
e23eb251493b8a89f7d0217590b593e696ac9bacaeac8605541e477533dc5838.

The preceding memory and complete-cycle equations apply with D=10 and a
maximum verifier batch of 11 rows. The core recurrent rollback allowance,
if 10 extra full recurrent states are retained, is 10 * 144 MiB = 1.40625 GiB.
The planning subtotal becomes approximately 17.304 GiB including the existing
1 GiB reserve, before convolution state, verification workspaces and other
intermediates. This passes the capacity screen only; native allocation is
unmeasured and must retain the reserve without CPU paging or context reduction.

For the user-reported 30 tok/s reference, accept preparation without a speed
claim; a future gain requires T_cycle/A < 33.333 ms at the actual populated
Pi context. Proposer cost includes up to ten head/KV passes, verifier cost
includes up to eleven rows, and rollback/replay/sampling cannot be omitted.
Actual A remains unknown. A hypothetical constant per-position acceptance
p=0.8 gives A(D)=1+sum(p**i,i=1..D): A(10)=4.5705 and A(50)=4.9999.
That illustration is not a measured acceptance rate.

Preparation completed on 2026-10-02 through `ops/llama.sh deploy swift`.
The canonical Xet task exited successfully; the exact file passed size,
GGUF v3 and SHA256 before transactional publication under
`${MODEL_ROOT}/swift-1.5-iq3-xxs-mtp10`. An independent stdlib GGUF reader
compared the raw payload, dimensions and encoding of every original tensor:
851 backbone tensors, 10,083,362,816 bytes, all byte-identical. The candidate
adds 15 tensors at block 64 with 348,469,248 payload bytes. Metadata confirms
65 total blocks, one nextn layer, 262,144 positions, four KV heads and key/value
width 256. This is payload identity evidence, not a native numerical or coding
quality pass. Detailed read-only comparison output is retained under ignored
`out/swift-mtp10/backbone-verification.json` on both hosts.

The registry and prepared launch select `--spec-type draft-mtp`,
`--spec-draft-n-max 10`, and Q8_0 draft K/V with no second draft model path.
Pi retains `swift-1.5-iq3-xxs` and its existing endpoint/context configuration.
The real prepare command passed manifest/launch validation, and the recorded
serving task action/state/last-run plus process-record presence/hash were
unchanged across deployment. No server lifecycle operation, health/generation
request or log read was performed. Source synchronization and a separate Xet
download task are not serving restarts.

Offline Windows tests passed artifact-driven MTP arguments, rejection of
invalid limits/policy fields, unchanged scalar KAT arguments and preparation
with every serving lifecycle/process/API operation replaced by a throwing
mock. Atomic launch replacement initially failed because Windows PowerShell
marshalled `$null` as an empty backup path; `[NullString]::Value` fixed it and
the complete gate passed. The canonical Python contract passed 99 tests with
one NumPy-dependent skip. No C++/CUDA/provider/ABI edit or build was performed.
Startup/residency, actually populated context, native sampling/rollback quality,
real Pi tool/coding behavior and throughput remain explicitly unrun, reserved
for the user's restart/test. Do not cite earlier non-MTP integration probes
as evidence for this candidate.

### Authorized Swift MTP-20 preparation

The user reports 45–60 tok/s displayed by Pi and approximately 19 GB VRAM
after their MTP-10 restart, then requests a limit of 20 proposals. This is a
user observation, not an agent-measured benchmark: populated context, Pi rate
accounting, acceptance and quality have not been inspected. Keep the no-log,
no-serving-start/stop/restart and no-agent-inference restrictions. The user
will activate and test the prepared limit as before.

Invariants remain Swift IQ3_XXS with its previously verified integrated MTP
file/SHA256, 262,144 configured positions, Q4_0 target K/V, Q8_0 MTP K/V,
one RTX 3090, xhigh, full GPU placement and the separate llama.cpp path.
Only the maximum proposal count changes. Publish a new policy manifest under
`${MODEL_ROOT}/swift-1.5-iq3-xxs-mtp20` using a hard link to the same verified
GGUF; retain the MTP-10 artifact unchanged. No download or native change is
required. The maximum verifier batch is 21 rows, within the existing 128-row
microbatch and 512-row batch.

Capacity and acceptance prerequisites before preparation:

```text
C_configured = 262,144; C_populated remains unknown
weights/backbone file allowance = 9.401 GiB
MTP head allowance = 0.324538 GiB
target Q4_0 KV = 2*C_configured*16*4*256*(18/32) = 4.500 GiB
MTP Q8_0 KV = 2*C_configured*1*4*256*(34/32) = 0.53125 GiB
recurrent base = 144 MiB = 0.140625 GiB
20 additional recurrent rollback states = 20*144 MiB = 2.8125 GiB
total + 1 GiB reserve = approximately 18.710 GiB before conv/workspaces
increment relative to D=10 = 10*144 MiB = 1.40625 GiB + extra workspaces

T_cycle20 = T_proposer(up to 20) + T_verifier(up to 21) + T_rollback/sampling
B_GPU_cycle20 = B_target_weights(21 rows) + B_target_KV(21,C_populated)
                + up to 20*(B_hot_head + B_draft_KV(C_populated))
                + B_recurrent/intermediate
B_GPU_per_accepted = B_GPU_cycle20/A20
T_cycle20 >= max(B_GPU_cycle20/BW_GPU, B_RAM_cycle20/BW_RAM,
                 B_PCIe_cycle20/BW_PCIe, B_NVMe_cycle20/BW_NVMe)
gain requires T_cycle20/A20 < T_cycle10/A10 at matched populated context
```

The existing bandwidth bounds apply: 936 GB/s theoretical GPU, historical
12.46 GiB/s PCIe, unknown sustained RAM/NVMe. Planned hot-state RAM/PCIe/NVMe
paging remains zero; host sampling/control/checkpoint traffic is unmeasured.
All full-softmax retained KV and dense weights remain hot; no compression,
eviction or CPU offload is introduced. The user's 45–60 tok/s display implies
16.667–22.222 ms/token only within that display's unknown accounting. More
proposals are beneficial only if accepted output offsets the extra complete
cycle cost. The capacity screen permits preparation, not a speed or native
allocation claim; startup, actual residency and real Pi quality/speed gates
remain reserved for the user's restart/test.

MTP-20 preparation completed on 2026-10-02. The same GGUF passed exact
size/header/SHA256 validation and was hard-linked into the new transactional
policy artifact before registry promotion. Only the Swift artifact container
and `speculative.max_tokens` changed; KAT, model ID, weights, context, target
and draft KV, sampling/reasoning policy and the native binary are unchanged.
The policy builder validated 20 offline, and `ops/llama.sh deploy swift`
validated publication and prepared the real host launch. Read-only launch
verification confirmed maximum 20 proposals, no second draft file and the
unchanged context/KV contract. Serving task action/state/last-run and process
record presence/hash were unchanged; the MTP-10 manifest and completion hashes
also remained unchanged. Verification is retained under ignored
`out/swift-mtp20/deployment-verification.json` on both hosts. No log reads,
API/inference requests, GPU probes or serving lifecycle operations were made.
No tests or native code were changed; live MTP-20 residency/quality/speed
remain unrun and reserved for the user.

### Authorized return to Swift MTP-10

After their MTP-20 restart, the user reports 30–40 tok/s in Pi and about
20 GB VRAM, then explicitly requests deploying the limit of ten without
restarting. The earlier MTP-10 observation was 45–60 tok/s and about 19 GB.
These are user observations with unknown populated context and rate accounting,
not a controlled comparison or proof of a specific rejection/rollback cause.
Do not inspect, test or read logs from the active development session.

Reuse the intact published `${MODEL_ROOT}/swift-1.5-iq3-xxs-mtp10` artifact
and its exact original policy. Only the artifact selection and maximum
proposal count return from 20 to 10. Swift IQ3_XXS/MTP weights, configured
262,144 context, Q4_0 target KV, Q8_0 draft KV, xhigh, single RTX 3090 and
the separate llama.cpp path remain unchanged. The preceding D=10 equations
apply: approximately 17.304 GiB including reserve before conv/workspaces,
with up to 11 verifier rows; complete-cycle time per accepted output still
determines speed. No new inference mechanism, download, fidelity change or
native build is introduced. Deployment must validate the existing artifact
and prepare the next launch only; activation and live gates stay with the user.

Return deployment completed on 2026-10-02 through `ops/llama.sh deploy swift`.
The existing MTP-10 artifact passed manifest/completion, file-size, GGUF v3
and exact SHA256 validation before selection. The prepared launch file declares
maximum ten proposals and the original context/KV/model-ID contract. Only the
Swift artifact selector and maximum proposal count changed. Prepared-file
evidence is retained under ignored
`out/swift-mtp10-return/deployment-verification.json` on both hosts. There were
no serving lifecycle operations, active-session/process/GPU inspection, API
requests, inference tests or log reads. The running session is not reconfigured;
the prepared policy takes effect on the user's next restart.

### Configurable artifact MTP execution policy

The user requests an environment-configurable Qwen/QPack proposal count,
deployment, no tests and no restart, and explicitly delegates separate KAT
MTP investigation. Existing active-session/log restrictions remain. This is
configuration of the accepted resident sampled-MTP mechanism, not reopening
the rejected exact offloaded MTP tree or claiming a speed improvement.

Invariants: current Qwen3.8-27B QPack weight/quantization ABI, operational Q4H
target KV and Q8 proposer KV, configured 262,144 context, unchanged populated
context (not inspected), xhigh/exact sampled rejection, one RTX 3090, fixed
12 GiB routed ceiling, artifact/capability selection without model branches.
Absent override retains the declared MTP-4 policy. The startup override edits
only the in-memory exact-decode parameters and emitted-token ceiling; stored
artifact/program hashes and tensor bytes remain unchanged. All binder/executor
and selected-provider views must receive the same effective descriptor.

For D proposals and R=D+1 verifier rows:

    resident = weight payload + Q4H target KV + Q8 MTP KV
               + base/retained recurrent state + R recurrent checkpoints
               + workspaces + 1 GiB reserve
    Q4H KV(C=262144) = 4.0625 GiB; Q8 MTP KV = 0.503906 GiB
    conservative complete QPack file allowance = 14775390208 bytes
    core recurrent state = 48*48*128*128*4 = 144 MiB per state
    D=4 -> D=10 checkpoint increment = (11-5)*144 MiB = 0.84375 GiB
    attention partial-output increment at 512 splits =
        (11-5)*512*24*256*4 = 0.0703125 GiB

Convolution, query/scales, logits and projection workspace increments are
additional and dimensioned by the effective policy. Exact-F16 at fully
populated maximum context still fails the prior resident inequality and is
not admitted by this change. Dense weights and every retained full-softmax KV
position are hot. Larger row counts can select existing GEMM/staged-attention
paths, whose traffic/cost must not be assumed equal to the batch-five path.

    B_GPU_cycle = B_target_weights(R) + B_target_KV(R,C_populated)
                  + D*(B_hot_MTP_head + B_MTP_KV(C_populated))
                  + B_recurrent/intermediate
    B_per_accepted = B_cycle/A
    T_cycle = T_proposer(D) + T_verifier(R) + T_rollback/sampling
    T_cycle >= max(B_GPU/BW_GPU, B_RAM/BW_RAM,
                   B_PCIe/BW_PCIe, B_NVMe/BW_NVMe)
    improvement requires T_cycle(D)/A(D) < T_cycle(4)/A(4)

Prior theoretical GPU bandwidth is 936 GB/s and historical PCIe bandwidth
12.46 GiB/s; sustained RAM/NVMe bandwidth is unknown. No hot-state paging,
fidelity downgrade, eviction or CPU fallback is introduced. Provider startup
must reject a depth exceeding its workspace/context or resource capacity.
This prerequisite admits configurable execution without a throughput claim.
The user's explicit no-test/no-restart instruction overrides native/live gate
execution for this deployment: use a clean Windows Release/CUDA candidate
build, retain the active executable, and report unrun numerical/chat/Pi gates.

Deployment completed on 2026-10-02: MSVC Release/CUDA 13.4, SM86, fresh
configuration and `--clean-first`, with both test-build options disabled and
only target `expert-moe-vm-runner` built. The produced runner is
`out/build/windows-msvc-release-mtp-env-20261002/runtime/Release/expert-moe-vm-runner.exe`,
6,298,112 bytes, SHA256
`9f1b608957025ca55f226f0df20bb6133b50eea45d91af6a7195e6ca2c7c44a0`.
The local ignored `.env` now selects this runner through `MODEL_VM_RUNNER`;
`MODEL_MTP_DRAFT_DEPTH=10` selects ten proposals for the next Qwen start.
The authenticated artifact still declares four as its unchanged stored policy.
Launcher/server/source changes are deployed. No scheduled serving task was
registered or altered and no service was stopped, started or restarted.
Build/deployment metadata is at ignored `out/mtp-env-policy/deployment.json`
on both hosts. Compilation and file-integrity evidence do not qualify native
execution, sampling fidelity, populated-context capacity or coding throughput;
all such gates remain unrun at the user's explicit instruction.

### KAT IQ4_XS with a community MTP head

The user explicitly requests delegated KAT MTP investigation and deployment
if available, without testing or restarting. The original KAT artifact has
no MTP head. The community repository
`thread13/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF-MTP`, revision
`b2c680bf22e192eae1fc30e6060cac77dd144a91`, publishes
`Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.bartowski.mtp.gguf`
(19,704,404,096 bytes; SHA256
`5037eb2b43466696aa5e3e918f4bfb8d12b6d3a769a28a36ceb84c9289a195a9`).
Its documented construction adds one Qwen3.6 MTP head to the bartowski KAT
backbone; this is a graft, not a natively trained KAT MTP head. Publisher
guidance recommends a maximum of four proposals. Construction reference:
https://gist.github.com/buzz/1c439684d5e3f36492ae9f64ef7e3f67 .
The published artifact is not a coding-quality or acceptance result.

Invariants: same KAT IQ4_XS target choice, one RTX 3090, configured 262,144
context (populated context not inspected), separate llama.cpp path, Q4_0
target KV/Q8_0 draft KV, ten full-attention layers and thirty recurrent
layers, four proposals, no changes to the prepared Swift MTP-10 launch.
At the configured maximum, the capacity screening is:

    resident >= 18.351156 GiB weights including grafted head
                + 1.406250 GiB target KV + 0.265625 GiB MTP KV
                + 0.058594 GiB recurrent state
                + 0.234375 GiB four rollback states
                + 1 GiB reserve
             = 21.315999 GiB before convolution and other workspaces
    target KV = 2*10*2*256*262144*(144/256) bytes
    MTP KV = 2*1*2*256*262144*(272/256) bytes
    recurrent state = 30*32*128*128*4 bytes
    B_cycle = B_hot_dense/shared/selected_MoE + B_full_target_KV
              + 4*(B_hot_MTP_head + B_MTP_KV) + B_intermediates
    1 <= A_4 <= 5; B_per_accepted = B_cycle/A_4
    T_cycle >= max(B_GPU/936 GB/s, B_PCIe/12.46 GiB/s,
                   B_RAM/BW_RAM, B_NVMe/BW_NVMe)
    speed gain requires (T_proposer + T_verifier + T_rollback)/A_4 < T_scalar

RAM/NVMe sustained bandwidth and proposal acceptance are unknown. This
screen permits a resident artifact deployment candidate; it does not prove
maximum populated-context residency or speed. Use the canonical Xet download
and transactional publisher with size/header/hash integrity checks. No
backbone comparison, numerical validation, model/API/Pi execution or active
session/log inspection is authorized for this deployment. The graft's
quality and acceptance remain unqualified, and activation stays with the user.

Scope clarification on 2026-10-02: the user questioned why a separate KAT
GGUF was being downloaded. Publication/promotion and production registry
changes were placed on hold pending clarification of whether the borrowed
Qwen3.6 MTP head is acceptable. The original KAT checkpoint declares zero
MTP layers. No KAT publication was performed, no production KAT configuration
or prepared Swift launch was changed, and no service was operated. The
canonical community-artifact download remains separate from the published
original KAT; copied-backbone fidelity is only the publisher's claim and was
not independently verified under the user's no-test instruction.
After the borrowed-head provenance was disclosed, the user explicitly
directed continuation of the download. The original deployment request
remains active; the temporary publication hold is lifted. No tests or
serving lifecycle operations are authorized.

KAT deployment subsequently completed on 2026-10-02. The exact published
GGUF passed size, v3 header and pinned SHA256 integrity gates and was
transactionally published at `${MODEL_ROOT}/kat-coder-v2.5-iq4-xs-mtp4`.
Only the KAT registry entry and its launcher documentation were promoted
and copied to `3090box`. Alias `kat` and Pi model ID
`kat-coder-v2.5-iq4-xs` are unchanged; the display name explicitly identifies
the borrowed Qwen3.6 MTP head. The configured policy is four draft-MTP
proposals, Q8_0 draft KV, Q4_0 target KV, and 262,144 configured context.
The original scalar KAT artifact remains intact. Read-only GGUF inventory
confirms 41 blocks with one next-token-prediction block and twenty head
tensors (Q8_0 matrices, BF16 routers, F32 norms). This is metadata evidence,
not numerical or backbone fidelity evidence. The existing llama.cpp binary
and prepared Swift MTP-10 launch were untouched. No model was started,
stopped or restarted and no tests were run. Activation stays with the user's
manual `./ops/llama.sh restart kat`. Ignored evidence is retained at
`out/kat-mtp/artifact-metadata.json` and
`out/kat-mtp/deployment-result.json`.

### Authorized Qwen MTP-6 configuration

The user reports a considerable coding-speed degradation with MTP-10 and
requests six proposals, deployment, and manual activation. This is a user
observation without measured populated context, acceptance or cycle timing;
it does not establish a particular proposer/verifier bottleneck.

Invariants: same Qwen3.8-27B QPack and existing protocol-13 executable,
one RTX 3090, configured 262,144 context, operational Q4H target/Q8 MTP KV,
xhigh, unchanged fidelity and fixed 12 GiB routed ceiling. The universal
environment policy changes to `MODEL_MTP_DRAFT_DEPTH=6`; no model branch,
native change, repack or rebuild is needed. Stored artifact MTP-4 remains
the fallback when the override is blank.

For D=6 and R=7 verifier rows, the preceding capacity/traffic equations apply:

    resident = weights + Q4H target KV + Q8 MTP KV
               + recurrent/retained state + 7 rollback states
               + workspaces + reserve
    D=10 -> D=6 recurrent-checkpoint reduction = (11-7)*144 MiB = 0.5625 GiB
    attention partial-output reduction at 512 splits =
        (11-7)*512*24*256*4 = 0.046875 GiB
    B_cycle = B_target_weights(7) + B_target_KV(7,C_populated)
              + 6*(B_hot_MTP_head + B_MTP_KV(C_populated)) + B_intermediates
    1 <= A_6 <= 7; B_per_accepted = B_cycle/A_6
    T_cycle >= max(B_GPU/936 GB/s, B_PCIe/12.46 GiB/s,
                   B_RAM/BW_RAM, B_NVMe/BW_NVMe)
    improvement requires T_cycle(6)/A_6 < T_cycle(10)/A_10

Other workspace differences and sustained RAM/NVMe bandwidth remain
unmeasured. The smaller allocation policy is admitted by the existing
workspace/context bounds; no speed improvement is claimed. Update the
local ignored environment, example, operations documentation and deployment
metadata, then copy only those public configuration/docs files to the host.
Do not register or start a serving task, read session logs, run tests, or
inspect the active session. The user's next `ops/model.sh start qwen` applies
six proposals through the existing launcher. Swift and KAT remain unchanged.

### Swift reasoning efficiency and transfer to Qwen, 2026-10-02

Scope: primary-source research and static source/configuration inspection.
The user reports shorter Swift thinking at xhigh and more selective tool use
than KAT. No active Pi transcript, serving log, live telemetry, test, inference,
download, configuration deployment or service lifecycle operation was used.

Invariants: independent single RTX 3090 serving; Swift 1.5 GSQ IQ3_XXS GGUF
through llama.cpp with Q4_0 target/Q8_0 draft KV and ten MTP proposals;
Qwen3.8-27B QPack through the artifact VM with operational Q4H target/Q8
draft KV and the prepared six-proposal policy. Both configure 262,144 context
and xhigh. Current populated context and effective running policies were not
inspected. This research does not restore the superseded exact-F16 selection
criterion or introduce model-specific runtime branches.

Swift's publisher attributes shorter reasoning to penalized overthinking
markers during post-training, followed by RL and on-policy distillation.
Version 1.5 emphasizes coding and long-horizon agent tasks. Its BF16 release
comparison uses the same xhigh protocols for both arms:

| Evaluation | Base score | Swift score | Mean token reduction |
|---|---:|---:|---:|
| LiveCodeBench v6 | 76.76 | 81.71 | 24.5% |
| Terminal-Bench 2.1 | 69.21 | 72.13 | 16.3% |

These are publisher results, not current IQ3_XXS/Pi/256K measurements.
[Release](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b).
The released warm-start dataset has 15,209 terminal/coding agent traces;
it does not supply the complete Swift RL/distillation recipe.
[Dataset](https://huggingface.co/datasets/ukisai/Qwen3.8-27B-multi-turn-agent-sft).
The older evaluation repository describes base plus Swift LoRA, whereas the
inspected 1.5 release tree contains merged weights and no adapter files.
Do not promise a downloadable 1.5 adapter or direct addition to packed weights.
[Older evaluation repository](https://github.com/UkisAI/Swift-Qwen3.8-27B-evals),
[1.5 release tree](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/tree/main).

Static local findings:

- Both entries in `~/.pi/agent/models.json` map xhigh to xhigh and send
  `enable_thinking`, `preserve_thinking=true` and `reasoning_effort` through
  chat-template kwargs. The installed Pi completion provider resolves the
  model's thinking-level map without lowering this value.
- The published Swift template and the deployed Qwen artifact template have
  the same xhigh instruction. It asks for careful validation; it is not a
  numeric token quota. No wire request or actual rendered prompt was observed.
- `expert_server.py` passes effort into the artifact tokenizer template.
  Its 32,768 `maximum_thinking_tokens` bound clamps the entire completion when
  thinking is enabled; it does not count only reasoning or guarantee room for
  a visible answer. Swift's Pi output allowance is also 32,768. The larger
  advertised Qwen `maxTokens` is not evidence of a larger effective thinking
  allowance on this path.
- Qwen's artifact presence penalty is 0.5, versus the publisher's 0.0.
  Earlier 0.0/0.5 diagnosis already failed to restore Romanian fidelity;
  sampling parity is not a demonstrated fix for reasoning or coding.

[Swift template](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/main/chat_template.jinja),
[Base template](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja).

Efficiency prerequisites, separating fewer generated tokens from faster decode:

```text
T_task = T_cold/reused_prefill + N_generated/r_accepted
         + sum(T_tools) + T_recovery
N_generated = N_reasoning + N_visible + N_tool_serialization

C_configured = 262,144; C_populated unknown
M_resident = W_dense + W_embedding/output + W_recurrent + W_MTP
             + S_recurrent + KV_target(C) + KV_MTP(C)
             + S_rollback(D) + workspaces + reserve
No routed experts/routers exist in these dense 27B backbones.
Swift whole-file allowance: 9.401 GiB backbone + 0.324538 GiB MTP
Swift KV_target = 2*16*4*256*C*(18/32) = 4.5 GiB at C_configured
Swift KV_MTP = 2*1*4*256*C*(34/32) = 0.53125 GiB
S_recurrent = 48*48*128*128*4 = 144 MiB
S_rollback allowance at D=10 = 10*144 MiB = 1.40625 GiB
Swift subtotal plus 1 GiB reserve = about 17.304 GiB + workspaces
Qwen known hot target weight aggregate = 12.690 GiB
Qwen full pack = 13.761 GiB, includes other roles/MTP; do not add it twice
Qwen KV_target = 4.0625 GiB; KV_MTP = 0.503906 GiB at C_configured
Separate dense/embedding/recurrent/MTP role traffic has not been remeasured.

T_cycle(D) = T_proposer(D) + T_verifier(D+1) + T_rollback/sampling
1 <= A_D <= D+1; r_accepted = A_D/T_cycle(D)
B_target_call = B_hot_target_weights + B_retained_full_attention_KV
                + B_recurrent/intermediates
B_cycle = B_target_verify(D+1,C_populated)
          + sum_D(B_hot_MTP_weights + B_MTP_KV(C_populated))
          + B_rollback/control
B_per_accepted = B_cycle/A_D
T_cycle >= max(B_GPU/936 GB/s, B_PCIe/12.46 GiB/s,
               B_RAM/BW_RAM, B_NVMe/BW_NVMe)
```

File sizes are capacity allowances, not hot weight traffic or a speed ratio.
936 GB/s is theoretical GPU bandwidth and 12.46 GiB/s is historical pinned
PCIe evidence; current sustained GPU/RAM/NVMe bandwidth is unknown. Planned
hot dense/KV offload remains zero. Longer proposals help only if accepted
tokens offset proposer, verifier and rollback costs. The user's slowdown at
larger depths is not a reason to copy Swift's ten proposals into Qwen.

GSQ refines assignments/scales and RCO allocates per-tensor precision under
a size budget; the Swift release reuses ISTA allocations with Swift-specific
refinement. This reduces weight footprint separately from trained thinking
efficiency. Its published GGUF KLD screens use context 512. Current QPack
FP4 providers cannot execute IQ3/GGUF encodings by relabeling the artifact.
[GSQ](https://arxiv.org/abs/2604.18556),
[RCO](https://arxiv.org/abs/2605.00649),
[Swift quantization](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF).

Transfer decision:

1. With existing Qwen weights, a concise working policy is a candidate:
   retain xhigh, validate material assumptions once, revisit decisions only
   after contradictory evidence, locate relevant paths/symbols before reading,
   and expand the search only to resolve a concrete dependency. No numeric
   thinking cutoff or reasoning-marker logit ban is justified. This is an
   unmeasured prompt hypothesis, not the Swift training method. Proceed only
   if real user coding keeps correctness/tool selection while reducing task
   time; stop if it merely shifts tokens to retries or loses correctness.
2. Reusing trained behavior requires adapted weights, such as the released
   Swift checkpoint or a reproducible efficiency/agent fine-tune. QPack
   qualification must verify source geometry, packing/provider quality and
   real coding before promotion; common runtime selection stays artifact-driven.
   Repacking into current FP4 does not establish retention of GSQ quality.
3. GSQ/RCO is a separate encoding/provider or existing GGUF-serving project.
   Its prerequisite is a supported ABI fitting the full memory equation and
   preserving coding behavior, followed by lower accepted-token/task cost.
   Do not reopen rejected local tensor-SSE tuning as a substitute.

No runtime improvement has been implemented or measured by this research.

### Authorized Swift two-slot preparation, 2026-10-03

The user requests two parallel users. They initially clarify 262,144 per user,
then explicitly direct changing ONLY parallelism, with no other option change.
The subsequent instruction also explicitly authorizes automatic RAM offload.
The final policy retains the total 262,144 pool and ten MTP proposals;
the pinned build consequently provides 131,072 per slot. This unavoidable
effect was disclosed before deployment. Do not increase total context or
change Pi metadata under this final instruction.
Prepare and deploy without restarting, testing, inference, session/log reads
or active-process inspection. Activation and real validation stay with the user.

Invariants: Swift 1.5 GSQ IQ3_XXS with the same integrated head and pinned
file identity, one RTX 3090 with its local CPU/RAM, xhigh, Q4_0 target/Q8_0
draft KV, ten proposals per slot. Configure two slots and retain the total
262,144-token pool; per-slot context becomes 131,072. KAT retains one slot.
The artifact declares parallelism; common launchers must not branch on names.

Capacity and traffic prerequisites:

```text
P=2; C_total=262,144; C_per_slot=C_total/P=131,072; D=10
C_populated_1 and C_populated_2 are unknown
M = W_shared_dense/embedding/recurrent + W_MTP
    + P*(KV_target(C_per_slot) + KV_MTP(C_per_slot)
         + S_recurrent + D*S_recurrent) + conv/workspaces + reserve
shared whole-file allowance = 9.725641 GiB (includes head, not exact VRAM)
target KV = 2*2.25 = 4.5 GiB
draft KV = 2*0.265625 = 0.53125 GiB
core recurrent + rollback allowance = 2*(1+10)*0.140625 = 3.09375 GiB
subtotal = 17.850641 GiB before convolution/workspaces
with 1 GiB reserve = 18.850641 GiB before convolution/workspaces

B_cycle = B_hot_shared_weights_verified_for_both_slots
          + sum_slots(B_retained_target_KV(C_populated)
                      + D*(B_hot_MTP_head + B_draft_KV(C_populated))
                      + B_recurrent/rollback/intermediates)
B_per_accepted_aggregate = B_cycle/(A_1+A_2); 1<=A_i<=11
T_cycle = T_proposer + T_verifier + T_rollback/sampling
T_cycle >= max(B_GPU/936 GB/s, B_PCIe/12.46 GiB/s,
               B_RAM/BW_RAM, B_NVMe/BW_NVMe)
```

These are estimates and historical/theoretical bandwidth bounds; actual
workspaces, sustained bandwidth and acceptance remain unknown. Additional
core recurrent/rollback allowance versus one slot is
1.546875 GiB. Actual capacity with workspaces is not established; startup
may fail allocation. The latest explicit user request enables `--fit on`
with `--gpu-layers auto` for this artifact only. The pinned fitter retains an
explicit nonzero context and includes the shared-weight MTP context in its
memory estimate. It can place layer/tensor operations in RAM/CPU at startup.
This is a capacity policy, not a dense decode-throughput solution or dynamic
spill guarantee. Fit errors and later OOM remain possible.
Two slots do not promise twice the throughput or unchanged per-user speed.
Do not lower context/MTP to hide a failure.
The user's request authorizes preparation of this unqualified configuration,
not a successful capacity, isolation, MTP, coding or throughput claim.

Publish a distinct policy artifact using the existing identical GGUF, retaining
the single-slot artifact. Make readiness checks follow declared slots and
per-slot context. Pi metadata remains at 262,144 and the API/model ID stay
unchanged. The Pi wrapper's existing strict context/slot check rejects the
new policy after activation; direct API requests are limited to 131,072 per
slot. Do not claim two 256K conversations under this parallel-only change.

For the authorized automatic placement, replace the full-GPU assumption by:

```text
M_GPU = W_GPU + KV_GPU + S_GPU + rollback_GPU + workspace_GPU + reserve
M_RAM = W_CPU + KV_CPU + S_CPU + rollback_CPU + workspace_CPU/host
B_GPU_cycle and B_RAM_cycle both include their resident hot dense/KV/state
B_PCIe_cycle = activation/state/control transfers between placed operations
B_per_accepted = B_cycle/(A_1+A_2)
T_cycle >= max(B_GPU/BW_GPU, B_RAM/BW_RAM, B_PCIe/BW_PCIe, B_NVMe/BW_NVMe)
```

CPU weights stay CPU-resident rather than being streamed to GPU each token.
Sustained RAM bandwidth, CPU quantized-kernel cost and the selected placement
are unknown. NVMe is startup storage rather than an intended hot decode tier.
No new speed claim or fit success is made. Quantization, model weights and
context/MTP settings stay unchanged. Neither QPack nor KAT receives this policy.
[Pinned fitter](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/common/fit.cpp).

Deployment completed: the canonical publisher validated the unchanged GGUF
identity and published `${MODEL_ROOT}/swift-1.5-iq3-xxs-mtp10-parallel2-auto`.
The real prepare action reports two slots, total context 262,144 and per-slot
context 131,072. Prepared native options change only parallelism and automatic
placement (`--parallel 2 --gpu-layers auto --fit on`); MTP remains ten, and
the previous binary/host/port/timeout are retained. The new artifact links
the identical weights and retains the original single-slot policy artifact.
KAT and QPack configuration are unchanged. Preparation performs no service
start/stop/restart or API request. No tests, session logs, GPU/process probes
or inference were used; fit success, residency, MTP isolation and coding
speed remain unqualified. Ignored deployment metadata is under
`out/swift-parallel2/deployment.json` on the serving host.

Two preparation transport/control failures were resolved without inference
edits: Windows rejected an oversized encoded command, so deployment used a
file; a successful PowerShell publisher does not necessarily initialize
`LASTEXITCODE`, so publication relies on terminating errors. Static inspection
confirmed the artifact completed while the prior registry/launch stayed
unchanged after that control failure, before successful final preparation.

### Swift correction: retain 256K per user, 2026-10-05

The user reported an actual API rejection at 227,164 prompt tokens against
131,072 available positions. Pi recovery also failed at 162,477 tokens.
The earlier parallel-only interpretation reduced the per-user capacity and
did not satisfy the user's intended two 256K conversations. The user now
explicitly reaffirms that tokens must not be reduced. This supersedes the
previous instruction to retain the total 262,144 pool and its Pi mismatch.

Prepare total context 524,288 with two slots, each 262,144. Preserve the same
Swift GGUF identity, IQ3_XXS weights, Q4_0 target/Q8_0 draft KV, MTP-10,
xhigh, one RTX 3090, and startup automatic GPU/CPU fitting. Pi metadata stays
at 262,144. Correct its launcher to check artifact-declared slots and per-slot
context. KAT and QPack are unchanged. Do not restart, run inference or tests,
or read active sessions/logs; the user activates and evaluates this policy.

Capacity prerequisite at two fully populated slots:

```text
P=2; C_total=524,288; C_per_slot=262,144; D=10
shared whole-file allowance = 9.725641 GiB (not exact VRAM or hot traffic)
target Q4_0 KV = 2*4.5 = 9 GiB
draft Q8_0 KV = 2*0.53125 = 1.0625 GiB
core recurrent/rollback = 2*(1+10)*0.140625 = 3.09375 GiB
subtotal = 22.881891 GiB before convolution/workspaces
with 1 GiB reserve = 23.881891 GiB before convolution/workspaces
M_GPU = W_GPU + KV_GPU + recurrent/rollback_GPU + workspaces_GPU + reserve
M_RAM = W_CPU + KV_CPU + recurrent/rollback_CPU + workspaces_CPU/host
B_cycle = B_shared_hot_target_verification
          + sum_slots(B_target_KV(C_populated)
                      + D*(B_hot_MTP_head + B_draft_KV(C_populated))
                      + B_recurrent/rollback/intermediates)
B_per_accepted = B_cycle/(A_1+A_2); 1 <= A_i <= 11
T_cycle = T_proposer + T_verifier + T_rollback/sampling
T_cycle >= max(B_GPU/936 GB/s, B_RAM/BW_RAM,
               B_PCIe/12.46 GiB/s, B_NVMe/BW_NVMe)
```

The GPU bandwidth is theoretical; PCIe is historical evidence, and current
sustained RAM/NVMe bandwidth, workspaces and acceptance are unknown. This
does not establish full-GPU capacity. Startup fit may place hot operations
in CPU/RAM, recovering capacity with potentially lower throughput. It is
not dynamic runtime spill and may still fail allocation. Do not lower the
requested context or MTP to hide a failure. NVMe is startup storage, not an
intended hot decode tier. No measured speed, fidelity or concurrent isolation
claim is made by preparation. Publish a distinct policy artifact linking the
identical GGUF before preparing the launch; retain previous artifacts.

Deployment completed at 2026-10-05T09:00:05Z: the canonical publisher
validated the file identity and published
`${MODEL_ROOT}/swift-1.5-iq3-xxs-mtp10-parallel2-ctx524288-auto`. The host's
prepare action reports total context 524,288, two slots and per-slot context
262,144, with MTP-10 and automatic fit retained. No restart, inference, tests
or session/log inspection was performed. Client metadata was not changed.
The currently running process retains its original limit until user restart.
Ignored host evidence is `out/swift-256k-parallel2/deployment.json`.

### Swift rollback to one GPU slot, 2026-10-05

After restarting the two-slot 524,288-total configuration, the user reported
increased CPU load with one connected Pi client and then severe slowdown.
Actual placement, populated context and rate accounting were not inspected;
this is user evidence, not a controlled throughput result. Do not treat
startup CPU/RAM fitting as a solution for fast dense decode.

The user explicitly requests rollback: one slot, context 262,144, full GPU
layers, fitting off, MTP unchanged at ten. Restore the existing authenticated
`swift-1.5-iq3-xxs-mtp10` artifact policy. Preserve IQ3_XXS weights, Q4_0
target/Q8_0 draft KV, xhigh, hardware, API/model identity and Pi defaults.
Prepare only; no restart, tests, inference or active session/log inspection.
The earlier single-slot capacity allowance is about 17.304 GiB including
1 GiB reserve before convolution/workspaces; this is not new measured
residency or proof of throughput. The user activates and evaluates rollback.

Rollback preparation completed at 2026-10-05T09:10:29Z. The canonical
publisher validated the existing artifact and the host prepared
`--parallel 1 --ctx-size 262144 --gpu-layers all --fit off`
with `--spec-draft-n-max 10`. The Swift registry entry removes the temporary
parallel/fit properties and restores the original artifact name. No service
lifecycle, tests, inference or Pi/log inspection was performed. Ignored host
evidence is `out/swift-single-slot-rollback/deployment.json`.

### Authorized Swift 1.5 Heretic source download, 2026-10-07

The user requests downloading the exact source repository
`akumaburn/Swift-1.5-Qwen3.8-27b-heretic` on the independent RTX 3090 host.
This is BF16 source acquisition only, not quantization, serving publication,
model activation or a coding-quality claim. Pin revision
`529f2bef62bbc2f757b8f2fa58d7e3dcb5c4da1c`: 55,593,237,644 repository bytes,
55,562,855,904 indexed tensor bytes, and 13 referenced safetensors files
(12 main shards plus the separate MTP file). Use the canonical
`Start-HuggingFaceModelDownload.ps1` / `Get-HuggingFaceModelDownload.ps1`
workflow with Xet and `${MODEL_ROOT}/.hf-cache`.

The first attempt failed before weight acquisition with Windows error 1314
while Hugging Face created a cache symlink for `NOTICE`. The installed Hub
library supports `HF_HUB_DISABLE_SYMLINKS`. The download worker now sets it
to `1` in its own environment, selecting supported copy/move caching while
retaining Xet transport and the limited scheduled-task identity. No machine
policy, serving configuration or model process was changed.

The resumed canonical task started at 2026-10-07T11:12:40Z. Its status
reported `Running`, four incomplete transfers and no exit error. Download
completion, final cache integrity and all model-quality gates remain pending.

The user subsequently clarified that the requested download must be a
llama.cpp GGUF or another runnable quantized artifact, rather than the BF16
source. The BF16 format choice was incorrect for that request. The corrected
candidate is `mradermacher/Swift-1.5-Qwen3.8-27b-heretic-i1-GGUF`, whose Hub
metadata names the akumaburn source, pinned at
`45c709974da7cdcb04ea580c7edf55a1804d1953`. Select
`Swift-1.5-Qwen3.8-27b-heretic.i1-IQ4_XS.gguf`, 15,309,040,416 bytes, SHA-256
`73f61a912ec30961161ab96083b95866eed7298448ddf1f53512e1711d6e2c2f`.
The attempted BF16 cancellation and corrected download could not execute:
SSH connections to the 3090 host timed out, including a bounded retry and a
direct port-22 connection probe. BF16 stop is unconfirmed and GGUF download
has not started. The exact request and canonical-script invocation are
prepared locally under `out/swift-heretic-gguf/`; no serving operation or
configuration change was performed.

The subsequent user-authorized WireGuard investigation on `node0` found the
A520M peer's last successful handshake at 2026-10-07T14:48:24+03:00. The
gateway service remained active since September 12 with zero restarts;
other peers had recent handshakes and the relevant gateway journals contained
no outage event. Targeted packet observation showed handshake requests sent
to the target's last endpoint with no authenticated reply from that peer.
Its generated client profile uses `0.0.0.0/0`, so a full-tunnel download is
plausible, but neither congestion nor a host resource failure was proven.
Do not report the BF16 download as the established outage cause. LAN
discovery did not identify a verified target address or a working management
path. No restart was executed. The corrected GGUF download remains pending
target connectivity; scoped evidence is `out/wireguard-3090/incident.json`.

After the user restarted the 3090 host, they explicitly requested removal
of the erroneous BF16 download and startup of the exact IQ4_XS GGUF above.
Cleanup removed the source repository cache, its repository-specific locks,
its Xet log and download logs/status, freeing 22,257,692,672 disk bytes;
both the BF16 cache and locks were verified absent. Other model caches and
serving processes/configuration were preserved. Evidence is
`out/swift-heretic-gguf/cleanup.json`.

The canonical Xet download task started at 2026-10-07T12:43:21Z for only
`Swift-1.5-Qwen3.8-27b-heretic.i1-IQ4_XS.gguf`, pinned to the revision above.
Use one worker with `XetHighPerformance=0`; Xet remains enabled. The user
requests leaving the download in the background and will announce completion.
No completion, integrity, serving, MTP or coding-quality gate is claimed.

On the user's subsequent completion check, the GGUF task had exited with
code 1 during Xet reconstruction: a request to the Hugging Face CDN failed.
The available error does not establish a lower-level network cause. The
same canonical download was resumed at 2026-10-07T13:05:40Z with the pinned
revision, exact selected file, one worker and Xet enabled; partial cache
was retained. Its initial status reported `Running`, one incomplete transfer,
4,162,738,680 downloaded bytes and no exit error. Completion and SHA-256
verification remain pending. No serving operation was performed.

The subsequent completion check found the exact selected GGUF present with
15,309,040,416 bytes and downloader exit code 0, with no worker or running
download task. The status script reported `completed=false` and
`integrity_checked=false` because its expected cached tree metadata was
unavailable. A direct read-only SHA-256 check of the downloaded snapshot file
matched the pinned published value
`73f61a912ec30961161ab96083b95866eed7298448ddf1f53512e1711d6e2c2f`.
Acquisition and file integrity are complete; this does not qualify serving,
MTP or coding quality. No serving configuration or process was changed.

### Swift Heretic Pi/launcher configuration and stable-release review, 2026-10-07

The user requested `swift-heretic` in the repository and home Pi settings,
and all existing `ops/llama.sh` actions for this model. Read-only GGUF
inventory found architecture `qwen35`, 64 backbone blocks plus one MTP block,
context metadata 262,144 and 15 MTP tensors. Reuse the existing Swift launch
policy: one RTX 3090, one slot, Q4_0 target KV, Q8_0 draft KV and ten maximum
proposals, all GPU layers, fitting disabled. Preserve the exact downloaded
IQ4_XS file and existing model/default settings. This is the separate
user-authorized llama.cpp path, not an artifact-VM capability claim.

The registry and launcher help were deployed to the 3090 host. The canonical
publisher passed exact size/header/SHA-256 validation and transactionally
published `${MODEL_ROOT}/swift-heretic-iq4-xs-mtp10`. The home model catalog
contains `llama-3090/swift-heretic`, and both Pi settings enable it at xhigh.
JSON/shell syntax and matching model/context/output limits were checked;
existing Pi settings and providers were compared against backups. No start,
stop, restart, prepared-launch replacement, inference request, benchmark or
active Pi/server log inspection was performed. Coding, transport, populated
context, residency and MTP acceptance remain unmeasured for this GGUF.

The installed executable's `--version` reports `0.5.0-dev`, commit `7fe450e`.
GitHub's latest non-prerelease is [v0.6.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0),
published October 5, 2026. The exact installed-to-stable comparison contains
283 subsequent commits. Relevant changes include
[probabilistic draft/MTP sampling](https://github.com/ggml-org/llama.cpp/pull/27694),
original speculative batch-order preservation, recurrent-state restore and
reservation fixes, KV rotation restore correction and a defensive
[tool-parser pointer reset](https://github.com/ggml-org/llama.cpp/pull/29942).
Probabilistic drafting requires `--spec-draft-sampling probabilistic`;
greedy remains the default. The publisher's throughput results for that PR
use RTX 5090/DGX Spark, short prompts and three proposals, not this 3090
Heretic/Pi/context/MTP-10 contract. Do not transfer their measured gains.

[CUDA FlashAttention scheduling](https://github.com/ggml-org/llama.cpp/pull/29435)
also changed, but its reported measurements use F16 KV on other GPUs; they
do not establish an improvement for this Q4_0 KV service. Shared-expert MMVQ
fusion applies to MoE paths rather than the dense Heretic model. The stable
release also changes the internal batch API and serialized session versions.
This was a read-only release review: no new binary was installed and no
sampling or serving setting changed. Evidence is
`out/swift-heretic-gguf/llama-update-research.json`.

### Authorized llama.cpp stable update, 2026-10-07

The user explicitly requests upgrading the installed llama.cpp rather than
leaving a release recommendation. Resolve the latest stable release as
`v0.6.0`, source commit `d81235049384534c167caea52b85a694f6103d14`.
Keep all downloaded/published weights, contexts, target/draft KV, proposal
counts, Pi defaults and model arguments unchanged. The active service must
not be stopped or restarted. No Pi/server log or inference gate is permitted.

`ops/llama.sh update` now invokes a source-verified, clean-first Windows
Release/CUDA build for architecture 86 and validates executable version and
CLI compatibility before installation. Generic launches resolve an authenticated
installation record instead of a hardcoded v0.5.0 executable. The update only
changes the prepared launch's binary path after its gates, retaining old
binaries for the active process. No native artifact-VM code or ABI changes.

A read-only capture check using the existing executable reproduced a Windows
PowerShell defect in the initial updater: native logging on stderr redirected
through `2>&1` becomes a terminating `NativeCommandError` with
`ErrorActionPreference=Stop`, even for `--version`. One corrective patch uses
`System.Diagnostics.Process` with independent asynchronous stdout/stderr
reads and explicit exit-code checking for CLI inspection. This is a shell
capture defect, not evidence of an inference or CUDA failure. The in-flight
clean build remains valid; its subsequent installation phase can be resumed
with the corrected capture without rebuilding unchanged native dependencies.
Build completion and installation gates remain pending at this checkpoint.

The clean Release/CUDA build completed successfully and produced
`out/llama-v0.6.0/build/bin/Release/llama-server.exe`. The initial updater
then encountered the already-reproduced stderr capture error. Resuming only
the corrected post-build phase on that exact executable passed version/tag
commit and required CLI-option checks; no native dependency was rebuilt or
changed after the successful clean build. Installation completed at
2026-10-07T13:54:11Z, executable SHA-256
`87b963a7e2f5e5b155d74558d6f186b38cff4b96d2b1cc843e1fac2c1fe1e262`.
The launcher and prepared launch now resolve this binary. Independent file
comparison confirmed the prepared model and every model argument unchanged,
and the active process record remained byte-identical. No inference or
service lifecycle operation was executed. Evidence is
`out/swift-heretic-gguf/llama-update-verification.json`.

An end-to-end updater audit also identified that repeating `update` at the
same installed version must not run `--clean-first` on a potentially active
binary directory. The final updater verifies the installed commit/SHA-256
and returns `already-installed` for that case. A subsequent invocation
confirmed this branch without compilation or service operations. Future
stable versions use separate versioned build directories. This installation
gate does not qualify model quality, MTP acceptance or Pi throughput; the
active process continues using its original executable until the user restarts.

### Authorized Swift Heretic CPU vision deployment, 2026-10-07

The user requests downloading the Q8 visual encoder through the canonical
Xet scripts, configuring CPU/RAM placement, enabling Pi image input and
deploying without restart. Keep the exact Heretic IQ4_XS backbone/MTP GGUF,
one RTX 3090, 262,144 configured positions, one slot, Q4_0 target KV, Q8_0
draft KV and ten maximum proposals. This extends the separate llama.cpp
operations path through artifact-declared projector metadata; no native or
artifact-VM code changes, inference gate or active session/log inspection.

Pinned projector repository:
`mradermacher/Swift-1.5-Qwen3.8-27b-heretic-GGUF`, revision
`19cb94a114abdd6c8fba7669e4743fb1f25d2fd0`, exact file
`Swift-1.5-Qwen3.8-27b-heretic.mmproj-Q8_0.gguf`, 629,247,712 bytes,
SHA-256 `7cd1484ca1b93b8d5d27aa0b8965a657389060565f16e08612ee4c253912eb88`.
The author reports the source vision tower untouched. Q8 quantization was
explicitly accepted by the user's request; OCR fidelity remains unmeasured.

Capacity prerequisite at the user's reported approximately 23.3 GiB GPU use:

```text
GPU-only tower lower bound: 23.3 + 629247712 / 2^30 = 23.886 GiB
remaining within nominal 24 GiB: about 0.114 GiB, before vision workspaces
CPU tower RAM >= 0.586 GiB stored payload + W_vision(image dimensions)
existing GPU weights + recurrent + KV + MTP + workspaces remain unchanged
visual embedding transfer allowance: V * 5120 * 4 bytes per image
  V=4096 -> 80 MiB; actual V depends on image resolution/preprocessing
PCIe transfer lower bound/image >= embedding bytes / measured PCIe bandwidth
CPU encoder time >= max(encoder FLOPs / CPU compute,
                       encoder/state bytes touched / RAM bandwidth)
encoder NVMe cold-load bytes >= projector payload; no planned per-token paging
target decode still touches the same hot dense weights and populated KV
MTP cycle cost and acceptance A retain the existing unmeasured model policy
```

This arithmetic rejects a GPU-resident tower as a safe deployment promise.
CPU vision performs encoding once per image, then supplies embeddings to the
existing GPU decoder. It is not hot dense target-weight or active-KV offload
as a throughput mechanism. Dynamic CPU/image and GPU graph workspace peaks,
visual latency and OCR quality require later user-authorized live gates;
configured context is not evidence of populated context or free GPU memory.
Prepare `--mmproj` plus `--no-mmproj-offload` only after exact
size/GGUF-v3/SHA-256 validation and transactional publication. Do not change
the active process. Deployment and file/argument gates were initially pending.

The canonical Xet selected-file task completed with exit code 0 and exactly
629,247,712 bytes. Download status still lacks cached tree integrity metadata;
the canonical publisher independently passed size, GGUF v3 and pinned
SHA-256 gates for both the unchanged main GGUF and the new projector before
transactional publication. The container
`${MODEL_ROOT}/swift-heretic-iq4-xs-mtp10-vision-cpu` hard-links both files;
it does not duplicate or modify the active backbone artifact.

An optional strict `projector` registry object declares identity and placement.
The publisher authenticates all declared files; launch preparation rechecks
the projector before emitting `--mmproj <published-file> --no-mmproj-offload`.
Absent or mismatched declared files fail before service lifecycle operations.
The user Pi catalog now enables `["text", "image"]` for `swift-heretic`;
both project/home settings already include this ID. Defaults, other providers,
compaction and other model policies remain unchanged. Host-only metadata
reported 63.93 GiB physical RAM and 54.94 GiB available before encoder loading;
this is capacity context, not a measured image-workspace requirement.

Deployment preparation completed without restart. Independent prepared-file
comparison confirmed context 262,144, one slot, GPU layers all, fitting off,
Q4_0 target KV, Q8_0 draft KV, MTP-10 and every other argument unchanged,
except the equivalent hard-linked backbone path and new projector options.
The active process record remained byte-identical. No API/inference request,
Pi test or active session/server-log inspection was performed. Activation and
live image quality/residency gates remain with the user after restart.
Evidence: `out/swift-heretic-vision/deployment-verification.json`.

### Authorized GPT-OSS-20B llama.cpp/Pi setup, 2026-10-07

The user requests GPT-OSS-20B GGUF download and Pi configuration on the
independent RTX 3090 host. Use the existing separate llama.cpp operations,
native author MXFP4 experts, one slot, 131,072 configured context positions,
F16 target KV and no speculative proposer. The model's native maximum
reasoning level is high; Pi xhigh maps to high. Existing Swift/Pi defaults,
other models and active service remain unchanged. No active-session probe,
inference gate, start, stop or restart is part of this setup.

Source: `ggml-org/gpt-oss-20b-GGUF`, pinned revision
`ef9b12f2ff56c69cf32153a02784e7a3c88bf524`, exact selected file
`gpt-oss-20b-MXFP4.gguf`, 12,109,566,624 bytes, SHA-256
`27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`.
Only this file is requested through canonical Xet scripts, MaxWorkers=1 and
XetHighPerformance=0. No Eagle draft, upcast BF16 or additional quantization.

Capacity prerequisite from the source config and published llama.cpp guide:

```text
W = 12109566624 / 2^30 = 11.278 GiB (mixed GGUF, MXFP4 experts)
F16 KV full-attention layers = 12 * 131072 * 8 * 64 * 2(K,V) * 2 bytes = 3 GiB
SWA layers use the source's native 128-token window, not added sparsification
conservative all-24-layer full-context F16 allocation bound = 6 GiB
recurrent state = 0; proposer/MTP state = 0
W + conservative KV + published 2.7 GB workspace + 1 GiB reserve < 21 GiB
resident weights + KV + workspace therefore fit nominal 24 GiB without CPU offload
hot bytes/token = dense shared matrices + routed experts + attention KV reads
full-attention F16 K/V hot bytes/token up to 3 GiB at populated 128K
GPU time >= hot GPU bytes / measured GPU bandwidth
no planned PCIe/RAM/NVMe weight or KV movement per decode token
NVMe cold-load bytes >= W; proposer acceptance/cost not applicable
```

The guide's historical 3090 result (about 162 tok/s) is a short-generation
llama-bench result, not Pi throughput at populated 128K. Current setup does
not qualify coding quality, tool behavior, populated context or live speed.
Canonical publication gates and configuration deployment remain pending.

The selected-file Xet task completed with exit code 0 and exactly the pinned
12,109,566,624 bytes. As with the projector, cached tree integrity metadata
was unavailable to download status; the canonical publisher independently
validated size, GGUF v3 and pinned SHA-256 before transactional publication
under `${MODEL_ROOT}/gpt-oss-20b-mxfp4`. No draft model was downloaded.

The generic launcher now accepts an optional validated `reasoning_effort`
registry field, preserving xhigh for existing policies. The offline argument
gate passed for the new high/F16/128K policy with an arbitrary model ID and
confirmed every existing model's arguments unchanged. The initial gate helper
used PowerShell's automatic `$args` variable inside an assertion function;
renaming that local verification variable corrected the assertion scope.
The next attempt found the lifecycle process record absent. Reassessment
changed only the evidence gate to preserve optional file existence/hash;
absence is not interpreted as a measured service or GPU status. No inference
or runtime correction was needed.

The catalog adds `llama-3090/gpt-oss-20b-mxfp4`, native maximum context 131,072,
maximum output 32,768, text input and low/medium/high reasoning mapping.
Pi xhigh/max map to high, and off maps to the model's lowest reasoning level.
The model supplies `reasoning_effort` through chat-template kwargs and uses
temperature/top_p 1.0. Both project/home enabled-model lists include it;
existing defaults, provider records, other models and compaction stay intact.
`llama.sh config` now reads context/KV/slots from the selected registry policy.

Publication, launch preparation, shell syntax, JSON scope and prepared-file
gates passed. Preparation resolves the installed v0.6.0 executable, and the
process record's absence remained unchanged. No serving lifecycle operation,
API request, Pi invocation or active-session log inspection occurred. Coding
quality, live tools, actual populated context, GPU residency and throughput
remain unqualified. Evidence:
`out/gpt-oss-20b-gguf/deployment-verification.json`.

### File-default MTP policy blocked a scalar artifact, 2026-10-07

The user reported `model.sh restart mistral` failing after an early service
stop because `.env` declares `MODEL_MTP_DRAFT_DEPTH=6`. Read-only authenticated
artifact contracts confirmed Mistral NVFP4 declares zero MTP layers/ABI/depth,
while Qwen declares one layer, ABI 2 and stored depth four. This is a launcher
policy defect, not evidence of model execution or performance failure.

The environment-file default now applies only to an artifact with adjustable
sampled MTP. Otherwise the launcher reports that the file default is ignored
and omits the worker override. An explicit invocation override still requires
the capability and fails closed when unsupported. No model-name or
architecture branch was added. Existing artifact-specific KV, contexts,
weights, placement and cache policies are unchanged; compatible Qwen retains
the six-proposal override. Depth validation uses the effective context after
the artifact-limit clamp.

Restart no longer stops the service before calling the validated start path;
the existing stop remains after artifact/policy and environment checks.
Four offline launcher regressions passed using a file-only fake Windows
transport: scalar file-default omission, retained MTP-6, unsupported explicit
override without a stop, and depth rejection after context clamp without a
stop. Bash syntax and deployed source SHA-256 passed. Only `ops/model.sh` was
deployed; no native binary, serving task, model artifact or active process was
operated. No Pi/server logs, API request, inference, start, stop or restart
were used to verify the fix. A real Mistral startup remains unrun.
Evidence: `out/mistral-mtp-policy-fix/deployment.json`.

### Mistral Large 4 release screen and tokenizer warning, 2026-10-07

The user requests research on Mistral's recent launch and explanation of the
startup Transformers/PyTorch warning. This is research/read-only diagnosis;
no model, weights, format, context, fidelity, runtime, hardware, installed
dependencies or serving process is changed.

Primary sources confirm Mistral Large 4 public API preview launched October 6,
2026, with 1M context and weights promised by the end of October. The launch
blog describes 1T total/49B active parameters; the current model card lists
1.05T total/52B active and a 1.6B vision encoder. These are inconsistent exact
counts, so the screen reports approximately 1T total/50B active rather than
silently selecting one. Publisher-reported agentic coding scores are 61.7%
DeepSWE v1.1, 59.4% SWE-Atlas-QnA and 28.3% Terminal-Bench 4. These are distinct
benchmarks and cannot be compared directly with GPT-OSS SWE-bench Verified.
No preview weights/GGUF or local coding qualification is claimed.

Fail-fast capacity screen: uniform 4-bit payload for 1.00--1.05T parameters
would be 500--525 GB, and 2-bit payload would be 250--262.5 GB, before scales,
KV, encoder and workspaces. Both exceed the separate 3090 host's 24 GiB VRAM
and approximately 64 GiB RAM combined. MoE active parameter count does not
remove the need to store all expert weights. No local deployment, paging,
quantization implementation or throughput claim is proposed for this release.

Sources checked October 7:
[launch](https://mistral.ai/news/mistral-large-4/),
[model card](https://docs.mistral.ai/models/mistral-large-4-0),
[changelog](https://docs.mistral.ai/resources/changelogs).

Local source diagnosis: `Install-ServerEnvironment.ps1` imports `transformers`
in its version probe. `requirements/server.txt` deliberately installs tokenizer,
Jinja and Pillow dependencies without PyTorch; `expert_server.py` imports
AutoTokenizer and launches the independent native worker with subprocess.
The warning concerns unavailable Transformers neural-model execution in that
Python environment. It does not establish a CUDA failure, CPU inference or a
throughput regression in the native worker. Installing PyTorch is unnecessary
for this serving contract. No active Pi/server logs or runtime probes were used.


### Mistral NVFP4 optimization audit and MLA tile prerequisite, 2026-10-07

Scope: existing `mistral-small-4-119b-nvfp4`, one RTX 3090, configured
262,144 positions, native W4A4 NVFP4 weights and artifact-native **BF16**
compressed MLA state. No new model, top-k reduction, MTP, context reduction,
hot-state offload or additional host is introduced. Prior Qwen exact-F16
rejections do not establish this different compressed-cache capacity result.
Active Pi/service logs and requests are excluded, as are lifecycle operations.

Read-only authenticated artifact metadata identifies
`mistralai/Mistral-Small-4-119B-2603-NVFP4` at
`b1a9048590131d38491bd23a7c9f6ed0962f0358`. The artifact declares 36 layers,
32 attention heads, 128 experts/layer and top-4 routing, with zero MTP.
Actual pack masses, not parameter-count estimates:

```text
resident dense reservation        4,716,924,928 bytes = 4.392979 GiB
all dense pack bytes              5,574,676,480 bytes = 5.191822 GiB
all routed records               65,248,690,176 bytes = 60.767578 GiB
selected records/target token     2,039,021,568 bytes = 1.898987 GiB
routed device ceiling                                     12 GiB
routed host ceiling                                       48 GiB
MLA KV(N) = 36 * N * (256 + 64) * 2 bytes
MLA KV(262144)                                             5.625 GiB
recurrent state / MTP state                                    0
VRAM = 4.392979 + 12 + 5.625 + workspace + reserve GiB
     = 22.017979 + workspace + reserve GiB
```

This is a capacity screen, not observed residency. With a 1 GiB reserve,
workspaces must fit within the remaining 0.982021 GiB of nominal 24 GiB.
The existing routed ceiling is not raised. Configured/populated positions and
harness overhead remain separate. Even disjoint 48+12 GiB expert caches cannot
contain 60.767578 GiB; retained host copies can further duplicate residency.
Storage misses depend on the actual selected routes, not uniform assumptions.

At scalar decode one target call produces at most one accepted token; no MTP
proposer exists. Let `D_hot` be actual dense matrix traffic, `m_gpu` the fraction
of selected record bytes missing device cache and `m_disk` the fraction needing
storage. Necessary per-call bounds (not speed predictions):

```text
PCIe bytes >= m_gpu * 1.898987 GiB
PCIe time >= m_gpu * 1.898987 / 12.46 seconds
RAM bytes >= corresponding host-record copies and staging transfers
RAM time >= RAM_bytes / measured_RAM_bandwidth
NVMe bytes >= m_disk * 1.898987 GiB
NVMe time >= NVMe_bytes / measured_NVMe_bandwidth
unique GPU hot bytes >= D_hot + selected weights + MLA KV(N)
GPU time >= those bytes / effective_GPU_bandwidth
```

12.46 GiB/s is the historical pinned-link measurement, not a fresh measurement
for this service. RAM/NVMe bandwidth and miss fractions are unknown here.
At all-device-miss routing the PCIe floor alone is 152.4 ms/token. Even an
optimistic 15 tok/s requires `m_gpu < 0.438` before any other work. Theoretical
936 GB/s GPU bandwidth gives 6.45 ms for one unique 5.625 GiB KV scan alone.
Logical kernel requests are larger: the serial implementation reads latent
values twice and RoPE values once per head, approximately 324 GiB across
36 layers at 256K; cache reuse determines actual DRAM traffic.

Existing common-path optimizations include immutable expert paging, bounded
GPU/RAM caches, CUDA selection/aggregation graphs and GPU hierarchical top-k
for bounded sampling. Request retention is supported; RAM parking/persistence
requires a separate residual host budget, which a 48 GiB expert pool can
exhaust. Native NVFP4 is explicitly excluded from standard-FP4 grouped prefill.
Its dense projection uses scalar decoded-weight accumulation after native
activation rounding. These mechanisms need separate native kernels; toggling
standard-FP4 flags would change encoding/semantics.

MLA tile prerequisite: preserve all causal positions, BF16 cache bytes and
projection geometry; retain the serial short-context path; add no global
workspace or launch ABI changes. Eight warps score eight positions together,
retain their decoded latent values in shared memory and update stable online
softmax per tile. This reduces source-level barriers from 11 per position to
3 per eight positions and avoids a second global latent read. It changes F32
reduction order, so independent numerical validation is required. No end-to-end
speedup or bitwise equality is inferred from this count.

Weight quantization screen: preserving original one-byte block-16 scales while
packing experts into three bits would occupy about 47.250103 GiB before record
alignment, satisfying the 48 GiB host-pool prerequisite narrowly. A new
quantization ABI, encoder, executing provider and quality gates are required;
existing NVFP4 cannot be relabeled as Q3. Requantizing the NVFP4 checkpoint
compounds source quantization loss. Published BF16-derived imatrix GGUF sizes
provide separate candidates: IQ3_XXS 49.87 GB (46.45 GiB), Q2_K 42.07 GB
(39.18 GiB), IQ2_XXS 33.66 GB (31.35 GiB). Their published mean KLD values are
0.2253, 0.3165 and 0.6182 respectively, versus IQ4_XS 0.1326. These are
quantization statistics, not coding scores or 3090 throughput. GGUF IQ formats
require their actual decoders; no compatible QPack implementation is claimed.

KV quantization screen, preserving the 64 RoPE values as BF16 and adding one
F16 scale per 32 latent values:

```text
BF16 = 36*N*(256*2 + 64*2)                 -> 5.625000 GiB at 256K
Q8   = 36*N*(256 + 64*2 + (256/32)*2)      -> 3.515625 GiB at 256K
Q4   = 36*N*(256/2 + 64*2 + (256/32)*2)    -> 2.390625 GiB at 256K
```

These are candidate layouts, not implemented cache codecs. Current MLA rejects
non-native cache formats; the Qwen separate-K/V head codec cannot be reused by
changing an environment variable. Lossy MLA Q8/Q4 require their own ABI,
attention decoder and numerical/long-agent quality gates. Smaller KV alone
does not eliminate expert storage/PCIe misses or scalar NVFP4 prefill work.

Sources: [official checkpoint](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4),
[published imatrix precision ladder](https://huggingface.co/bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF).
Read-only artifact evidence: `out/mistral-optimization-audit/artifact-metadata.json`.


The MLA-only allocation audit finds unused standard-attention arenas:
512 rows * 32 heads * 8192 positions allocate 512 MiB scores and 256 MiB
probabilities, plus 30.125 MiB queries/K/V/accumulator/split state, totaling
798.125 MiB at this artifact geometry. MLA bypasses every consumer. The
candidate omits these arenas only if the declared operation set contains no
non-MLA full-attention operation; mixed sets retain them. This transfers the
accepted lazy-workspace mechanism without changing cache bytes or context.
Its MLA launch keeps null unused partial pointers, covered by the independent
kernel gate. Actual service residency remains unmeasured.

Grouped NVFP4 prefill prerequisite: reuse the existing four-selection work
layout and grouping by exact expert ID, preserve each selection's separately
rounded native gate/up/down activations and BF16 boundaries, leave scalar
routing order/aggregation unchanged. A work item of size `g<=4` shares each
weight decode across `g` selections instead of decoding it `g` times. Required
selection output bytes and per-selection activation traffic are unchanged;
weight instruction/load requests approach `1/g` for those groups, not a
claimed DRAM or wall-time speedup. No launch structure, wire format,
quantization ABI, route width, cache budget or model identifier is changed.
Independent NVFP4 reference gates and standard-FP4 regression gates must pass
before admitting this candidate. Grouping does not solve disk misses or
native dense projection GEMV throughput.


The completed native candidate passed 3/3 selected CTest targets after the clean
Windows Release/CUDA build. Eight MLA numerical fixtures include the exact
artifact query geometry (32 heads, latent rank 256, no-position width 64,
RoPE width 64, value width 128). Maximum relative L2 was 1.8148e-05 against
independent FP64 materialized K/V. Native NVFP4 grouped outputs were bit
identical to the scalar CUDA path for the tested 6/8/12-row mixed-expert,
masked and tail fixtures and passed the independent CPU NVFP4 reference.
Standard-FP4 grouping remained numerically unchanged. Canonical Python suites
passed after correcting a stale existing MTP metadata expectation (99 run,
one skipped). These gates do not qualify populated 256K or coding throughput.

The common candidate runner is staged on the 3090 host and selected in local
`.env` for the next user-owned start/restart. Existing running processes and
installed task arguments were not changed. Real-model chat, live populated
context, Pi, performance and lifecycle gates remain unrun under the user's
manual restart/test preference; no production-ready or universal chat pass is
claimed. Rollback runner selection is recorded in
`out/mistral-optimization-audit/deployment.json`. Weight/KV low-bit candidates
remain research only; no new checkpoint or lower-bit artifact was downloaded,
compiled or selected. Dense NVFP4 projection GEMV is still a distinct
remaining prefill optimization opportunity.

### Bounded pinned staging for retained RAM uploads, 2026-10-07

The user subsequently authorized deployment and Pi `hi` testing. The tiled
MLA/grouped NVFP4 candidate returned correct greetings through direct chat
and Pi at xhigh. Direct cold chat: 541 prompt / 69 generated tokens,
60.468 s prefill, 85.828 s total, 70,360,403,968 storage-read bytes and
125,915,841,000 H2D bytes. The following Pi request: 486 prompt / 54 generated
tokens, 36.985 s prefill, 54.578 s total, 28,914,458,624 read bytes and
114,336,396,600 H2D bytes; cumulative upload wait was 14.453 s. Its
6,035 retained-RAM hits bypassed the pinned staging pool and supplied pageable
bytes directly to `cudaMemcpyAsync`, allowing driver-side synchronous staging.
These short greetings establish wiring, not coding or populated-256K gates.

Candidate prerequisites and fixed invariants:

```text
VRAM(N) = 4.393 GiB dense + <=12 GiB experts + 23040*N bytes KV + workspaces
RAM cache <=48 GiB; existing staging =32*14,159,872 bytes =432.125 MiB
additional reserved CPU/GPU bytes =0; recurrent/draft/MTP state =0
per scalar target call: 1 accepted token, 1.899 GiB selected record bytes
per RAM upload: CPU copy S pageable -> pinned; GPU upload payload <=S
RAM traffic >=2*S; PCIe traffic and SSD miss traffic remain unchanged
t_explicit >=2*S/B_RAM + payload/B_PCIe + queue overhead
t_pageable >=driver staging + payload/B_PCIe + implicit synchronization
```

GPU dense/MLA/expert execution and NVMe bounds remain those in the preceding
audit; this change does not compress weights/KV or claim to remove hot PCIe
traffic. RAM bandwidth is unmeasured. The quantitative prerequisite is
positive measured net latency/transfer-wait improvement on the same direct
`hi` then Pi `hi` sequence, with identical token counts and execution formats;
driver synchronization avoided must exceed explicit copy/queue costs.
No improvement is inferred from the historical pinned 12.46 GiB/s alone.
Stop/reassess if the complete service gate fails that prerequisite.

Reuse the existing fixed pinned pool only for pageable retained RAM hits.
Hold its slot until upload completion, keep bytes unchanged, honor demand
reservations and asynchronous cancellation, and preserve direct access for
already pinned host banks or non-pinned pools. Native ownership/byte/failure
tests and actual service gates must qualify this common cache change.
The added pool query changes no object layout, launch/protocol structure,
quantization ABI or artifact/wire version; declaration, definition and cache
consumer are audited together. A clean Windows Release/CUDA build is required.

Pinned staging passed 4/4 native gates and returned identical greeting tokens.
The comparable direct/Pi sequence measured 85.109/53.000 s versus
85.828/54.578 s previously. Pi prefill was 35.656 versus 36.985 s, but summed
upload wait increased to 17.013 s from 14.453 s, and storage bytes were
unchanged. This single pair establishes only a modest observed wall-time
difference, not a robust link-bandwidth or general coding-speed improvement.

### Expert cache union under RAM pressure, 2026-10-07

Scope and formats remain the same single-3090 Mistral artifact, NVFP4 W4A4,
BF16 MLA, configured 256K, no MTP, <=12 GiB device and <=48 GiB host experts.
Cache eviction only removes immutable copies; it does not change routes,
attention support or numerical weights. No family-specific runtime branch.

```text
VRAM(N) = 4.393 GiB dense + <=12 GiB experts + 23040*N KV + workspaces
RAM = <=48 GiB expert copies + existing 432.125 MiB pinned staging + other state
unique cached expert bytes U = host bytes + device bytes - duplicated bytes D
ideal U <=60 GiB <60.7676 GiB stored expert pool
removing a host duplicate increases union headroom by its record size
selected payload/target token =1.899 GiB, accepted tokens/target call =1
PCIe lower bound =GPU misses*payload/B_PCIe (unchanged)
NVMe lower bound =storage misses*record_size/B_NVMe (must decrease)
RAM copy bound =2*RAM_upload_bytes/B_RAM (unchanged per upload)
GPU dense, expert and retained-MLA bounds remain unchanged; MTP cost =0
```

The original RAM victim ranking ignores whether another cache tier still owns
the expert. With a hot device pool it can evict the sole remaining host copy
of a cooler expert while preserving duplicated hot records. Under RAM-only
pressure, prioritize unprotected device-backed host copies, then retain the
existing temperature, quota, recency and key ranking within each class.
References, pending waiters, protected RAM classes and device replacement
temperature constraints remain binding. Device eviction remains unchanged.
Removing duplicates may increase later reads if those device records are soon
evicted; the capacity equation alone is insufficient to claim a speedup.

Fail-fast prerequisite: the same direct-then-Pi greeting sequence must preserve
exact output/token counts, reduce Pi storage reads below 28,914,458,624 bytes
and reduce Pi wall time by at least 10% versus 54.578 s. Otherwise reject this
victim-ranking change rather than increasing memory caps or weakening fidelity.
Native tests must verify that a hot device-backed duplicate releases RAM while
a cooler sole host copy remains available without rereading storage.

**Rejected after the complete real gate.** Four native CTest targets passed,
and source-independent greeting text/token counts remained identical. Direct
chat measured 60.625 s prefill / 91.078 s total versus 59.859 / 85.109 s for
the pinned candidate; storage reads increased to 75,939,393,536 bytes. Pi
measured 27.985 s prefill / 50.657 s total, 22,259,318,784 storage bytes and
115,044,186,600 H2D bytes. Pi storage decreased 23.0% and prefill improved,
but generation wall increased from 17.344 to 22.672 s: device victims whose
RAM duplicates had been removed needed storage again. Pi total improvement
versus the original 54.578 s was only 7.2%, below the declared 10% gate.
The full direct+Pi sequence regressed from 138.109 to 141.735 s versus pinned
staging alone. Do not promote this policy using prefill or union capacity as
a substitute for decode/complete-loop evidence. Restore the exact prior
clean-built pinned candidate source and executable; retain this negative
result and isolated rejected build as evidence. No memory cap is raised.

### Parallel context splits for resident compressed MLA, 2026-10-07

The user reports that real Pi use remains 0.1 tok/s and explicitly requests
further improvement. The short greeting gate does not explain this real
context. Preserve the existing Mistral artifact, native NVFP4 W4A4, BF16
latent KV, all causal positions, configured 262K, xhigh, one RTX 3090 and
12/48 GiB expert cache ceilings. Runtime scope is the generic compressed-MLA
capability, not a model-name branch. No MTP or another model is introduced.

Code inspection shows that scalar decode launches `heads*rows` blocks:
32 blocks at artifact geometry. Each block serially scans eight-token tiles,
32,768 iterations at a populated 262K. This is an identified scaling risk,
not proof of the user's uninspected request bottleneck. The earlier rejected
staged exact-F16/Qwen attention proposal had a different capacity inequality;
here the entire BF16 latent cache fits the existing GPU-resident page organ.

```text
GPU =4.393 GiB dense + <=12 GiB experts +23040*N bytes KV +workspace/reserve
KV(262144) =5.625 GiB; recurrent/draft/MTP state =0
S =ceil(N/1024); split partial floats =rows*heads*S*(rank+2)
one-row 262K scratch =32*256*258*4 =8.0625 MiB
eight-row maximum scratch =64.5 MiB; reused across the 36 layers
old scalar grid =32 blocks, 32768 iterations/block
new scalar grid =32*256 =8192 blocks, <=128 iterations/block
one-row scratch GPU write+read ~=16.125 MiB/layer
GPU unique retained KV/dense/expert bounds and selected bytes remain unchanged
GPU logical KV reads/token =heads*36*N*(256+64)*2 =180 GiB at 262K
GPU split scratch write+read/token ~=36*16.125 MiB =580.5 MiB at 262K
GPU unique KV is 5.625 GiB; inter-head L2 reuse may reduce DRAM vs logical reads
GPU theoretical bound >=(5.625 +4.393 +1.899) GiB / theoretical GPU GiB/s
actual GPU DRAM/L2 traffic is unmeasured; logical reads are not DRAM counters
PCIe >=GPU_miss_fraction*1.899 GiB per accepted token; no scratch crosses PCIe
RAM >=retained record copy/staging bytes; NVMe >=storage-miss record bytes
accepted tokens/target call =1; proposer/verifier state and cost =0
```

Allocate bounded dedicated MLA partial maxima/sums/latent vectors only for
up to eight rows. Larger prefill batches retain the already parallel row/head
grid and existing tiled path. Each context split computes stable unnormalized
softmax mass and latent sum; a second GPU kernel combines them using the
global maximum, then applies the original value projection and BF16 output
rounding once. No position, expert or K/V precision is removed. F32 reduction
order changes and requires independent materialized-K/V and full-length
closed-form numerical oracles.

Fail-fast prerequisites before service promotion: >=2x attention-kernel
speedup against the existing eight-position path at actual artifact geometry
for 64K and 256K, numerical gates including split/page/causal boundaries,
bounded scratch, clean Release/CUDA build, affected direct/Pi greeting and
Qwen regression on the exact binary. These attention gates cannot establish
end-to-end coding throughput or prove that 0.1 tok/s has been resolved.
Public launch layout, artifact/wire versions and quantization ABI remain
unchanged; audit the header, provider aggregate initializer and native tests
for the revised optional scratch contract together.


Offline prerequisite passed on the clean Windows MSVC Release/CUDA 13.4,
SM86 build. The 277-file source snapshot matches local native sources. All
five selected CTest targets pass: cache ownership, standard-FP4 grouping,
native NVFP4, independent MLA numerical cases and the long-context gate.
Thirteen materialized-K/V cases cover actual geometry, split/page boundaries,
sharp softmax, eight-row splits, nine-row fallback and absent scratch; worst
relative L2 is 0.000437592 against the 0.006 threshold. Scratch canaries pass.
The full-length gate additionally checks both paths against an independent
FP64 mean for uniform scores; it does not compare two copies of the decoder.

| Populated native fixture | Previous tiled attention, ms | Split attention, ms | Speedup |
|---|---:|---:|---:|
| 8192 | 4.10105 | 0.594837 | 6.8944x |
| 65536 | 32.5001 | 2.95734 | 10.9896x |
| 262144 | 124.515 | 9.67058 | 12.8756x |

These are GPU-event means over three timed calls following warm-up at one
layer's actual head/rank/value geometry. No runtime prompt or expert paging is
included. The allocated single-layer 256K fixture is not a whole-model VRAM
or coding gate. Candidate runner SHA256 is
`aa3bd5b2d659c314254565c4e6b61395f1f70f65fbd7e41f4f67ada9e0bc4566`.
Canonical Python suites pass 99 tests, with one skipped. Service/Pi/Qwen and
cleanup results are recorded separately in the benchmark ledger after the
exact binary executes them.


The actual service gates passed on the hash above: direct Mistral hi, Pi hi,
Pi with 4408 populated prompt tokens (split decode exercised), then Qwen
model.sh chat hi. The latter produces 30 tokens including 18 reasoning,
first visible 1.959 s and total 2.286 s. Each tested model stop removes the
native worker/listener and returns GPU residency to 336 MiB. The 4408-token
Pi request has 166.468 s prefill and 205.718 s total for 85 generated /69
reasoning /16 visible tokens. Its useful end-to-end rate is only 0.078 tok/s;
this is a remaining negative service result, not evidence that the reported
0.1 tok/s has been resolved. No old-binary long service comparison or real
coding quality/throughput gate was run. Retain the independently qualified
attention gain while leaving whole-model performance unqualified.


Final readiness and running-hash verification on 2026-10-08 confirms the
qualified candidate is selected and Mistral is running with context 262144,
BF16 latent KV and MTP zero. No native source changed after the clean build;
all 277 snapshotted source files still match. The preceding pinned candidate
remains available for rollback. This is a qualified component optimization,
not completion of interactive Mistral coding performance.


### Independent Strata service qualification, 2026-10-21

The user explicitly authorizes installing Strata as an independent service,
analogous to the existing llama.cpp operations, using canonical Xet downloads.
This is an external runtime exception, not a new artifact-VM family runner.
QPack and llama.cpp source/configurations/defaults are preserved.

Inspected source: Niko1221/Strata commit
9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4; latest stable release v0.2.1,
Terminal build 0.1.46.3, signed CUDA13 engine 0.1.46. Hardware readback:
RTX3090 24576 MiB, Ryzen5 5600/6 cores, 68641001472 physical RAM bytes,
516550348800 free D: bytes, driver 617.14. Initial GPU residency was 309 MiB.

Selected target: ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF IQ3_S,
revision 99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d. Both GGUF v3 shards total
83743737120 bytes and passed the canonical downloader's SHA256 integrity gate.
This is the Flash-Next MoE derivative, not the existing dense Swift 27B.
The higher bit Unsloth/NVFP4 alternatives need more host RAM or SSD expert
traffic; Coder removes half the experts. IQ3_S is the selected practical
quality-first option for 64 GiB RAM, not a universal capability ranking.

Sources: upstream docs/MODELS.md, docs/DETAILS.md, model cards for
Qwen/Qwen3.8-Flash-Next, ukisai/Swift1.5-Qwen3.8-Flash-Next and the selected
quant repository. Swift's reported BF16 LCBv6 90.39 and Terminal-Bench2.1
69.66 are not IQ3_S or local-hardware coding scores. IQ3_S code-text KLD
0.069016 versus IQ3_XXS 0.118707 measures distribution drift, not task accuracy.
Upstream 3090 speed tables are estimates; other card measurements are not
promoted to this host's throughput.

Initial serving invariants: one GPU, configured context 262144, xhigh, one
slot, existing IQ3_S tensor formats, FP16 KV, 32768 resident KV cells with
host backing, full top10 routed contributions (STRATA_ROUTE_TAIL_SKIP=0),
a maximum 12 GiB of routed slots and 1024 MiB VRAM reserve. No new model or
KV quantizer is implemented. Offline preparation uses the installed upstream
packer, an exact packed-layout copy and a generated tokenizer. Speculation
uses the existing one-layer Qwen MTP proposer, not a second target model.

Preflight arithmetic (exact pack geometry/placement must be read back):

```text
Disk target =83.743737120 GB; PLE table remains on SSD
upstream estimated routed payload =50.3 GB, across 48*512 experts
maximum routed slots =floor(12 GiB / aligned maximum packed expert bytes)
GPU =dense matrices +routers/shared experts +recurrent state +resident KV
     +MTP/draft head +bounded routed slots +workspace +1 GiB reserve
RAM =routed experts outside GPU +full FP16 KV backing +CPU/staging workspace
FP16 QSA K/V payload =12 *N *2048 bytes =6 GiB at N=262144
resident FP16 payload at32768 =0.75 GiB, plus separate index/scales/state
MTP prepared payload =824125604 bytes including dense/vocabulary/index text
routed logical bytes/token approximately =50.3 GB*(10/512) =0.9824 GB
bytes/verify call depend on actual route union across the speculative window
CPU RAM floor >=CPU-missed routed bytes / measured host bandwidth
PCIe floor >=GPU-staged routed bytes plus adaptive swaps / measured link rate
NVMe floor >=selected PLE row bytes and any actual expert misses / storage rate
accepted tokens/cycle and proposer/verifier cost require real request telemetry
```

These bounds are not throughput promises. Hot dense projections stay on GPU;
CPU computes cold routed experts in RAM. This differs from the rejected hot
dense/active-KV full-scan streaming proposal. No lossless equivalence to the
BF16 checkpoint or populated262K quality/capacity is claimed by startup.
The serving gate requires real greeting, a bounded coding/tool transport
request, resident-memory readback and owned-process/GPU cleanup on this exact
installed engine, followed by the final ready deployment.

Only target GGUFs are downloaded from the selected repository. MTP files come
through the same canonical Xet scripts from the pinned Maximilian228
Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata repository: mtp/dense.bin, dense.txt,
experts.bin and draft_vocab.bin (824125604 bytes). No NVFP4 target weights or
BF16 checkpoint shards are requested. The upstream default range-fetch
MTP downloader is not used. Large files use LFS SHA256; the small dense.txt
uses its Git blob SHA1. The existing generic selected-file status only checks
LFS metadata, so this small asset requires the independent publication check.

Publication independently verified all four MTP assets and both target shards.
The installed packer generated a 46.84 GiB expert pack. Its reported 96 exact
layout conversions are packer diagnostics, not an independent numerical oracle
or BF16 fidelity result. Actual startup readback gives 11.98 GiB routed GPU
cache, 37.93 GiB pinned expert RAM and 13.4 GB/s measured PCIe bandwidth.
Only 3.9 GiB Windows commit remained during startup; populated long-context
capacity and throughput remain open gates. No pagefile or cache-budget change
was made to conceal that limit.

The first startup rejected configuration reasoning_effort=xhigh: the server's
configuration enum accepts high, whose HIGH_EFFORT instruction is the native
xhigh text. Requests independently accept xhigh. The prepared configuration
uses high and actual API gates submit xhigh.

Direct detached startup reached generation_ready, but Windows OpenSSH ended
the owner/child processes when SSH closed. Keeping SSH open preserved both
processes and passed local greeting/tool requests. The lifecycle was therefore
moved to the user-authorized independent QuantumLLM-Strata scheduled task,
whose worker remains alive throughout the managed runtime lifetime. Readback
after SSH disconnect confirms the owner, child and API listener survive.

The first remote API gate timed out despite a working local listener. Windows
had generated an inbound TCP block for the versioned Strata Python executable;
this overrides allow rules and resolves through terminal/current's junction.
Operations now resolve the actual executable, disable only its inbound TCP
block and install a TCP/API-port allow rule scoped to the host's WireGuard
address and 10.10.88.0/24 peers. The remote greeting/tool retest passes.
Scheduled-task stop leaves zero native workers/listeners and 263 MiB GPU use.

Subsequent startup exposed a launcher defect when no enabled inbound block
rules remain: Get-NetFirewallRule's filtered CIM query raises
CmdletizationQuery_NotFound, terminating the strict-mode launcher before
model startup. The correction enumerates rules with ErrorAction Stop and
filters the returned objects locally. The real host returns zero matches
without error; actual enumeration/access failures remain fatal.
After deploying this correction, the user's exact `./ops/strata.sh start`
command exits zero with generation_ready=true. An authenticated remote
GET /v1/models returns HTTP200 and swift-flash-next-iq3-s after SSH closes.

## AREX-2 single-3090 documentary screen, 2026-11-07

Scope: user-requested feasibility analysis of BAAI/AREX-2; one RTX 3090
24 GiB on the existing host, text-only, one slot, xhigh, scalar target decode
without a draft. Context cases below mean populated positions, not a short
prompt inside a configured large window. No model payload download, remote
host operation, inference, runtime edit or serving-policy change was performed.
Weight/KV quantization alternatives are screening candidates, not new fidelity
approval or qualification of the universal artifact VM.

The official card describes a dense Qwen3.8-27B fine-tune. Source config uses
the Qwen3_5 encoding: 64 layers, 16 full-attention layers, 48 recurrent layers,
4 KV heads, 256 head width, recurrent 48 value heads of width 128 and key
width 128, and 262144 maximum positions. No routed experts or routers exist.
The official safetensors index reports 54,713,457,120 payload bytes and no MTP
tensors, despite config declaring mtp_num_hidden_layers=1. Do not infer usable
MTP from that field or reuse another fine-tune's acceptance/speed result.
[Card](https://huggingface.co/BAAI/AREX-2),
[config](https://huggingface.co/BAAI/AREX-2/blob/main/config.json),
[index](https://huggingface.co/BAAI/AREX-2/blob/main/model.safetensors.index.json).

Published evaluation reports Frontier-CS 70.7, MLE-Lite Any Medal 81.8,
BrowseComp 84.0 and GAIA 92.2. Frontier-CS permits five hours/task; MLE-Lite
permits twelve and includes skills. These are upstream agent-environment
results, not quantized Pi repository repair, Romanian quality, or local speed.
Self-improvement means iterative solution refinement at inference time, not
online weight updates. No controlled AREX-versus-Swift local result or
reproducible populated-256K single-3090 coding measurement was established.
[Paper and protocols](https://arxiv.org/html/2609.38288),
[evaluation code](https://github.com/VectorSpaceLab/AREX-2).

Publisher tree metadata gives these exact file masses; GiB = bytes / 2^30:

| Bartowski artifact | Bytes | GiB |
|---|---:|---:|
| IQ4_XS | 15,236,967,520 | 14.191 |
| Q4_K_M | 17,203,416,160 | 16.022 |
| Q5_K_M | 20,684,893,280 | 19.264 |

These whole-file allowances are not measured VRAM or per-token traffic.
Dense projections, recurrent weights, embeddings and output head have not
been separately measured in the quantized artifact. Input embeddings gather
selected rows; dense matrices and output head are hot per target call.
Vision is excluded; the optional F16 projector is another 927,607,264 bytes
plus image-dependent encoder workspaces. The GGUF publisher explicitly lists
no speculative decoding.
[GGUF publisher](https://huggingface.co/bartowski/BAAI_AREX-2-GGUF),
[file metadata](https://huggingface.co/api/models/bartowski/BAAI_AREX-2-GGUF/tree/main?recursive=true&limit=1000).

Capacity prerequisite:

```text
M = W_dense/recurrent + W_embeddings + W_head + W_norms
    + S_recurrent + S_convolution + KV_target(C)
    + W_draft + KV_draft + S_rollback + workspaces + 1 GiB reserve
W_routed = W_router = W_draft = KV_draft = S_rollback = 0 in this screen
core S_recurrent = 48*48*128*128*4 = 144 MiB
KV_target(C) = 2*16*4*256*C*bytes_per_value
F16 bytes_per_value=2; Q8_0=34/32; Q4_0=18/32
```

For Bartowski Q4_K_M, the following subtotals include the whole file,
core recurrent state and 1 GiB reserve, but exclude convolution state,
workspaces, other allocations and vision. They are capacity arithmetic,
not runtime allocation measurements:

| Populated context | F16 KV / subtotal GiB | Q8_0 KV / subtotal GiB | Q4_0 KV / subtotal GiB |
|---|---:|---:|---:|
| 32768 | 2 / 19.163 | 1.0625 / 18.225 | 0.5625 / 17.726 |
| 65536 | 4 / 21.163 | 2.125 / 19.288 | 1.125 / 18.288 |
| 131072 | 8 / 25.163 | 4.25 / 21.413 | 2.25 / 19.413 |
| 262144 | 16 / 33.163 | 8.5 / 25.663 | 4.5 / 21.663 |

Stop the Q4_K_M fully resident 256K F16/Q8 proposals at capacity failure.
Q4_K_M + 64K F16 or +256K Q4_0 passes only the preliminary subtotal test.
The latter leaves 2.337 GiB before uncounted allocations and requires explicit
KV-fidelity approval and real long-context quality/residency gates. Q4_0 is
not the artifact VM's Q4H ABI. IQ4_XS saves another 1.831 GiB but cannot
inherit the original checkpoint's benchmark quality without validation.

Traffic, bandwidth and speculative prerequisites remain:

```text
B_target_call >= B_hot_dense/recurrent/head + KV_target(C)
                 + B_recurrent_state + B_intermediates
A_scalar=1; B_per_accepted=B_target_call; proposer/verifier extra cost=0
T_call >= max(B_GPU/936e9, B_RAM/BW_RAM,
              B_PCIe/(12.46*2^30), B_NVMe/BW_NVMe)
15 useful tok/s requires complete useful-token latency <=66.667 ms
T_task = T_cold_or_reused_prefill + N_generated/r_useful + T_tools/retries
If speculation is added:
T_cycle=T_proposer(D)+T_verify(D+1)+T_rollback/sampling
r_useful=A_D/T_cycle; B_per_accepted=B_cycle/A_D; 1<=A_D<=D+1
```

936 GB/s is theoretical GPU bandwidth; 12.46 GiB/s is historical host-link
evidence. Actual GPU/RAM/NVMe rates and hot quantized role masses are unknown.
The fully resident candidate plans zero per-token hot-state RAM/PCIe/NVMe
traffic; NVMe remains startup storage. Transferring the full 16 GiB F16 KV
over that historical PCIe link alone would cost at least 1.284 s/call.
Hot dense/KV offload therefore does not rescue the throughput proposal.
No positive generated-token rate, cold/reused prefill time, TTFT or tool-loop
wall time is claimed. MTP has a separate community adapted W8A16 release,
with offline acceptance evidence and a nonstandard draft-vocabulary ABI;
it is not a drop-in MTP speed result for this GGUF candidate.
[Community MTP limitations](https://huggingface.co/numsu/AREX-2-27B-INT8-W8A16-MTP).

Candidate qualification order: source/reference fidelity, native numerical
gate, direct chat, streaming/tools, then real Pi coding with separately
measured cold/reused prefill, TTFT, reasoning/visible output, decode,
tool-loop wall time, residency and cleanup. Existing llama.cpp operations
are the practical evaluation candidate; adding AREX to a registry or claiming
universal artifact-VM support is outside this documentary analysis.

### AREX-2 deployment, 2026-11-07

The user requested Xet download, deployment, home Pi selection and a single
`hi` prompt. The canonical download worker transferred the pinned Bartowski
Q4_K_M file above with exit code zero. The status reader lacked its cached
tree metadata; Publish-LlamaModel.ps1 independently verified the pinned file
size, GGUF v3 header and SHA256 before transactional publication.

Artifact: `${MODEL_ROOT}/arex-2-q4-k-m-256k`. Aliases: `arex`, `arex-2`,
`arex-2-q4-k-m`. Existing llama.cpp v0.6.0, one slot, 262144 context,
Q4_0 K/V, all GPU layers, fitting off, xhigh, 32768 maximum output tokens,
no draft or vision projector. Repository `.env` already has
MODEL_MAX_CONTEXT=262144. Home Pi defaults now select
`llama-3090/arex-2-q4-k-m:xhigh`; other providers and compaction are preserved.
Private backups and operation evidence: `out/arex-20261107/`.

`./ops/llama.sh pi arex --no-session --mode json -p hi` exited zero and
returned `Salut! Cu ce te pot ajuta în proiectul quantum-llm?`. Pi issued
one incidental `bash` call, `echo "OK"`, before the greeting. Stop removed
the AREX process and process record; GPU use returned to 288 MiB.
