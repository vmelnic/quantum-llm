# Benchmarks and evidence

Status: canonical measurement ledger, 2026-09-20.

## Reporting rules

- TTFT is server time to first generated token unless marked first-visible.
- After-first rate excludes TTFT and the first token.
- Prompt, reasoning, visible and total generated tokens are distinct.
- Configured, populated and harness-provided context are distinct.
- Cold, warm, prefix-reused, novel-route and settled-route runs are not
  interchangeable.
- A numerical gate or microbenchmark is not service throughput.

## Validation baseline

The latest native changed-code gate used the Windows Release/CUDA build
(MSVC 19.44, CUDA 12.1), built `--clean-first --parallel 22`. Windows CTest
passed 6/6 and the canonical compiler/server suite ran 118 tests successfully.
The later Python-only artifact migration and populated-window driver changes
passed the expanded local 122-test suite with nine optional NumPy skips.
Python byte compilation, shell syntax and `git diff --check` passed. The
Windows binary is authoritative.

## Service results

### Matched direct `hi`

The original six rows use build `ea1106d`; Qwen Abliterated uses build
`e77f8e1`. Each used a fresh model start and `--thinking off`; model startup is
not included. Every response was coherent and ended normally.

| Model | KV | Prompt/output/reasoning | First-visible | Wall | End-to-end | After-first |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | K1 | 13/10/0 | 0.827 s | 1.118 s | 8.95 tok/s | 30.92 tok/s |
| Qwen3.8-27B Abliterated | K1 | 13/10/0 | 0.910 s | 1.265 s | 7.90 tok/s | 25.33 tok/s |
| Qwen3.8-Flash-Next | F16 | 13/10/0 | 27.642 s | 43.028 s | 0.23 tok/s | 0.58 tok/s |
| Mistral Small 4 | native BF16 MLA | 541/13/0 | 271.256 s | 281.948 s | 0.05 tok/s | 1.12 tok/s |
| Muse-Glimmer | F16 global/sliding | 57/53/34 | 2.559 s | 2.866 s | 18.49 tok/s | not separable |
| Ornith | F16 | 13/10/0 | 12.977 s | 22.342 s | 0.45 tok/s | 0.96 tok/s |
| DeepSeek | native | 5/10/0 | 12.953 s | 29.181 s | 0.34 tok/s | 0.55 tok/s |

The exact hierarchical top-k change was measured separately on 2026-09-17
after a clean build. Qwen returned the same coherent `hi` response at
13/10/0 tokens, 0.689 s first-visible, 0.890 s wall and 44.86 tok/s
after-first. The comparable prior Qwen row was 30.92 tok/s. The mandatory
DeepSeek regression returned a coherent 5/10/0 response at 3.190 s
first-visible, 9.244 s wall and 1.49 tok/s after-first.

Muse emitted hidden reasoning despite the off request, which is current
artifact-template behavior. These rows prove service wiring, not coding or
maximum-context performance.

### Minimal Pi wiring

Command: `./ops/pi.sh <alias> --no-context-files --no-tools --no-session -p hi`.
One RTX 3090, 12 GiB routed-VRAM ceiling, default `xhigh`, 2026-09-02.

| Alias | Prefill | Generated | TTFT | Wall | Result |
|---|---:|---:|---:|---:|---|
| `qwen` | 5 | 33 | 0.625 s | 1.938 s | pass |
| `qwen-abliterated` | not captured | not captured | not captured | 5.495 s client | pass |
| `qwen-flash` | 477 | 67 | 41.829 s | 53.688 s | pass, slow |
| `mistral` | 439 | 67 | 54.797 s | 77.312 s | pass, slow |
| `muse` | 422 | 65 | 2.297 s | 4.297 s | pass |
| `ornith` | 439 | 34 | 3.703 s | 5.390 s | pass |
| `deepseek` | 504 | 256 | 81.985 s | 310.672 s | pass, impractical |

Pi may add system instructions even with project context/tools disabled. The
abliterated row was run on 2026-09-15 and only client wall time was captured.
This table qualifies API/template/generation wiring only.

#### Clean-build isolated Pi regression, 2026-09-20

Immediately after the canonical clean Windows Release/CUDA build passed CTest
6/6 and the canonical Python suite 119/119, six published artifacts were
started sequentially on the RTX 3090. Each start used its declared KV codec,
disabled server-side session retention and reported `ready=true`. The Pi
command added `--no-extensions --no-skills --no-prompt-templates` to the
minimal gate above, so no project context, tool, extension, skill, prompt
template or prior session was available. The values below are client wall
time, not service throughput measurements.

| Alias | Declared KV | Client wall | Result |
|---|---|---:|---|
| `qwen` | K1 | 2.75 s | pass, coherent text |
| `qwen-abliterated` | K1 | 5.42 s | pass, coherent text |
| `qwen-flash` | F16 | 62.11 s | pass, coherent text |
| `mistral` | artifact-native | 78.44 s | pass, coherent text |
| `muse` | F16 | 3.88 s | pass, coherent text |
| `ornith` | F16 | 7.12 s | pass, coherent text |

DeepSeek was intentionally not run in this regression at the user's request;
its earlier result above remains historical evidence only. After the six
gates, the common task was stopped and `/ready` was unreachable with no active
model advertised.

### Coding harness

| Model/mode | Measured result |
|---|---|
| Qwen F16, effective `medium` | Todo fixture passed 3/3 in 503.240 s; about 17.8K final prompt tokens |
| Qwen K1, `xhigh` | Todo fixture passed 3/3 in 979.17 s; 13 assistant turns, 17 tools, one 15,082-token reasoning response |
| Qwen K1, `off` | changed code but finished 1/3 after 569 s |
| Qwen Flash, corrected `medium` | valid tool calls, zero edits; action stall exceeded 1,000 s |
| Qwen Flash, `off` | zero edits and invalid generated paths |
| Mistral | 14 valid tool calls, zero edits at 30-minute cutoff |
| Ornith K1+fit, `xhigh` | stopped at 1,832.80 s; 2/3 tests passed |
| Muse | not qualified |
| DeepSeek | minimal Pi only; project prefill/recovery not qualified |

For Ornith, the largest retained state was 34,664 tokens. After its cold
request, later turns had zero SSD misses, but the complete run still incurred
74,493 routed-cache evictions and 139,467,325,440 host-to-device bytes. It
failed the cursor-termination assertion; this was a semantic and latency
failure, not merely a cutoff.

## Artifact and numerical evidence

| Artifact | Validated evidence |
|---|---|
| Qwen3.8-27B | 14,775,390,208-byte QPack, 1,199 records; FP4 relative L2 0.121706, cosine 0.992577; bounded official/service corpus 4/4 |
| Qwen3.8-27B Abliterated | 14,775,390,208-byte QPack, 1,199 records; independent FP4 relative L2 0.121710, cosine 0.992577; direct and minimal-Pi `hi` passed, coding and maximum context unqualified |
| Qwen3.8-Flash-Next | 1,562 dense + 24,576 expert records, 95,915,634,688 bytes; exact QSA resident/staged/selected CUDA parity, max error 0 |
| Mistral Small 4 | 70,801,904,048 source tensor bytes; native NVFP4 sidecars and BF16 organs validated |
| Muse-Glimmer | 15,832,002,560-byte QPack; aggregate cosine 0.9999981, FP4 cosine 0.9930505 |
| Ornith | 19,229,777,920-byte pack; aggregate relative L2 0.006980/cosine 0.999976, FP4 0.118762/0.992932 |
| DeepSeek | 11,008 compact expert records, 147,169,738,752 routed bytes; independent source, attention, route, expert and I/O oracles |

Source reconstruction is necessary but does not replace provider and service
execution.

## Qwen maximum-context evidence

### Capacity

```text
hot target weights             13,625,700,352 bytes = 12.690 GiB
exact F16 KV at 262,144        17,179,869,184 bytes = 16.000 GiB
subtotal                       30,805,569,536 bytes
RTX 3090 physical              25,769,803,776 bytes
```

At the measured 12.46 GiB/s pinned link, a full 16 GiB host-KV scan has a
1.284-second lower bound per scalar decode call. This is why ordinary host
offload recovers capacity but not 15 tok/s.

K1 stores block-32 FP4 values and FP4 keys with one FP16 outlier correction per
key block. Qwen target KV is 4.7500 GiB at 262,144 positions; target plus MTP
pages is 5.015625 GiB. K1 is lossy and is not called exact F16.

### Numerical and kernel gates

| K1 gate | Result |
|---|---:|
| independent layout byte mismatch | 0 |
| CUDA vs independent CPU attention max abs | 3.72529e-09 |
| K1 vs F16 synthetic max abs | 0.00810682 |
| ordinary FP4 vs F16 synthetic max abs | 0.00841153 |
| 262K attention, 16-layer projection | 29.0079 ms; repeat 29.0161 ms |
| effective encoded traffic | 175.824 GB/s |

The first service allocation omitted the MTP term and failed closed. The
correct 5,136 MiB page pool and later workspace correction produced:

| Populated K1 run | Prompt/output | TTFT/total | Post-first | Free VRAM | Cleanup |
|---|---:|---:|---:|---:|---|
| corrected capacity | 262,016/128 | 1,277.250/1,286.094 s | 14.36 tok/s | 105 MiB | all logical pages released |
| corrected workspace | 262,016/128 | 1,277.750/1,286.578 s | 14.39 tok/s | 1,095 MiB | all logical pages released |
| exact hierarchical top-k | 262,016/128 | 536.469/543.907 s | 17.07 tok/s | not sampled | request completed; worker stopped and cleared |

Lazy one-row logits and request-scoped dense staging released about 994 MiB
without changing arithmetic.

### Prefill and matched full-context request

The compact path uses vendored FlashAttention, a value-major recurrent update
and 1,024-row dense batches. Native maximum absolute errors were
`1.49e-07` recurrent, `6.09e-06` attention and `9.59e-05` complete prefill.

| 32K segment | Previous TTFT | Current TTFT | Recurrent | Dense FFN | Attention |
|---|---:|---:|---:|---:|---:|
| cold 0 -> 32,768 | 60.485 s | 37.968 s | 13.219 s | 18.565 s | 5.856 s |
| retained 32,768 -> 65,536 | 69.141 s | 45.867 s | 13.160 s | 18.600 s | 12.739 s |

A deterministic request placed retrieval/code facts at positions 2,068,
131,013 and 196,510 and queried them at 261,904. K1 and F16 returned the same
47 output tokens byte-for-byte, retrieved the facts, and made the same unrelated
arithmetic error.

| KV | Populated prompt | Wall | Boundary |
|---|---:|---:|---|
| K1 | 262,016 | 546.173 s | one matched lossy-quality case |
| exact F16 | 262,016 | 2,278.082 s | fidelity reference; latency failure |

The 4.171x wall-time ratio and single identical output do not establish general
K1/F16 equivalence. Exact F16 remains the target; `qwen` is an operational K1
opt-in and `qwen-f16` is the reference alias.

### K1 kernel optimization gate

The populated-context attention baseline was 29.0161 ms across 16 layers. A
64-bit packed-load candidate preserved all numerical gates and reduced the
microbenchmark to 28.758 ms, only 0.89%; it missed the <=26.190 ms prerequisite
for 15 generated tokens/s. A 512-thread corrective launch regressed to 30.9719
ms. A later token-striped decode layout reduced excessive shared wavefronts
from 36.44M to 11.27M but reached only 28.5245 ms because global accesses
returned to 75% excessive sectors. All three changes were removed, so none of
these numbers is service throughput or a promoted optimization.

### Exact sampling optimization

The artifact's recommended sampling profile uses `top_k=20` over a
248,320-token vocabulary. Nsight Systems showed that the original single-block
selection rescanned the full vocabulary once per rank and cost 406.572 ms over
41 decode calls, or 9.916 ms/call. This exceeded the 2.826 ms/token saving
required to move the populated K1 result from 14.39 to 15 tok/s.

The replacement partitions the vocabulary into deterministic 4,096-item
ranges, reduces one candidate per range, and performs an exact final reduction
for every rank. An independent CPU sort, including tied logits and NaNs,
matched all 20 token ids. The final clean native smoke measured 0.423 ms/call; the
whole-worker trace measured 10.583 ms over 41 calls, or 0.258 ms/call. Values,
tie order and NaN exclusion are unchanged.

The real populated-context gate above reached 17.07 tok/s post-first, an 18.6%
improvement over 14.39 tok/s. Short chat reached 44.86 tok/s post-first versus
30.92 tok/s previously. This is an exact sampling-path improvement, not an
attention, KV-quality or prefill result: K1 remains lossy and cold 262K TTFT
remained 536.469 s.

### K1 one-draft MTP baseline, 2026-09-19

Build `871ff4d` was run through the real streaming completion service with an
exact 262,016-token repository prompt, 128 generated tokens, `temperature=0`
and the K1 alias. The prompt token-id SHA-256 was
`5a6457d4279ecb5958d7765f6f7cc6feed532dc5dcbf089e13b6dd322599dfff`.

| Gate | Result |
|---|---:|
| cold TTFT | 563.615 s |
| total wall | 568.499 s |
| post-first decode | 26.00 useful tok/s |
| exact target calls / positions | 67 / 127 |
| accepted one-token drafts | 60 / 67 = 89.55% |
| useful positions per target call | 1.8955 |
| complete measured cycle | 72.91 ms/call |
| cleanup | 0 logical pages; 1,024 physical pool pages retained |

The installed one-draft path therefore provides a real K1 prerequisite that
the older exact-F16/offload MTP rejection did not: if the measured conditional
acceptance remains `p=60/67`, a linear MTP-3 rollout has
`A3=1+p+p^2+p^3=3.4157` useful positions/call and MTP-4 has
`A4=1+p+p^2+p^3+p^4=4.0588`. The corresponding complete-cycle limits are:

| Rollout | 40 tok/s | 50 tok/s |
|---|---:|---:|
| MTP-3 | <=85.39 ms | <=68.31 ms |
| MTP-4 | <=101.47 ms | <=81.18 ms |

Against the 72.91 ms one-draft cycle, MTP-4 has 28.56 ms of measured
incremental budget for 40 tok/s but only 8.27 ms for 50 tok/s. This admits an
MTP-4 implementation experiment for the 40 tok/s gate. It does not establish
the sampled `p/q` acceptance rate, multi-row target cost, or 50 tok/s.

The deployed page telemetry reported 4,980,736 target bytes and 278,528 MTP
bytes per 256-token page. A symmetric Q8 MTP record with 256 signed bytes plus
one FP16 scale per token/head/K-or-V requires 528,384 MTP bytes/page, or
541,065,216 bytes at 1,024 pages. This is 255,852,544 bytes (244 MiB) more than
the current MTP-K1 cache. After the completed gate `nvidia-smi` reported
1,286 MiB free, so a direct replacement would leave about 1,042 MiB: only
18 MiB above the required 1 GiB reserve before wider logits or other new
state. The 6,144-row `sequence_hidden` prefill buffer is 125,829,120 bytes
(120 MiB); releasing it after prefill and recreating it only when a later
prefill has headroom is therefore a capacity prerequisite for Q8 MTP.

### Sampled MTP-4 implementation result, 2026-09-19

Exact-decode ABI 2 implemented a recurrent four-draft rollout, a 65,536-token
draft head, unchanged 248,320-token target head, Q8 proposer K/V with
checkpoint/rollback, and five-position target verification. The compact CUDA
attention path processed all speculative queries per K/V tile; direct telemetry
reported ABI 2, draft depth 4, Q8 page bytes 528,384 and multi-query fused
launches. Independent gates passed for Q8 layout/attention (maximum absolute
error `1.49e-08`), the real five-query Qwen geometry (`1.86265e-09`), and
400,000-trial exact `p/q` rejection against an independent scalar oracle.

The real service used terminal windows from the same 262,016-token coding
prompt so the task remained present at the end of both shorter contexts. Both
runs generated exactly 128 tokens with the artifact's thinking profile:
`temperature=1.0`, `top_p=0.95`, `top_k=20`, seed 1. K1 remains explicitly
lossy relative to F16.

| Populated prompt | TTFT | Post-first | Target calls / positions | Accepted drafts | Positions/call |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 39.295 s | 15.67 tok/s | 52 / 128 | 76 | 2.4615 |
| 131,072 | 203.145 s | 8.59 tok/s | 48 / 127 | 79 | 2.6458 |

The first attempted 32K prefix window was excluded: removing the terminal task
made it stop after eight output tokens. It was a prompt-construction failure,
not a throughput sample.

For the two valid runs, the measured post-first seconds per emitted token were
`8.1032434/127 = 0.0638051` at 32K and
`14.7817454/127 = 0.1163917` at 128K. A simple linear extrapolation of the
complete observed decode time per useful token is:

```text
d(C) = 0.0638051 + (C - 32768) *
       (0.1163917 - 0.0638051) / (131072 - 32768)
d(262144) = 0.1865072 seconds/useful token
rate(262144) = 5.36 useful tok/s
```

This is an extrapolation, not a 262K measurement. It is sufficient for the
fail-fast decision because the implementation already misses 40 tok/s by
2.55x at 32K; the two-point 262K estimate misses it by 7.46x. MTP-4 sampled
acceptance also reached only 2.46-2.65 positions per target call, below the
admitted `A>=3.0` boundary. The candidate artifact was therefore not retained
as the default; promotion was transactionally rolled back to exact-decode ABI
1, whose real `hi` smoke passed after rollback.

### MTP-4 rejection-path elimination and clean 32K/128K gates, 2026-09-20

The failed sampled result above was reopened only after profiling found a
complete redundant target execution on every rejected speculative branch. The
five-row target pass already computed the recurrent state after each causal
row, but rejection discarded those states and reran the target prefix. The
replacement emits FP32 convolution and matrix-state checkpoints for all five
rows during the original target pass and restores the selected row after exact
`p/q` acceptance. It does not rerun target weights or target K/V. A direct
recurrent oracle passed with maximum absolute error `1.49e-07`.

The generic program executor also stopped requesting a useless next draft when
the accepted target position reaches the reserved context boundary. Benchmark
service starts explicitly disabled session parking and disk snapshots; no
post-output cache write is included in the decode numbers below.

Both service gates used suffix windows from the same 262,016-token coding
prompt, generated 128 tokens, and kept the real thinking sampler
`temperature=1.0`, `top_p=0.95`, `top_k=20`, seed 1. No 262K gate was run.

| Populated prompt | TTFT | Post-first | Target calls / positions | Accepted drafts | Positions/call | Cycle |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 39.617 s | 26.19 tok/s | 54 / 127 | 73 | 2.3519 | 89.786 ms |
| 131,072 | 203.341 s | 23.89 tok/s | 45 / 127 | 82 | 2.8222 | 118.119 ms |

The clean artifacts are
`qwen-k1-mtp4-recurrent-checkpoint-32768-suffix.json` and
`qwen-k1-mtp4-no-retention-131072-suffix.json`. At 128K the worker ended with
zero allocated and zero reserved pages, proving that the reported wall time
does not contain a session snapshot or parked GPU state.

Observed useful-token time extrapolates as requested from 32K and 128K only:

```text
d32  = 4.8484416 / 127 = 0.0381767 s/useful token
d128 = 5.3153668 / 127 = 0.0418533 s/useful token
d262 = d32 + (7/3) * (d128 - d32) = 0.0467554 s/useful token
rate262 = 21.39 useful tok/s
```

This is an extrapolation, not a 262K measurement. At the measured 128K
acceptance, 40 tok/s requires a complete cycle no slower than
`2.8222 / 40 = 70.56 ms`; the current cycle is 118.12 ms. Extrapolating the
cycle itself gives 155.90 ms at 262K. Even the impossible perfect MTP-4 result
of five useful positions every cycle would then cap at 32.07 tok/s, so the
remaining gap is cycle cost, not another acceptance-only fix.

A one-token context phase run isolated the short-cycle GPU cost across 67
target calls:

| GPU phase | Total | Per target call |
|---|---:|---:|
| dense FFN | 1.695497 s | 25.306 ms |
| recurrent blocks | 1.030414 s | 15.379 ms |
| full attention | 0.290482 s | 4.336 ms |
| MTP proposer | 0.333457 s | 4.977 ms |
| vocabulary head | 0.113024 s | 1.687 ms |

The target dense matrices therefore contribute about 40.69 ms of the fixed
cycle before long-context attention.

An SM86 small-batch BF16 Tensor Core candidate was tested against the selected
DP4A weight-reuse kernel for the real batch of five. It was numerically within
`5.34058e-05`, but the first version reached only 148.60 GB/s on the
17,408x5,120 projection and 67.14 GB/s on the 5,120x17,408 projection, versus
453.70 and 365.78 GB/s for DP4A. Removing every block-wide K-loop barrier
regressed it further to 121.74 and 51.17 GB/s. The candidate was removed; no
context gate was run for a kernel that failed its bandwidth prerequisite.

## Sparse MoE evidence

### Ornith startup fitting

Ornith's routed pool is 15.9765625 GiB. The artifact-neutral `fit` policy
reserved future K1 state/workspaces and admitted 15.9375 GiB (99.7555%).

| Run | Prompt/output | First-visible | Wall | After-first | Evictions |
|---|---:|---:|---:|---:|---:|
| K1+fit cold | 13/12 | 4.590 s | 8.618 s | 2.73 tok/s | 0 |
| following turn | 39/13 | 1.817 s | 3.435 s | 7.42 tok/s | 0 |
| fixed-F16 control | 13/12 | 12.211 s | 21.708 s | 1.16 tok/s | not a fit gate |

This proves a short-chat paging improvement, not maximum-context quality.

### DeepSeek paging

One scalar output can select approximately 3.21 GiB of immutable experts:

```text
43 layers * 6 experts * 13,369,344 bytes = 3.212 GiB/token
```

| Workload | Rate/result | Tier evidence |
|---|---:|---|
| settled identical primary-path prompt, 24 output | 6.54-6.69 tok/s | zero SSD; about 47.8 GB H2D/request |
| novel suite | mean 0.57 tok/s | 8.2-16.8 GiB SSD/request |
| retained six-turn suite | mean 0.73 tok/s | 10.3-14.0 GiB SSD on later turns |
| exact CPU/GPU split | 1.29 tok/s | 377 CPU + 1,653 GPU decisions; no speed gain |
| P100 first request | 0.57-0.63 tok/s | exact active-expert execution |
| P100 settled request | 3.10-3.18 tok/s | zero expert-weight transport from storage |

The P100 allocator's final arena kept about 1.66 GiB free per card during a
28,380-selection `xhigh` soak, but that run was cancelled after 105 hidden
tokens. It proves bounded capacity, not useful throughput. The settled P100
path is slower than the best recorded settled primary path.

### Qwen Flash

Short profiling attributed 84.7% of the program to routed MoE and about 1.2%
to QSA/full attention. A historical 18 GiB cache raised a comparable first-warm
result from 5.85 to 8.30 tok/s, but 18 GiB is not the validated common profile.

The exact P100 expert path completed `31 * 48 * 10 = 14,880` selections with
zero failures. It needed 157.743 s for 31 generated tokens (0.20 tok/s
end-to-end), versus 108.456 s for 40 tokens (0.37 tok/s) on the earlier primary
path. Output lengths differ, but the P100 path plainly failed its acceleration
gate.

## P100 admission gates

The host has one RTX 3090 and two P100 PCIe 16 GB cards without CUDA peer
access. Every cross-device boundary therefore passes through pinned host
memory.

| Proposed placement | Best measured auxiliary wall | Required | Decision |
|---|---:|---:|---|
| exact 262K attention split across P100s | 77.101 ms compute-only | <=32 ms | reject |
| resident compact FP4 dense shard | 35.939 ms/card before integration | <=20 ms/card | reject |
| resident lossless FP16 MLP neuron shards | 50.871 ms parallel, before remaining RTX work | >=500 GB/s/card; measured 307.7/311.0 | reject |
| Qwen Flash selected experts | 0.20 tok/s end-to-end | beat 0.37 tok/s primary | reject for speed |
| DeepSeek selected experts | 3.18 tok/s settled | >=10 tok/s | reject for speed |

The implementation may provide capacity for compatible sparse experts, but the
P100s are not a demonstrated throughput tier.

## Sessions and cleanup

| Gate | Result |
|---|---|
| two exact-F16 sessions, one hot slot | four parks/restores, suffix-only prefill, 351,952,896 parked bytes |
| alternating Pi sessions | about 6,741 tokens/session, 54 pages, 1,223,811,072 parked bytes |
| cancellation/resume | active count returned to zero; committed state survived |
| progressive F16 growth | 200 -> 300 -> 300 used 300 cumulative prefill tokens; final turn was zero-delta |

These prove retained continuity, not simultaneous decode. Recorded release
gates stopped the service and returned device memory to desktop baselines.

### Real Pi retained-prefix observation, 2026-09-17

A large Pi transcript reached the service immediately after a fresh Qwen
process. Pi supplied the transcript again, but the provider had no retained
state from the previous process, so the first request was correctly reported
as a cold runtime request. The immediately following turn found the exact
retained prefix and prefilled only its suffix:

| Runtime request | Resumed | Provider prefill | Output | TTFT/wall | Park/restore evidence |
|---|---:|---:|---:|---:|---:|
| fresh-process transcript | no | 120,192 | 8,586 | 171.718/545.640 s | retained 120,187 tokens; parked 2,630,774,784 bytes |
| next Pi turn | yes | 9,035 | 220 | 17.578/27.765 s | restored 2,630,774,784 bytes; retained 129,217 tokens; parked 2,814,849,024 bytes |

At the time of this earlier observation, the existing cache worked only across
turns in one live service process, including hybrid recurrent state. Pi's own
transcript resume was not equivalent to a provider state hit; the durable
restart path is measured separately below.

### Durable Pi continuation, 2026-09-17

The provider-backed NVMe continuation store was then tested with a real Pi
session, a complete Qwen service restart, and the next Pi turn. This is a
small-session functionality gate, not a 262K throughput claim.

| Stage | Retained prefix | Durable bytes | Restore / provider prefill | TTFT / wall |
|---|---:|---:|---:|---:|
| initial Pi turn | 443 tokens | 169,443,798 | no restore / 450 tokens | 1.031 / 1.312 s |
| after complete restart | 443 tokens | 169,443,798 | 0.203 s / 34 tokens | 0.625 / 0.938 s |

At restart, `retained=0` and `disk_retained=1`: the snapshot consumed no
session RAM until the exact prefix matched. The service logged one provider
restore of 169,439,232 bytes and then atomically committed the new generation.
Metadata corruption and byte-bounded LRU eviction are covered by the server
contract suite. A fixed-seed, `temperature=0` parity gate produced
`PARITY-BETA` both live and after a complete restart/restoration of the same
snapshot. A large real coding-session gate remains unrun.

## Current conclusion

- all six artifacts pass direct chat and minimal Pi wiring;
- short Qwen/Muse decode can be responsive;
- Qwen exact-F16 262K is correct but not interactive; K1 reaches the capacity
  and approximate decode target only as an unqualified lossy experiment;
- Flash and Mistral are callable but fail practical coding latency/quality;
- exact sparse paging makes DeepSeek executable beyond RAM+VRAM, but novel and
  settled performance remain below target;
- the installed P100s do not provide a demonstrated throughput improvement.
