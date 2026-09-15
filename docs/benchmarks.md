# Benchmarks and evidence

Status: canonical measurement ledger, 2026-09-15.

## Reporting rules

- TTFT is server time to first generated token unless marked first-visible.
- After-first rate excludes TTFT and the first token.
- Prompt, reasoning, visible and total generated tokens are distinct.
- Configured, populated and harness-provided context are distinct.
- Cold, warm, prefix-reused, novel-route and settled-route runs are not
  interchangeable.
- A numerical gate or microbenchmark is not service throughput.

## Validation baseline

The latest changed-code gate used the Windows Release/CUDA build (MSVC 19.44,
CUDA 12.1), built `--clean-first --parallel 22`. Windows CTest passed 6/6;
the canonical compiler/server suite passed 117/117; Python byte compilation,
shell syntax and `git diff --check` passed. The Windows binary is authoritative.

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

## Current conclusion

- all six artifacts pass direct chat and minimal Pi wiring;
- short Qwen/Muse decode can be responsive;
- Qwen exact-F16 262K is correct but not interactive; K1 reaches the capacity
  and approximate decode target only as an unqualified lossy experiment;
- Flash and Mistral are callable but fail practical coding latency/quality;
- exact sparse paging makes DeepSeek executable beyond RAM+VRAM, but novel and
  settled performance remain below target;
- the installed P100s do not provide a demonstrated throughput improvement.
