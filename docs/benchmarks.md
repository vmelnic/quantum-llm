# Benchmarks and evidence

Status: canonical measurement ledger, 2026-09-02.

## Metric rules

- TTFT is server time to the first generated token unless explicitly labelled
  first visible content.
- Total rate is generated tokens divided by request wall time. Post-first-token
  rate excludes TTFT and the first token.
- Prompt, generated, reasoning and visible/useful output are distinct.
- Configured context is not populated context. A maximum-context claim reports
  actual positions and KV format.
- Cold, warm, reused-prefix, novel-route and settled-route results are not
  interchangeable.
- Microbenchmarks and bandwidth equations are ceilings, never service results.

## Validation baseline

Latest recorded changed-code gate on 2026-09-01:

- Windows Release/CUDA build completed with MSVC 19.44 and CUDA 12.1;
- Windows CTest passed 6/6;
- canonical compiler/server suite passed 113/113;
- Python byte compilation, shell syntax and `git diff --check` passed.

The Windows build is authoritative. The build workflow uses `--clean-first`;
an earlier mixed-object CUDA ABI failure proved that `cmake --fresh` does not
guarantee recompilation after an internal header change.

## Current Pi wiring gate

Command for every row:

```bash
./ops/pi.sh <alias> --no-context-files --no-tools --no-session -p hi
```

Configuration: one RTX 3090, current artifacts, common 12 GiB routed-VRAM
ceiling, default `xhigh`. Results are server telemetry from 2026-09-02.

| Alias | Prefill | Generated | TTFT | Wall | Result |
|---|---:|---:|---:|---:|---|
| `qwen` | 5 | 33 | 0.625 s | 1.938 s | valid response |
| `qwen-flash` | 477 | 67 | 41.829 s | 53.688 s | valid response, slow |
| `mistral` | 439 | 67 | 54.797 s | 77.312 s | valid response, slow |
| `muse` | 422 | 65 | 2.297 s | 4.297 s | valid response |
| `ornith` | 439 | 34 | 3.703 s | 5.390 s | valid response |
| `deepseek` | 504 | 256 | 81.985 s | 310.672 s | valid response, impractical latency |

This is a lifecycle/template/API/generation gate. It is not a coding, tool,
maximum-context or concurrency benchmark. Pi may still supply its own system
instructions when project context and tools are disabled.

DeepSeek's row also recorded 17,359 SSD misses, 232,078,442,496 cache-read
bytes, 128,653,197,312 reread bytes, 265,354,739,712 uploaded bytes,
23,728 RAM hits, 24,828 VRAM hits and 271.635 s storage wait. That result is
movement-bound despite completing correctly.

## Direct service evidence

These are the latest retained short-chat regressions for each artifact; they
use `model.sh chat` and are not all from one matched performance run.

| Model | Prompt/output | TTFT or first visible | Wall | Evidence boundary |
|---|---:|---:|---:|---|
| Qwen3.8-27B | 13 / 10 | 0.625 s | 1.015 s | current common-path short chat |
| Qwen3.8-Flash-Next | 13 / 10 | 3.004 s visible | 5.686 s | coherent short chat only |
| Mistral Small 4 | 541 / 13 | 58.480 s visible | 62.298 s | native NVFP4 text path; cold latency failure |
| Muse-Glimmer | 57 / 100 | 3.540 s visible | 3.826 s | text path; 81 reasoning tokens |
| Ornith | 13 / 10 | 0.814 s visible | 1.401 s | common routed path |
| DeepSeek | 5 / 10 | 3.203 s | 9.672 s | exact paged path, about 1.38 tok/s after TTFT |

Qwen also generated a 476-token four-stanza answer at 29.26 tok/s after the
first token. That is useful short-context decode evidence, not end-to-end or
262K throughput.

## Artifact and numerical evidence

| Artifact | Evidence |
|---|---|
| Qwen3.8-27B | 14,775,390,208-byte QPack, 1,199 records; 666 FP4 and 533 F32 records matched the pinned source samples; FP4-only relative L2 0.121706, cosine 0.992577; bounded official/service corpus passed 4/4 |
| Qwen3.8-Flash-Next | 1,562 dense and 24,576 expert records in 95,915,634,688 bytes; exact QSA resident/staged/selected CUDA parity reported maximum absolute difference 0 |
| Mistral Small 4 | pinned 13-shard official source, 70,801,904,048 tensor bytes; source-native NVFP4 values/sidecars and BF16 organs validated; corrected activation-scale oracle agrees with coherent service text |
| Muse-Glimmer | 15,832,002,560-byte QPack, 723 FP4 + 713 F32 records; aggregate cosine 0.9999981, FP4-only cosine 0.9930505; bounded official prefix/service behavior passed |
| Ornith | 19,229,777,920 total pack bytes, 1,731 dense and 10,240 expert records; aggregate relative L2 0.006980/cosine 0.999976, FP4-only 0.118762/0.992932 |
| DeepSeek | 11,008 compact expert records and 147,169,738,752 routed bytes; independent source, attention, routing, expert and I/O oracles precede bundle publication |

Source reconstruction is necessary but not sufficient: a real provider and
service gate remain required.

## Qwen long-context evidence

Current target-call weight traffic, excluding sparse embedding lookup, vision
and MTP but including the vocabulary head:

```text
hot target weights             13,625,700,352 bytes = 12.690 GiB
exact F16 KV at 262,144        17,179,869,184 bytes = 16.000 GiB
subtotal                       30,805,569,536 bytes
RTX 3090 physical              25,769,803,776 bytes
```

The subtotal exceeds VRAM before recurrent state, MTP, activation workspace
and display/emergency reserve. At the measured 12.46 GiB/s pinned link, one
complete 16 GiB host-KV scan costs at least 1.284 s per scalar decode call.

| Gate | Result | Verdict |
|---|---:|---|
| exact F16, 7,052 prompt + 1 output | 23.913 s, two GPU tiles, zero host activation spill | functional prefill |
| exact F16, 262,001 prompt + 1 output | 2,183.815 s, 43 GPU tiles, zero host activation spill | capacity pass, latency fail; no saturated decode rate |
| historical compact artifact KV, 262,016 + 128 | 5,423.422 s TTFT, about 3.24 tok/s after first | wrong KV format for exact-F16 target |
| rejected FP8 KV, 262,016 + 128 | 1,168.594 s prefill, 3.064 tok/s decode | fidelity and speed fail |

The progressive exact-F16 device mirror improved the matched bounded Pi Todo
gate from 1,645 s to 503.240 s end-to-end and from 1,493 s to 481.563 s to
green. The run reached 3/3 tests with about 17.8K final prompt tokens and no
host-attention call or mirror spill. This validates short/medium-prefix
placement; it does not change the 262K capacity inequality.

## Harness quality evidence

| Model/mode | Result |
|---|---|
| Qwen effective `medium` | synthetic Todo fixture reached 3/3 in 481.563 s with the device mirror; 503.240 s total |
| Qwen `off` | zero reasoning, implemented code but remained 1/3 after 569 s |
| Qwen Flash corrected `medium` | valid tools but zero edits; later reproduced an action stall beyond 1,000 s |
| Qwen Flash `off` | zero edits and invalid generated paths; parser trace later preserved measured native arguments exactly |
| Mistral | 14 valid/error-free tool calls, zero edits at the 30-minute cutoff |
| Ornith | bounded project text and retained `read` loop passed |
| Muse | no representative coding qualification |
| DeepSeek | minimal Pi `hi` passed; representative project prefill/recovery not qualified |

The fixture is bounded and does not establish broad coding quality. It does
establish that minimal chat or valid tool transport cannot be promoted as an
autonomous coding pass.

## Sessions and multi-agent evidence

| Gate | Result |
|---|---|
| two exact-F16 sessions, one hot slot | four parks, two restores, suffix-only prefill, 351,952,896 parked bytes |
| real alternating Pi sessions | about 6,741 tokens/session, 54 parked pages, 1,223,811,072 parked bytes |
| cancellation/resume | active count returned to zero; committed sessions survived; next turn added only the suffix |
| progressive F16 growth | 200 -> 300 -> 300 prompt used exactly 300 cumulative prefill tokens and two pages; last turn was zero-delta |

These prove retention and progressive allocation, not parallel decode or
several simultaneous populated 262K sessions.

## Sparse-routing evidence

Qwen Flash phase profiling attributed 84.7% of a cold short program to routed
MoE and about 1.2% to QSA/full attention. A historical 18 GiB cache experiment
raised a comparable first-warm result from 5.85 to 8.30 tok/s and reduced
storage reads, but 18 GiB is not the current common validated configuration.
The active common ceiling is 12 GiB because DeepSeek fails preflight at 13 GiB.

DeepSeek cold scalar routing can select:

```text
43 layers x 6 experts x 13,369,344 bytes = about 3.21 GiB/token
```

Historical matched probes:

| Workload | Result | Placement evidence |
|---|---:|---|
| settled identical prompt, 24 output | 6.54-6.69 tok/s | zero SSD, about 47.8 GB H2D/request |
| eight novel prompts, 12 output each | 321.47 s total | 97.156 GiB read, 58.558 GiB reread, 315.102 GiB H2D |
| novel suite before indexing | mean 0.57 tok/s | 8.2-16.8 GiB SSD/request |
| retained six-turn suite | mean 0.73 tok/s | 10.3-14.0 GiB SSD on later turns |
| exact CPU/GPU split short gate | 1.29 tok/s after first visible | 377 CPU and 1,653 GPU decisions; no measured throughput gain |

Demand paging makes the 147 GB pool executable. Novel routes remain movement
bound, and the settled 10-15 tok/s target has not been reached.

## Current conclusion

- all six artifacts execute through the common service and minimal Pi gate;
- Qwen short-context decode can exceed 15 tok/s after TTFT, but populated 262K
  exact-F16 inference is not interactive;
- exact retention/progressive allocation work for measured shorter histories;
- Qwen Flash and Mistral are wired but fail practical Pi latency/quality gates;
- DeepSeek exact paging works beyond RAM+VRAM, but novel routes remain near the
  accepted 1 tok/s class and settled performance is below target.
