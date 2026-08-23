# Benchmarks and evidence

Status: canonical measurement ledger, 2026-08-23. Numbers removed from this
document are historical context, not current claims.

## Metric rules

- `TTFT` is server time from accepted request to first generated token unless
  explicitly labelled first visible content.
- End-to-end generation rate is `generated_tokens / wall_seconds`.
- Post-first-token rate uses the remaining generated tokens and remaining wall
  time. Hidden reasoning makes first visible content unsuitable for this
  calculation.
- Every result identifies the model, KV format, prompt size and cache state.
- A maximum-context test must actually populate the reported positions.
- CUDA microbenchmarks and bandwidth equations are ceilings, not service
  throughput.

## Current repository validation

After the repository/documentation cleanup on 2026-08-23:

- the complete active local Python suite passed 108 tests, with eight skips
  limited to the optional NumPy fast path absent on the control host;
- the official Windows Release/CUDA build compiled the common VM and both
  providers with MSVC 19.44 and CUDA 12.1;
- Windows CTest passed 5/5 native/CUDA tests;
- the Windows compiler/server contract suite passed 90/90 tests, including the
  NumPy reference paths available on that host;
- all local Markdown links and `git diff --check` passed.

The control host does not have CMake installed, so no portable C++ build was
claimed there. The Windows build is the authoritative CUDA compilation gate.

## Current service evidence

| Gate | Configuration | Result | Verdict |
|---|---|---|---|
| Qwen chat | common VM, exact F16 KV, `model.sh chat`, `hi` | 53 prompt, 45 generated, 0.829 s TTFT, 2.672 s wall, 16.51 tok/s end-to-end | short-context functional pass |
| Qwen earlier long answer | common VM, exact F16 KV, four-stanza prompt | 68 prompt, 476 generated, 2.812 s TTFT, 19.047 s wall, 29.26 tok/s after first token | short-context quality/throughput pass |
| DeepSeek chat | common VM, artifact BF16 KV, `model.sh chat`, `hi` | 5 prompt, 10 generated, 3.788 s first visible, 11.276 s wall, 0.89 tok/s end-to-end, 1.20 tok/s after first visible | lifecycle pass, slow |
| Qwen tiled prefill | exact F16, 7,052 prompt + 1 output | 23.913 s, two GPU tiles, zero host activation spill | functional pass |
| Qwen near-maximum prefill | exact F16, 262,001 prompt + 1 output | 2,183.815 s, 43 GPU tiles, zero host activation spill | capacity pass, latency fail |
| Qwen full maximum, historical | experimental artifact FP4 KV, 262,016 prompt + 128 output | 5,423.422 s TTFT, 5,462.672 s wall, about 3.24 tok/s after first token | capacity only; wrong KV format for exact-F16 goal |
| Qwen full maximum, FP8 | FP8 E4M3 target KV + FP4 MTP KV, 262,016 + 128 | 1,168.594 s prefill, 3.064 tok/s decode | rejected for speed and quality |
| Qwen startup | native Windows hash path | 35.95 s from `model.sh start qwen` to ready; pack verification 11.56 s | pass |
| introspection under load | `/metrics` during 7,052-token prefill | 0.0206 s | pass |

The near-maximum exact-F16 run generated only one token. It proves the current
prefill/capacity path and does not provide a saturated exact-F16 decode rate.
The 3.24 tok/s result used compact artifact KV and cannot be relabelled F16.

## Qwen artifact and quality evidence

Published artifact:

```text
source:          Qwen/Qwen3.8-27B
revision:        1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
artifact:        ${MODEL_ROOT}/qwen3.8-27b-fp4
dense.qpack:     14,775,390,208 bytes
records:         1,199
matrix format:   FP4 E2M1 + UE8M0 block-32 scales, ABI 3
```

The source-fidelity gate compared 666 FP4 and 533 F32 records with the pinned
official checkpoint and found no sampled payload, scale or F32 mismatches.
Across 159,152 sampled FP4 values, FP4-only relative L2 was 0.121706 and cosine
similarity was 0.992577.

A bounded official-BF16 behavioral corpus covered instruction following,
reasoning, code semantics and tool output. Official and service paths passed
4/4 cases and their prompt token counts matched (24/67/123/326). This is useful
bounded evidence, not a broad model-quality certification.

Functional integration gates also passed:

- official XML tool call parsed as `get_weather({"city":"Chisinau"})` and a
  resumed tool-result turn produced the final answer;
- a 320x240 red image produced `Red` with one image/320 patches in telemetry;
- Pi attached a blue PNG and produced `Blue` with one image/256 patches.

The earlier Claude Code gate failed: its cold `hi` carried 29,487 prompt tokens,
needed 82.875 s TTFT and 88.281 s wall. A factual request carried 29,660 prompt
tokens and returned a fabricated Maia Sandu biography. The Anthropic wire
adapter works, but that result rejects unrestricted Claude Code production use.

## Session and multi-agent evidence

| Gate | Result |
|---|---|
| two compact-KV sessions, one hot slot | 2 parks + 2 restores; 58 delta-prefill tokens instead of 170 complete-prompt tokens; restored greedy continuation matched fresh execution |
| two exact-F16 sessions, one hot slot | 4 parks, 2 restores, 123 total prefill tokens (113 cold + 10 suffix), 351,952,896 parked bytes |
| real Pi alternating sessions | about 6,741 tokens/session, 54 parked pages, 1,223,811,072 parked bytes, delta-only continuation |
| cancellation/resume | Ctrl-C returned active count to zero; both sessions survived; next turn added only 22 prefill tokens |
| progressive F16 growth | 200 -> 300 -> 300 token prompt used exactly 300 cumulative prefill tokens and two 256-token F16 pages; last turn was zero delta |

These gates prove exact retention and progressive allocation. They do not prove
parallel execution or several simultaneous 262K sessions. With one hot slot,
requests time-share the GPU and populated F16 pages consume host RAM.

## Qwen capacity and traffic equations

The current target-call weight scan, excluding sparse embedding lookup, vision
and MTP but including the vocabulary head, is:

```text
text FFN                         9,091,940,352 bytes
recurrent-attention weights     2,963,472,384
full-attention weights            891,682,816
vocabulary head                   675,434,496
norms/other                         3,170,304
total                           13,625,700,352 bytes = 12.689922 GiB
```

Exact F16 target KV at 262,144 populated tokens is:

```text
16 layers * 2 (K,V) * 4 heads * 256 values * 2 bytes * 262,144
= 17,179,869,184 bytes = 16 GiB
```

Weights plus target KV are therefore 30,805,569,536 bytes before recurrent
state, MTP, activations, workspaces and allocator/display reserve. They cannot
be simultaneously resident on a 24 GiB RTX 3090.

If the complete set could reside behind the RTX 3090's 936 GB/s memory system,
the impossible zero-overhead scalar floor would be about 32.9 ms/call, or 30.4
calls/s. Capacity prevents that placement. With exact KV in host RAM, scanning
16 GiB over the measured 12.46 GiB/s pinned link costs about 1.284 seconds per
scalar call before attention compute. Ordinary paging cannot meet 15 tok/s.

## DeepSeek evidence

The routed compact pack contains 43 artifact-declared shards, 256 experts per
shard, 11,008 records total and 147,169,738,752 bytes. Each exact route selects
six 13,369,344-byte records per routed layer, so an entirely cold scalar token
requires about 3.21 GiB of routed payload before overfetch or churn.

Historical matched probes establish the range, not one universal rate:

| Workload | Result | Traffic state |
|---|---:|---|
| settled identical prompt, 24 output tokens | 6.54-6.69 tok/s | zero SSD, about 47.8 GB H2D/request |
| eight novel prompts, 12 output tokens each | 321.47 s total | 97.156 GiB read, 58.558 GiB reread, 315.102 GiB H2D |
| pre-index novel suite | 0.38-0.69 tok/s, mean 0.57 | 8.2-16.8 GiB SSD/request |
| retained six-turn suite | 0.59-0.86 tok/s, mean 0.73 | 10.3-14.0 GiB SSD on turns 2-6 |
| current common-runner `hi` | 0.89 tok/s end-to-end | tiny lifecycle smoke, not a matched performance rerun |

Task indexing removed control scans but not the expert supply bytes. MTP was
throughput-neutral in the storage-bound regime because adjacent routes did not
share enough experts; later compact FP4 residency made it useful on settled
turns. Any DeepSeek claim must therefore state cold/novel/settled placement.

## What the evidence proves

- The common service, Qwen and DeepSeek paths are functional.
- Qwen short-context FP4/F16 service can exceed 15 tok/s.
- Exact session retention and progressive F16 allocation work for measured
  short/medium histories.
- Qwen exact-F16 maximum prefill is still about 36 minutes and the requested
  maximum-context interactive target is not met.
- DeepSeek paging makes the 147 GB expert pack executable without full
  residency, but novel routes remain movement-bound.
- NVMe improves cold loading and disk misses; it cannot remove hot dense/KV
  bytes from every decode call.
