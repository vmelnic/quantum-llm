# Benchmarks and evidence

Status: canonical measurement ledger, 2026-08-26. Numbers removed from this
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

Latest changed-code gate on 2026-08-26:

- the official Windows Release/CUDA build compiled the common VM and both
  providers with MSVC 19.44 and CUDA 12.1;
- Windows CTest passed 5/5 native/CUDA tests;
- the Windows compiler/server contract suite passed 101/101 tests, including
  generic compact/standard FP4 host execution, progressive session accounting
  and a prefill-sized (>32 selection) universal FP4 host work group;
- the broader control-host suite passed 119 tests with 9 optional dependency
  skips;
- `git diff --check`, Python byte compilation and shell syntax checks passed.

The control host does not have CMake installed, so no portable C++ build was
claimed there. The Windows build is the authoritative CUDA compilation gate.

## Current service evidence

| Gate | Configuration | Result | Verdict |
|---|---|---|---|
| Qwen chat | common VM, exact F16 KV, `model.sh chat`, `hi` | 13 prompt, 10 generated, 0.957 s first visible, 1.311 s wall, 7.63 tok/s end-to-end, 25.42 tok/s after first visible | short-context regression pass |
| Muse chat | common VM, exact mixed global/window F16 KV, `model.sh chat`, `hi` | 57 prompt, 59 generated (40 reasoning), 2.037 s first visible, 2.336 s wall, 25.26 tok/s end-to-end, `finish=stop` | text common-path pass; not a populated 131K gate |
| Ornith chat | common VM, FP4 standard expert pages, exact F16 KV policy, `model.sh chat`, `hi` | 13 prompt, 10 generated, 1.647 s first visible, 2.255 s wall, 4.44 tok/s end-to-end, 14.82 tok/s after first visible | common-path regression pass |
| Qwen earlier long answer | common VM, exact F16 KV, four-stanza prompt | 68 prompt, 476 generated, 2.812 s TTFT, 19.047 s wall, 29.26 tok/s after first token | short-context quality/throughput pass |
| DeepSeek chat, exact hybrid q-star active | common VM, artifact BF16 KV, `model.sh chat`, `hi` | 5 prompt, 10 generated, 4.355 s first visible, 11.227 s wall, 0.89 tok/s end-to-end, 1.31 tok/s after first visible | common-path regression pass; q-star speedup still not proved |
| Qwen tiled prefill | exact F16, 7,052 prompt + 1 output | 23.913 s, two GPU tiles, zero host activation spill | functional pass |
| Qwen near-maximum prefill | exact F16, 262,001 prompt + 1 output | 2,183.815 s, 43 GPU tiles, zero host activation spill | capacity pass, latency fail |
| Qwen full maximum, historical | experimental artifact FP4 KV, 262,016 prompt + 128 output | 5,423.422 s TTFT, 5,462.672 s wall, about 3.24 tok/s after first token | capacity only; wrong KV format for exact-F16 goal |
| Qwen full maximum, FP8 | FP8 E4M3 target KV + FP4 MTP KV, 262,016 + 128 | 1,168.594 s prefill, 3.064 tok/s decode | rejected for speed and quality |
| Qwen startup | native Windows hash path | 35.95 s from `model.sh start qwen` to ready; pack verification 11.56 s | pass |
| introspection under load | `/metrics` during 7,052-token prefill | 0.0206 s | pass |

The near-maximum exact-F16 run generated only one token. It proves the current
prefill/capacity path and does not provide a saturated exact-F16 decode rate.
The 3.24 tok/s result used compact artifact KV and cannot be relabelled F16.

## Muse artifact and bounded text evidence

```text
source:       meta-models/Muse-Glimmer-30B
revision:     f84ecc3a0ea984a4c04542a84269e3d065350a6e
artifact:     ${MODEL_ROOT}/muse-glimmer-30b-fp4
dense.qpack:  15,832,002,560 bytes
records:      723 FP4 + 713 F32; zero INT8
```

The source gate found zero payload, scale or F32 mismatches. FP4-only cosine
was 0.9930505 and relative L2 was 0.1177243; aggregate cosine was 0.9999981.
On one official 68-token prompt, the first 32 generated token IDs from BF16
decoded to the same raw prefix produced by QPack after accounting for the
API's special-token stripping. The service then completed the bounded
instruction with visible `ALPHA-17` and an official EOS rather than exhausting
the output budget.

The tokenizer-declared response grammar exposed two generic service bugs that
were fixed without family branches: protocol parsing now preserves structural
special tokens internally, and generation respects every EOS ID from
`generation_config.json`. Streaming with artifact codecs withholds split EOS
terminals instead of leaking them to clients. This also removed the visible
DeepSeek EOS regression in its repeated real chat gate.

This evidence qualifies short text inference only. Vision records are
auxiliary, no Muse image path is advertised, and 131,072 positions were not
populated or timed.

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

The 2026-08-24 real Pi tool-mode streaming gate used the default coding
harness and `xhigh` reasoning.  The first implementation proved incremental
delivery but failed after generation because the Transformers stream parser
exposed the structural `\n\n` after `</think>`, while its authoritative final
parse removed it.  That run generated 209 tokens but ended with a protocol
error and zero client usage; it is a rejected result.

The bounded final-parser reconciliation made the repeated Qwen gate pass with
7,923 populated prompt tokens and 217 generated tokens, but it remained a
fragile two-parser design and is no longer the active implementation. On
2026-08-25 it was replaced by one artifact-declared parser instance per
streamed response: its deltas are authoritative, and its own finalized message
supplies structured tool calls. There is no independent final reparse or
content reconciliation.

The replacement passed the normal Pi `read` loop on Qwen. The first turn used
7,942 input and 86 output tokens, emitted `read({"path":
"ops/model-aliases.tsv","offset":1,"limit":3})`, and Pi executed it. The
follow-up used 8,112 input and 53 output tokens and returned
`deepseek-v4-flash`. The contemporaneous direct `model.sh chat` regression
used 13 prompt and 10 output tokens, reached visible text in 0.762 s, completed
in 1.139 s, and decoded at 23.85 tok/s after first visible. These are
short/medium-context integration results, not a 262K gate.

The earlier Claude Code gate failed: its cold `hi` carried 29,487 prompt tokens,
needed 82.875 s TTFT and 88.281 s wall. A factual request carried 29,660 prompt
tokens and returned a fabricated Maia Sandu biography. The Anthropic wire
adapter works, but that result rejects unrestricted Claude Code production use.

## Ornith artifact and common-path evidence

Published artifact:

```text
source:                 ornith-ai/Ornith-1.5-35B-A3B
revision:               10fbf86fed7ecee4a061f8b499a618f46001cac1
artifact:               ${MODEL_ROOT}/ornith-1.5-35b-a3b-fp4
adapter/runtime family: hybrid_delta / hybrid_delta_moe
dense.qpack:            2,075,074,560 bytes
expert files:           4 files, 17,154,703,360 bytes
records:                1,230 dense FP4 + 501 F32 + 10,240 expert FP4
matrix/expert format:   FP4 E2M1 + UE8M0 block-32 scales
```

The strict Xet inventory covered all 1,811 official BF16 source tensors and
71,903,645,408 indexed tensor bytes. Conversion classified every source
tensor, produced the artifact in a candidate directory, validated it, and
promoted it atomically below `MODEL_ROOT`. The final validator reported five
packs, 1,731 dense records, 10,240 expert records and 19,229,777,920 total pack
bytes.

The source qualification sampled 8,282,432 values. It found zero F32, FP4
payload or FP4-scale reconstruction mismatches. Aggregate relative L2 was
0.006980 with cosine 0.999976; FP4-only relative L2 was 0.118762 with cosine
0.992932. The worst sampled tensor remained within the declared gates at
0.187530 relative L2 and 0.984286 cosine.

The real short chat gate loaded the published artifact through the common
service and generic VM; there is no Ornith task, runner or model-family branch.
Request telemetry recorded 4,046 routed RAM hits, 2,005,401,600 uploaded bytes,
2,846 CPU decisions and 1,206 GPU-upload decisions. It recorded no SSD miss or
storage read for this request because the 15.98 GiB routed pool fits the host
bank. This is a short-context execution and placement pass, not a populated
262K, visual-quality or broad behavioral qualification.

The first 2026-08-25 Pi wiring gate exposed and verified removal of a
decode-only 32-selection limit in the universal FP4 host executor. That
reduced `--no-context-files --no-tools` check proved only prefill wiring.

The subsequent normal Pi project gate used 7,874 prompt tokens, generated 157
tokens, reached first output in 24.578 s and completed in 34.937 s. It answered
the project question without the former post-generation parser failure. The
explicit tool-loop gate then used 7,906 prompt/101 output tokens for the
initial turn (25.406 s TTFT, 32.156 s wall), emitted a structured `read` call,
and Pi executed it. The retained follow-up used 230 suffix prompt tokens and
191 output tokens (1.282 s TTFT, 14.032 s wall) and returned the requested
`deepseek-v4-flash` value. No `response_stream_*_failed` event was present.
This qualifies normal short/medium-context Pi text and tool wiring, not
maximum-context throughput or broad coding quality.

The same server revision passed DeepSeek `model.sh chat --thinking off` with 5
prompt and 10 output tokens, 4.479 s first visible and 11.748 s wall (1.24
tok/s after first visible). Its normal Pi request was stopped while still in
the known slow harness prefill, so no DeepSeek Pi pass is claimed. The server
initially acknowledged `model prefill was cancelled`, then its worker later
became unhealthy while the Windows wrapper task remained running. A clean
service restart restored `ready`, and the repeated `chat hi` passed with 4.475
s first visible, 11.037 s wall and 1.37 tok/s after first visible. Automatic
DeepSeek worker recovery after cancellation remains open.

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
| pre-change GPU-only common-runner `hi` | 0.94 tok/s end-to-end, 1.42 tok/s after first visible | 850 SSD misses, 11.364 GB read, 19.265 GB uploaded |
| exact CPU/GPU q-star common-runner `hi` | 0.96 tok/s end-to-end, 1.29 tok/s after first visible | 377 CPU decisions, 1,653 GPU decisions, 5.054 GB CPU source-weight footprint assigned |

The active q-star gate processed CPU work for 221.158 ms and preserved exact
top-6/MTP aggregation. It reduced first-visible latency from 4.333 s to
3.409 s, but the tiny ten-token decode fell from 1.42 to 1.29 tok/s after the
first visible token. This proves that the mechanism is wired and exact enough
to complete the service path; it does not prove a throughput improvement on
the current host. No Pi measurement was run in this change.

Task indexing removed control scans but not the expert supply bytes. MTP was
throughput-neutral in the storage-bound regime because adjacent routes did not
share enough experts; later compact FP4 residency made it useful on settled
turns. Any DeepSeek claim must therefore state cold/novel/settled placement.

## What the evidence proves

- The common service and the Qwen, Muse-text, Ornith and DeepSeek paths are
  functional for their recorded short-chat gates.
- Qwen short-context FP4/F16 post-first-visible decode can exceed 15 tok/s;
  the current short gate does not meet 15 tok/s end to end.
- Exact session retention and progressive F16 allocation work for measured
  short/medium histories.
- Qwen exact-F16 maximum prefill is still about 36 minutes and the requested
  maximum-context interactive target is not met.
- DeepSeek paging makes the 147 GB expert pack executable without full
  residency, but novel routes remain movement-bound.
- NVMe improves cold loading and disk misses; it cannot remove hot dense/KV
  bytes from every decode call.
