# Benchmarks and evidence

Status: measurements through 2026-10-21. Short wiring observations and
long-context/product gates are distinguished explicitly below.

## Independent Strata API qualification, 2026-10-21

### Actual Pi request with embedded code, 2026-10-21

The user requested home Pi configuration and a larger-prompt throughput test.
The real installed Pi client loads the new home provider `strata-3090` and
model `swift-flash-next-iq3-s`, with native xhigh and the unchanged serving
policy below. An ephemeral session receives 49442 bytes from ten repository
source/document files and requests a lifecycle review, a revised worker and
four test cases. No existing user sessions/logs are inspected. Tools, context
discovery, extensions, MCP and skills are disabled for this isolated source
generation request. No server restart or inference-code change is made.

| Measurement | Actual Pi request |
|---|---:|
| Populated prompt / cached tokens | 15256 / 0 |
| Completion tokens, reasoning plus visible | 25401 |
| Native prompt processing | 16.990 s, 897.9 tok/s |
| Native generation | 363.916 s, 69.8 tok/s |
| Client total | 381.235 s |
| Output tokens / total client wall | 66.63 tok/s |
| First reasoning delta | 17.302 s |
| First visible delta | 307.249 s |
| Draft proposals / accepted | 21806 / 15648 |
| GPU residency at completion | 20047 MiB |
| System RAM used at completion | 51.8 GiB of 63.9 GiB |

Pi exits zero with one complete assistant response and finish reason stop.
The server request counter increases by exactly one, and the native timing
output count matches Pi usage, tying the measured generation rate to this
Pi request. The configured 262144 context positions were not populated.
The output count includes reasoning: Pi reports reasoning usage zero despite
streamed thinking, so that field is not a valid separate token breakdown.
The response has 268 lines, exceeding the requested 120-180; generated code
and tests were not executed or qualified. This measures coding-generation
throughput, not coding correctness or a tool-loop benchmark. In particular,
the 69.8 tok/s generation rate does not imply a fast visible response at xhigh.
Evidence: `out/strata-pi-speed-20261021/result.json`, source prompt, timestamped
Pi events, response and before/after service timing snapshots. The service is
left running and Pi defaults are preserved; the model is enabled in home
settings for explicit selection.

### Initial API and lifecycle gates

Signed upstream engine 0.1.46, SHA256
`c2335807a15bc5f4f6da8eb3f807b63c73e05fe6b49202436d7da74a8e482c22`,
serves Swift Flash-Next GSQ-RCO IQ3_S on one RTX 3090/Ryzen5 5600/64 GiB
RAM host. Policy: configured context 262144, one slot, FP16 KV with 32768
resident cells, MTP window 4, complete top-10 routing, routed cache <=12 GiB.
The managed runtime is owned by its separate Windows scheduled task and
remains reachable after the initiating SSH connection closes.

Own API requests from the Mac over WireGuard use xhigh:

| Request | Greeting, streamed | Python repair, tool call |
|---|---:|---:|
| Populated prompt tokens | 53 | 362 |
| Cached prompt tokens | 48 | 0 |
| Completion tokens, including reasoning | 36 | 77 |
| First reasoning/content delta, s | 0.391 | not instrumented |
| First visible content, s | 1.037 | structured tool response |
| Total request wall time, s | 1.114 | 4.002 |
| Result | stop, coherent greeting | one valid write_file call |

The repair changes `return a - b` to `return a + b` in add.py; independent
AST checks validate the proposed function and tool arguments without executing
generated code. These are short API/streaming/tool transport gates, not a real
coding benchmark or populated262K qualification. Cold/reused prefill, isolated
decode throughput, reasoning/visible token counts, speculative acceptance and
tool-loop performance were not measured independently. No tok/s claim is
derived from these end-to-end timings.

Idle loaded readback: 19697 MiB GPU, one native worker, 8082 listening,
11380376 KiB available physical RAM. Startup reports 37.93 GiB pinned expert
RAM and 11.98 GiB expert GPU cache. Successful owned stop leaves no native
worker, no 8082 listener, task Ready and 263 MiB GPU. Evidence:
`out/strata-20261021/`; API gate logs, responses, lifecycle and cleanup readback.

## Resident MLA context split deployment, 2026-10-07/08

The user requested further improvement after reporting 0.1 tok/s. The new
common runner parallelizes 1024-position chunks of complete causal MLA for
at most eight rows, then merges stable softmax latent sums on the GPU. It
preserves the existing Mistral NVFP4 artifact, BF16 latent KV, 262,144
configured context, xhigh, zero MTP and 12/48 GiB routed GPU/RAM limits on one
RTX 3090. Additional maximum scratch is 64.5 MiB, reused between layers.
No existing user Pi conversation/logs were inspected; only isolated test
outputs and telemetry from requests submitted to this freshly started service.

Clean Windows MSVC Release/CUDA 13.4, SM86 build: five selected CTest targets
pass; canonical Python suites run 99 tests with one skipped. Thirteen MLA
cases use an independent materialized-K/V FP64 oracle and scratch canaries;
maximum relative L2 is 0.000437592 (limit 0.006). The full-length gate uses an
independent FP64 uniform-attention mean and GPU-event timings over three calls
following warm-up, at actual 32-head/256-rank/128-value geometry.

| Populated single-layer fixture | Existing tiled kernel, ms | Split kernel, ms | Speedup |
|---|---:|---:|---:|
| 8192 | 4.10105 | 0.594837 | 6.8944x |
| 65536 | 32.5001 | 2.95734 | 10.9896x |
| 262144 | 124.515 | 9.67058 | 12.8756x |

This qualifies the declared >=2x attention prerequisite at 64K and 256K.
These timings exclude model prefill, dense projections, routed experts,
storage and client overhead; 256K here is a one-layer fixture, not a populated
256K whole-model capacity/coding pass. Long-context F32 reduction order differs;
no bit-identical logit or real coding quality result is claimed.

The running executable hash was independently verified as
`aa3bd5b2d659c314254565c4e6b61395f1f70f65fbd7e41f4f67ada9e0bc4566`.
Direct `hi`, then ephemeral Pi `hi`, then Pi `hi` with an inert C++ reference
system context executed sequentially on this binary at xhigh:

| Real request | Direct cold hi | Pi hi after direct | Pi with reference context |
|---|---:|---:|---:|
| Populated prompt tokens | 541 | 486 | 4408 |
| Generated / reasoning / visible tokens | 69 / 57 / 12 | 54 / 34 / 20 | 85 / 69 / 16 |
| Prefill, s | 60.109 | 35.953 | 166.468 |
| First generated token, s | 60.437 | 36.406 | 167.172 |
| Total server wall, s | 85.484 | 53.359 | 205.718 |
| Storage-read bytes | 70,374,563,840 | 28,956,938,240 | 52,377,366,528 |
| H2D bytes | 125,915,841,000 | 114,265,617,600 | 185,879,809,800 |

All responses finish with `stop` and a coherent greeting, without tools or
edits. The 4408-token request exercises split decode with the provider's
scratch and allocated 18 KV pages. Its approximate generation wall rate is
2.17 tok/s including reasoning; useful visible end-to-end rate is only
16/205.718 =0.078 tok/s. Prefill remains slow. Short Pi hi takes 53.359 s versus
53.000 s on the preceding pinned candidate; those below-threshold requests
retain the existing path. No comparable old-binary 4408-token service run was
performed, so no whole-model acceleration is inferred. The user's reported
0.1 tok/s and interactive coding failure remain unresolved.

Mistral gate cleanup confirms zero native workers, no 8080 listener and
336 MiB desktop GPU residency. Own-request and native evidence is under
`out/mla-context-splits/`. The common runner also passes Qwen direct `model.sh chat` hi at xhigh:
53 prompt tokens, 30 generated (18 reasoning), 1.959 s first visible and
2.286 s client total, `stop`. Its running executable hash matches the Mistral
candidate. Subsequent cleanup again confirms zero workers, no 8080 listener
and 336 MiB desktop GPU residency. These are short regression/transport gates,
not Qwen's populated-context coding gate.

Final deployment on 2026-10-08 leaves Mistral running, `ready=true`, context
262,144, BF16 latent KV, MTP depth zero and no active test request. Process
18652 executes the candidate hash above. The final startup is a readiness/hash
verification after qualified chat gates and cleanup, not an additional cold
Pi measurement. Pi defaults remain unchanged. This promotion retains the
kernel improvement without declaring the whole-model coding target achieved.

## Authorized Mistral deployment and Pi greetings, 2026-10-07

The user authorized deployment, Pi `hi` and further improvement after the
offline MLA/NVFP4 candidate. Tests use one RTX 3090, the existing
`mistral-small-4-119b-nvfp4` artifact, native W4A4 NVFP4, BF16 latent MLA,
262,144 configured context, xhigh, zero MTP, 12 GiB routed VRAM and 48 GiB
routed RAM. The direct-then-Pi sequence is repeated after each restart; Pi
uses an ephemeral session with tools/context files/extensions disabled.
The Pi default model is unchanged. This is a real client wiring/latency gate,
not a coding, tool-use or populated-256K qualification.

| Measurement | MLA/grouped candidate | Pinned RAM-upload candidate |
|---|---:|---:|
| Direct prompt / generated / reasoning tokens | 541 / 69 / 57 | 541 / 69 / 57 |
| Direct prefill / first token / total, s | 60.468 / 60.797 / 85.828 | 59.859 / 60.187 / 85.109 |
| Pi prompt / generated / reasoning tokens | 486 / 54 / 34 | 486 / 54 / 34 |
| Pi prefill / first token / total, s | 36.985 / 37.313 / 54.578 | 35.656 / 36.000 / 53.000 |
| Pi storage-read bytes | 28,914,458,624 | 28,914,458,624 |
| Pi H2D bytes | 114,336,396,600 | 114,364,708,200 |
| Pi summed upload wait, s | 14.453 | 17.013 |

Greetings and reasoning text are identical across these two candidates.
The 2.9% Pi wall-time difference is one observation; increased summed upload
wait does not establish a bandwidth improvement. Approximate generation wall
rates including hidden reasoning are 3.07 and 3.11 tok/s; useful visible
end-to-end output is only 20 tokens over 54.578 and 53.000 s. Cold storage,
prefill, decode and visible latency remain independent gates. Three allocated
256-token KV pages on these short requests do not populate configured 256K.

The pinned candidate was built clean with Windows MSVC Release/CUDA 13.4 and
passed 4/4 CTest targets (cache ownership and existing regressions, MLA,
native NVFP4 and standard-FP4 grouping). Canonical Python suites passed
99 tests with one skipped. Running executable SHA256 was independently read:
`6d198d3924f56375c4c154cef54554500359f7f10942ce014d86c03ba9830e01`.
Both deployment stops released native workers and GPU residency to 356 MiB.
Own-request evidence is in `out/mistral-live-improvement/`.

A further RAM-victim ranking trial reduced Pi storage reads to
22,259,318,784 bytes and prefill to 27.985 s, but increased generation wall to
22.672 s and total to 50.657 s. Its cold direct chat rose to 91.078 s; combined
direct+Pi wall regressed to 141.735 s versus 138.109 s with pinned staging.
The declared 10% Pi improvement prerequisite failed. The trial was rejected;
its cache policy and test were removed by restoring the exact 277-file
clean-built pinned source snapshot. See the negative result in
[Research decisions](research-decisions.md).

The retained common runner subsequently passed Qwen direct `model.sh chat`
`hi` at xhigh: 53 prompt tokens, 29 generated (17 reasoning), 1.723 s first
visible and 2.150 s client total. This is a short regression smoke, not the
populated-context or useful-output Qwen product gate. No existing Pi session
was read or modified; only ephemeral test outputs and own-request telemetry
were inspected. Mistral weights/KV/context/cache limits and the Pi default
remain unchanged.

Final deployment verification: the selected pinned runner was restarted as
Mistral, its executable hash matched the qualified build, and an ephemeral Pi
xhigh `hi` returned the identical greeting with 486 input / 54 generated /
34 reasoning tokens and `stop`. This first request after restart had a cold
expert cache: 57.406 s prefill, 58.125 s first token and 80.375 s total, with
70,601,121,792 storage-read bytes and 113,727,697,200 H2D bytes. It is not
comparable to the 53.000 s Pi request following a separate direct warm-up.
Mistral remains slow for interactive use; no large-context coding pass is
claimed. The final API reports `ready=true`, context 262,144, BF16 latent KV
and MTP depth zero. Qwen gate cleanup confirmed zero native workers, no
8080 listener and 347 MiB desktop GPU residency before the final deployment.
Mistral is deliberately left running for the user after this verification.

## User-reported Mistral service failure, 2026-10-07

The user reports Pi displaying approximately 0.1 tok/s for
`mistral-small-4-119b-nvfp4`. This is a user observation, not an independently
measured decode gate. Populated prompt, prefill/TTFT, reasoning/visible token
split, route/cache traffic and instantaneous residency were not inspected.
No Pi/server logs, live requests, lifecycle operations or benchmarks were run.
The existing coding gate already records 14 valid tools and zero edits at a
30-minute cutoff; isolated wiring smokes do not qualify interactive throughput.

Read-only source/config inspection confirms the selected path is the separate
QPack/native worker, with 12 GiB routed VRAM and 48 GiB host-cache limits.
Authenticated artifact metadata declares no MTP and an artifact-native BF16 latent-KV
minimum of 23,040 bytes/token: 5.625 GiB at 262,144 populated positions. This
does not establish the reported session's population or actual placement.
Source tensor payload is 70,801,904,048 bytes; it is not all GPU resident.

The NVFP4 dense projection path uses custom activation quantize/dequantize
and scalar decoded-weight CUDA accumulation in
`runtime/src/cuda/transformer_kernels.cu`, rather than an NVFP4 Tensor Core
GEMM in that function. Routed expert misses acquire/cache/upload weights;
the standard-FP4 grouped prefill path explicitly excludes native NVFP4.
These are concrete performance risks, not proof of the dominant bottleneck
or a quantitative explanation of 0.1 tok/s. The missing-PyTorch warning is
unrelated to these native execution paths. No runtime optimization or memory
policy change was made; Mistral remains unqualified for interactive coding.

## Mistral MLA/NVFP4 candidate, 2026-10-07

The requested optimization candidate was compiled on the RTX 3090 host with
MSVC Release/CUDA 13.4, SM86 and `--clean-first` in an isolated source/build
snapshot. This is numerical/build evidence, not interactive coding throughput.
The active service was not restarted, probed or sent inference requests;
Pi/server logs were not read. The local `.env` now selects this common VM
candidate for the user's next `model.sh` start/restart. No model weights, KV
format, context, MTP count or cache ceilings changed.

Candidate changes:

- MLA scores eight causal positions per tile, sharing decoded latent values;
- native NVFP4 routed prefill groups up to four selections of the same expert,
  preserving separate activation scales and BF16 rounding boundaries;
- MLA-only operation sets omit unused standard-attention arenas. The computed
  reservation reduction is 798.125 MiB at 512 rows, 32 heads, 160 layout width,
  8192 staging positions and configured 256K context. This is allocation
  arithmetic, not a measured service VRAM delta. Mixed attention sets retain
  their standard arenas.

| Offline gate on the clean candidate | Result |
|---|---|
| grouped standard-FP4 regression | passed; maximum absolute error 0, cosine 1 |
| native NVFP4 independent CPU reference | passed; maximum absolute error 0 in the 64x64 fixture |
| native NVFP4 grouped vs scalar | bit identical for 6/8/12-row, top-2, mixed-expert/masked/tail fixtures |
| MLA materialized-K/V FP64 CPU oracle | 8 cases passed, including exact 32-head/256-rank/64-nope/64-rope/128-value geometry |
| MLA greatest tested prefix | 8191 + 1 current position; not a 256K service gate |
| MLA worst relative L2 across fixtures | 1.8148e-05 |
| CTest selected changed/regression targets | 3/3 passed |
| canonical Python suites | 99 run: 98 passed, 1 skipped |
| real Mistral chat / Pi / throughput / live 256K | not run; user owns restart and real-model testing |

The Python contract initially failed because an existing metadata fixture
omitted the previously added `mtp_draft_depth: None` field. Updating that
expected contract made the canonical retest pass; no Python service behavior
was changed for this optimization.

Candidate executable SHA-256:
`ef9920b1f85371fdc6456c590e18adb2e46bce6c89c2694f37980dff7a500993`.
Build source archive SHA-256:
`110bd8b9d8f6b22d481eeaf57e53b1ca9c8bdf56ac632c9e47ca710c7e3a7d8a`.
Evidence: `out/mistral-optimization-audit/{deployment.json,native-result.json,
native-gates.log,complete-candidate-build.log,python-contract-retest.log}`.
This remains a candidate for user testing, not a production-qualified release
or a measured fix for the reported 0.1 tok/s.

## Reporting contract

- TTFT is server time to the first generated token unless labelled otherwise.
- Decode rate excludes TTFT and the first token.
- Prompt, reasoning, visible and total generated tokens are separate.
- Configured context and populated context are separate.
- Cold, retained-prefix, disk-restored and warm runs are separate.
- A kernel or numerical gate is not service throughput.
- All rates below are single-request measurements unless stated otherwise.

## Validation baseline

The last documented Windows gate used MSVC 19.44 and CUDA 13.4 with a verified
clean build directory and `--clean-first`.

| Gate | Result |
|---|---|
| CTest | 8/8 passed |
| canonical Python suites | 99 passed |
| compiler/service tests in the current checkout | 99 passed, one optional NumPy skip |
| benchmark helper tests in the current checkout | 22 passed |
| build warnings | 14,992 from vendored CUTLASS/FlashAttention; zero first-party warnings |

The six models in the Pi catalog each returned visible text from an isolated
`xhigh` `hi` request on that build. This qualifies service, tokenizer,
template and generation wiring only. One Ornith request produced a coherent
answer inside reasoning without visible text; a later temperature-0.6 series
produced visible text in 30/30 attempts, but that sampling policy has not been
published.

## Artifact evidence

| Artifact | Validated payload evidence | Advertised context |
|---|---:|---:|
| Qwen3.8-27B | 14,775,390,208-byte QPack; 1,199 records | 262,144 |
| Qwen3.8-27B Abliterated | 14,775,390,208-byte QPack; separately published third-party weights | 262,144 |
| Qwen3.8-Flash-Next | 95,915,634,688 bytes; 1,562 dense and 24,576 expert records | 262,144 |
| Mistral Small 4 | 70,801,904,048 validated source-tensor bytes; native NVFP4/BF16 | 262,144 |
| Muse-Glimmer | 15,832,002,560-byte QPack | 131,072 |
| Ornith | 19,229,777,920-byte pack | 262,144 |

Source reconstruction is necessary but does not qualify provider execution,
model behavior or maximum context.

## Qwen capacity

```text
hot target weights             13,625,700,352 bytes = 12.690 GiB
exact F16 KV at 262,144        17,179,869,184 bytes = 16.000 GiB
subtotal                       30,805,569,536 bytes
RTX 3090 physical              25,769,803,776 bytes
```

The measured pinned-host link was 12.46 GiB/s. Scanning the full 16 GiB host KV
therefore has a 1.284-second transfer floor for one scalar target call.

Q4H stores a 260-byte K/V pair for each layer, KV head and token:

```text
target KV at 262,144 positions   4.062500 GiB
Q8 MTP pages                     0.503906 GiB
combined                         4.566406 GiB
```

Q4H is lossy relative to F16.

## Maximum populated context

A deterministic prompt placed retrieval and code facts at positions 2,068,
131,013 and 196,510 and queried them at 261,904.

| KV path | Populated prompt | Output | Wall | Result |
|---|---:|---:|---:|---|
| historical K1 | 262,016 | 47 | 546.173 s | retrieved all facts |
| exact F16 | 262,016 | 47 | 2,278.082 s | byte-identical output |

Both responses made the same unrelated arithmetic error. This single parity
case does not establish general K1 fidelity. Exact F16 proves capacity and
fails the interactive latency target.

The exact hierarchical top-k implementation reduced sampling from an observed
9.916 ms per target call to 0.258 ms in the worker trace. On the populated K1
request, post-first decode improved from 14.39 to 17.07 tok/s. This is an exact
sampling improvement; K1 remains lossy and is not an active default.

## Operational Q4H and MTP-4

The Qwen exact-decode ABI verifies one guaranteed target position plus up to
four MTP proposals in one target batch. Target sampling uses exact `p/q`
rejection and residual correction. Target KV is Q4H; proposer KV is Q8.

Live service gates used suffix windows from the same 262,016-token coding
prompt, 128 generated tokens, `temperature=1.0`, `top_p=0.95`,
`top_k=20` and seed 1. Session retention was disabled.

| Populated prompt | TTFT | Post-first decode | Target calls / positions | Positions/call |
|---:|---:|---:|---:|---:|
| 32,768 | 39.627 s | 54.54 tok/s | 48 / 128 | 2.6667 |
| 131,072 | 203.250 s | 47.16 tok/s | 42 / 128 | 3.0476 |

Only those two measured points produce this linear estimate:

```text
estimated 262,144 decode = 39.94 useful tok/s
```

No live Q4H 262K service gate has passed.

### External endpoint checks

Both tools called the real OpenAI-compatible endpoint with one active request,
no session retention and the artifact thinking sampler.

| Tool | Actual prompt | Output | TTFT | Post-first decode |
|---|---:|---:|---:|---:|
| llama-benchy 0.4.0 | 34,817 | 32 | 42.131 s | 56.99 tok/s |
| GuideLLM 0.7.4, concurrency 1 | 32,820 | 128 | 39.861 s | 60.58 tok/s |

GuideLLM generated reasoning only and measured 3.05 tok/s end to end including
prefill. The two decode rates differ with MTP acceptance. Neither run is a
quality or latency-distribution result.

## Long-agent and language quality

Two Qwen Abliterated Q4H Pi requests entered lexical loops:

| Prompt | Generated before cancellation | TTFT | Wall |
|---:|---:|---:|---:|
| 133,610 | 12,952 | 208.906 s | 449.750 s |
| 148,441 | 79,608 | 240.218 s | 1,533.172 s |

A coherent 4,786-token request completed between them. The evidence rejects a
claim of reliable long-agent behavior and does not isolate Q4H because no
matched codec replay was captured.

The current artifacts use `presence_penalty=0.5` and a 32,768-token thinking
circuit breaker. The earlier value 1.5 was removed after visible Romanian
degradation.

A controlled Romanian matrix crossed official and Abliterated weights, Q4H and
F16 target KV, MTP-4 and scalar target decode, presence penalties 0.5 and 0.0,
and temperatures 1.0 and 0.6. Material malformed Romanian remained in every
row. This rules out target-KV format, MTP and sampling policy as sufficient
fixes. It does not identify whether the remaining cause is source behavior,
stored FP4 weights or provider numerics.

An English synthetic request generated 39 complete numeric records and 38
complete prose records without observed numeric corruption, then hit the
32,768-token completion limit before record 40 and its checksum. It is a
bounded negative-corruption observation, not a general fidelity pass.

## Coding harness

| Model/path | Result |
|---|---|
| Qwen F16, effective `medium` | Todo fixture 3/3 in 503.240 s; about 17.8K final prompt |
| historical Qwen K1, `xhigh` | Todo fixture 3/3 in 979.17 s |
| Qwen Flash, corrected `medium` | valid tools, zero edits; exceeded 1,000 s |
| Mistral | 14 valid tool calls, zero edits at 30-minute cutoff |
| historical Ornith K1+fit, `xhigh` | 2/3 tests after 1,832.80 s |
| Muse | no representative coding result |

These runs use different model paths and reasoning modes. They establish only
the result stated in each row.

## Pi 256K benchmark

The current benchmark records every completed Pi model call at its actual
populated prompt size. It uses a gradual offline-first conference application,
real file operations and live web retrieval through Pi's standard PowerShell
tool. The terminal gate is a prompt from 255,000 through 255,999 tokens.

The latest Quantum run is incomplete:

| Metric | Result |
|---|---:|
| completed model calls | 25 |
| maximum populated prompt | 58,664 |
| generated / reasoning / visible tokens | 50,323 / 29,613 / 20,710 |
| provider prefill / decode wall | 1,327.951 / 719.779 s |
| weighted generated-token decode | 69.915 tok/s |
| visible end-to-end rate through last completed call | 9.950 tok/s |

The next call emitted 690 thinking deltas and then produced no event, tool call
or provider completion for more than two hours while the GPU was idle. The run
was stopped and cleaned up. No 256K result, stall root cause or correctness
pass exists.

All 25 completed calls reported full-prompt prefill and
`session_retain_skipped` because an unemitted speculative bonus prevented an
exact client-visible retained prefix.

### Pinned llama.cpp comparison

The comparative Pi benchmark ran the two backends sequentially on the same RTX
3090 with the same English dialogue, Pi `xhigh` and a configured 256,000-token
window. Quantum used official Qwen3.8-27B standard FP4 weights, Q4H target KV
and MTP-4. The comparison used pinned llama.cpp v0.5.0 with the official
Q4_K_M GGUF, `q4_0` target KV and an MTP-4 draft. The formats are not
numerically equivalent.

The opening `hi` call populated exactly 2,239 prompt tokens on both backends:

| Metric | Quantum LLM | llama.cpp | Observed ratio |
|---|---:|---:|---:|
| generated tokens | 46 | 37 | — |
| client TTFT | 2.813 s | 6.281 s | llama.cpp took 2.23× longer |
| provider prefill | 2.687 s | 6.061 s | llama.cpp took 2.26× longer |
| post-first decode | 62.670 tok/s | 6.740 tok/s | Quantum reached 9.30× the rate |
| complete client call | 3.547 s | 11.765 s | llama.cpp took 3.32× longer |

The other two completed llama.cpp calls decoded at 8.390 and 6.672 tok/s. The
run was stopped because its early rate did not justify continuing the
long-context dialogue. It reported 48/65 target layers and 66/66 draft layers
on GPU, 5,727 MiB of CPU-mapped model buffers and 1,125 MiB of CPU KV. Its
Q4_K_M weights plus Q4_0 target KV, MTP weights and the mandatory 1 GiB reserve
have a 24.630 GiB lower bound before workspaces, so full placement cannot fit
in 24 GiB.

Quantum completed 25 calls, reached 58,664 populated prompt tokens and recorded
69.915 generated tok/s weighted across completed decode. It later stalled with
an idle GPU before reaching the terminal context gate. Neither run reached
256K, and the difference in formats prevents a numerical-fidelity comparison.
The 9.30× figure applies only to the same measured opening call.

## Sparse and auxiliary paths

### Qwen Flash cold prefill

One real Pi prompt populated 4,190 tokens with exact F16 KV and the 12 GiB
routed-VRAM ceiling.

| Metric | Prior | Current |
|---|---:|---:|
| prefill batches | 10 | 1 |
| storage read | 158.8 GB | 62.03 GB |
| H2D upload | 236.8 GB | 61.98 GB |
| TTFT | 325.3 s | 163.640 s |

The grouped standard-FP4 route preserved the exact selected experts and passed
its independent numerical oracle. The request remained too slow for an
interactive coding claim.

### P100 admission

The two P100 PCIe cards have no peer access with the RTX 3090.

| Proposed path | Best result | Required | Decision |
|---|---:|---:|---|
| exact 262K attention split | 77.101 ms compute-only | <=32 ms | rejected |
| compact dense shard | 35.939 ms/card before integration | <=20 ms/card | rejected |
| lossless F16 MLP shards | 307.7/311.0 GB/s | >=500 GB/s/card | rejected |
| Qwen Flash selected experts | 0.20 tok/s | beat 0.37 tok/s primary | rejected for speed |

The secondary executor remains a capacity mechanism for compatible routed
pages.

## Sessions and cleanup

| Gate | Result |
|---|---|
| two exact-F16 sessions | four park/restores; suffix-only prefill; 351,952,896 parked bytes |
| alternating Pi sessions | about 6,741 tokens/session; 1,223,811,072 parked bytes |
| cancellation | active count returned to zero; committed prefix survived |
| progressive growth | 200 -> 300 -> 300 used 300 cumulative prefill tokens |

A historical K1 Qwen restart/resume gate persisted 443 tokens in 169,443,798
bytes. After a complete service restart it restored in 0.203 s, prefilled a
34-token suffix and completed in 0.938 s. Fixed-seed continuation produced the
same next text before and after restart. A large real coding-session
persistence gate and a matched current-Q4H restart gate remain unrun.

Recorded gates stopped the service and verified process, port and GPU cleanup.

## Current conclusion

- six published models pass isolated wiring smokes;
- Q4H plus MTP-4 provides fast 32K and 128K Qwen decode;
- no live Q4H 262K result exists;
- exact-F16 262K capacity works and its latency fails the product target;
- long-context Q4H behavior and Romanian fidelity are unqualified;
- the autonomous Pi run has not reached its 256K terminal gate;
- Flash, Mistral, Muse and Ornith remain below their complete qualification
  gates;
- persistent exact-prefix continuation works on the measured small Qwen gate;
- the P100s have not improved measured throughput.

## Qwen MTP-10 startup investigation, 2026-10-02

The user reported `ops/model.sh start` waiting after task installation and
explicitly authorized running the startup to investigate. The selected
artifact remains Qwen3.8-27B QPack, Q4H target KV/Q8 MTP KV, configured
262,144 context, one RTX 3090 and the fixed 12 GiB routed ceiling. Runner:
`out/build/windows-msvc-release-mtp-env-20261002/runtime/Release/expert-moe-vm-runner.exe`
(SHA256 `9f1b608957025ca55f226f0df20bb6133b50eea45d91af6a7195e6ca2c7c44a0`).

The original attempt reached native `worker_ready` but had no recorded Python
`service_ready` before the user stopped the task. Windows result `0x41306`
was that user termination, not evidence of a model crash. Its startup hang
was not reproduced. An initial unauthenticated HTTP diagnostic did not
establish authenticated API readiness and was corrected.

Starting the existing task subsequently reached `service_ready`. Then the
exact full command `./ops/model.sh start qwen` completed with exit code 0,
`ready=true` and no advertised-contract mismatches. Python `service_ready`
was recorded 16.127 seconds after that task's last-run timestamp. The
running API confirms worker protocol 13, MTP enabled, and draft depth 10.
This is task-to-service startup timing, not cold/reused prefill, TTFT,
decode speed, populated-context capacity or a coding harness result. No
inference request, numerical test, coding benchmark or unrelated model gate
was run. The service is left running for the user. No runtime corrective
patch was made: a causal diagnosis of the initial intermittent hang remains
unconfirmed rather than being replaced by a speculative transport change.
