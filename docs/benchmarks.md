# Benchmarks and evidence

Status: canonical measurement ledger, 2026-09-21.

## Reporting rules

- TTFT is server time to first generated token unless marked first-visible.
- After-first rate excludes TTFT and the first token.
- Prompt, reasoning, visible and total generated tokens are distinct.
- Configured, populated and harness-provided context are distinct.
- Cold, warm, prefix-reused, novel-route and settled-route runs are not
  interchangeable.
- A numerical gate or microbenchmark is not service throughput.

## Validation baseline

### 2026-09-25 CUDA 13.4 clean build and isolated Pi `xhigh` smoke

The Windows MSVC 19.44 / CUDA 13.4 Release build used a verified clean build
directory and `--clean-first`. CTest passed 8/8 and the canonical remote Python
suite passed 99 tests. The build still emitted 14,992 warnings, all from the
vendored CUTLASS/FlashAttention headers; first-party CUDA/C++ and generated
stub warnings were eliminated. This is **not** a zero-warning release gate.

After this build, each of the six models in the remaining `quantum-llm` Pi
catalog was started sequentially on the RTX 3090 and answered `hi` through Pi
at `xhigh`. Pi ran from an empty temporary directory with `--no-session`,
`--no-context-files`, `--no-extensions`, `--no-skills`, and
`--no-prompt-templates`. No additional output-token ceiling was applied;
Pi's built-in system instructions still contributed prompt tokens.

| Pi model | Isolated `xhigh` `hi` |
|---|---|
| `qwen3.8-27b-fp4` | visible coherent answer |
| `qwen3.8-27b-abliterated-fp4` | visible coherent answer |
| `qwen3.8-flash-next-fp4` | visible coherent answer; 1,655 prompt and 144 generated tokens, 113.672 s server wall |
| `ornith-1.5-35b-a3b-fp4` | visible coherent answer |
| `muse-glimmer-30b-fp4` | visible coherent answer |
| `mistral-small-4-119b-nvfp4` | visible coherent answer |

An earlier direct Ornith `xhigh` `hi` generated 19 reasoning tokens, stopped
normally, and produced no visible text. A later default-temperature series
reproduced this at attempt 8: the model emitted a coherent final reply inside
`reasoning_content` and stopped without `</think>`. The response parser did not
lose a final-channel token. With only temperature changed from the artifact's
`1.0` to the author's [general-use recommendation of `0.6`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
30/30 independent uncapped `xhigh` `hi` requests produced visible text. This
supports an artifact sampling-policy candidate, not a guarantee against future
empty replies. The policy has not yet been published: the existing artifact
lacks current placement metadata, so a fresh source compile is required.
The separate `pascal-llm` provider's first Pi
request timed out, and TCP to its shared `10.10.88.7:8080` endpoint timed out;
its seven configured models were not qualified by this run.

The prior native changed-code gate used the Windows Release/CUDA build
(MSVC 19.44, CUDA 12.1), built `--clean-first --parallel 22`. Windows CTest
passed 7/7, including the independent grouped standard-FP4 CUDA smoke, and the
canonical compiler/server suite ran 119 tests successfully. The telemetry-only
server follow-up then passed the full 119-test canonical suite locally, with
nine optional NumPy skips. Python byte compilation and `git diff --check`
passed. The Windows binary is authoritative.

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

#### Default-policy isolated Pi regression, 2026-09-20

After promoting `q4-f16-per-head` for every compatible artifact, six published
artifacts were started sequentially on the RTX 3090. Each start disabled
server-side session retention and reported `ready=true` with the required KV
codec. Pi ran from a temporary directory outside the repository with
`--no-context-files --no-tools --no-extensions --no-skills
--no-prompt-templates --no-session --thinking off -p hi`. No project context,
tool, extension, skill, prompt template or prior session was available. The
values below are client wall time, not service throughput measurements.

| Alias | Required and reported KV | Client wall | Result |
|---|---|---:|---|
| `qwen` | `q4-f16-per-head` | 1.87 s | pass, coherent text |
| `qwen-abliterated` | `q4-f16-per-head` | 1.42 s | pass, coherent text |
| `ornith` | `q4-f16-per-head` | 8.92 s | pass, coherent text |
| `muse` | F16 | 3.58 s | pass, coherent text |
| `qwen-flash` | F16 | 48.38 s | pass, coherent text; slow |
| `mistral` | artifact-native | 59.48 s | pass, coherent text; slow |

DeepSeek was intentionally not run at the user's request; its earlier result
above remains historical evidence only. After the six gates, the common task
was stopped, `/ready` was unreachable with no model advertised, and the RTX
3090 reported 25,161,629,696 of 25,769,803,776 bytes free. This is a wiring and
cleanup gate, not a quality, coding or long-context result.

#### Canonical standard-FP4/Q4H/MTP-4 serving and isolated Pi regression, 2026-09-24

The two standard-FP4 Qwen artifacts declared exact-decode ABI 2, MTP depth 4,
65,536 draft-vocabulary entries, and Q8 draft KV. Their operational aliases
selected `q4-f16-per-head`; explicit exact-F16 and diagnostic aliases remained
distinct. The user-local Pi default selected standard Qwen at `xhigh`, and the
generic service fallback selected the artifact-declared KV format. This is an
operational lossy-Q4H selection, not qualification of the exact-F16 product goal.

Each Pi `hi` ran at default `xhigh` from a fresh temporary directory with
`--no-context-files --no-tools --no-extensions --no-skills
--no-prompt-templates --no-session`. No project context or prior Pi session was
available; Pi's built-in instructions still contribute prompt tokens. These
are wiring checks, not language-quality, long-context or throughput gates.

| Alias | Selected target KV | Isolated Pi `hi` |
|---|---|---|
| `qwen` | `q4-f16-per-head` | pass |
| `qwen-abliterated` | `q4-f16-per-head` | pass |
| `ornith` | `q4-f16-per-head`, no declared MTP | pass |
| `muse` | F16 | pass |
| `qwen-flash` | F16 | pass |
| `mistral` | artifact-native | pass |

A direct Pi `hi` with no provider/model flags also passed using the user-local
default. Direct chat `hi` passed for both standard-FP4 Qwen aliases. The
launcher had incorrectly applied `minimum_exact_kv_bytes_per_token` to lossy
Q4H, expanding the Abliterated KV pool from the configured 5,136 MiB to
16,384 MiB. The format-conditioned fix restored 5,136 MiB while preserving
the exact-format minima observed for Flash (6,912 MiB) and Mistral (5,760 MiB).

One Qwen direct-chat request after sequential model switching failed when the
native worker faulted inside `nvcuda64.dll` (Windows exception `0xc0000409`).
The service was stopped and an unchanged controlled retry passed both Pi and
direct chat. The root cause and rapid-restart reliability remain unqualified;
the single successful retry is not proof of a fix. After the final stop, port
8080 was closed, no runner process remained, and GPU usage was 292 MiB.

#### Qwen English numeric-output single request, 2026-09-25

One direct API request used `qwen3.8-27b-fp4` on one RTX 3090, the standard
FP4 artifact, Q4H target KV, enabled MTP-4, `xhigh`, no prior conversation,
and `max_completion_tokens=32768`. The 614-token prompt requested exactly 40
fictional archive entries. Each entry required a calculated year, two ISO 8601
timestamps, a retention date, a cyclic ISO country/currency-code pair, an
identifier, and source/verified values; the same numbers and codes had to recur
in 70--100 words of English prose. The exact request and raw JSON response are
saved under `out/gates/qwen27b-numeric-english-20260925-{request,response}.json`.

| Gate | Observed result |
|---|---|
| HTTP / client wall | 200 / 459.528 s |
| completion / reasoning tokens | 32,768 / 24,108 |
| finish reason | `length` |
| complete machine-readable lines | 39/39 correct for records 1--39 (years 1985--2023) |
| complete prose paragraphs | 38/38 repeated the required ID, year, source, verified, retention date and ISO codes correctly; no unexpected numeric runs |
| truncation | record 39 prose ends mid-sentence; record 40 and the requested checksum are absent |
| text encoding | no non-ASCII decimal digits or replacement characters in visible content |

The observed output provides no evidence of numeric corruption in the
completed portion, including late records. It does **not** pass the requested
40-record completion gate: `xhigh` reasoning consumed most of the output
allowance, leaving 8,660 completion tokens outside the reported reasoning
count. One synthetic request cannot establish general numeric fidelity or
separate model-weight error from runtime error.

#### Planned Pi 256,000-token comparative dialogue, 2026-09-25

Status at 2026-09-25: **the latest Qwen run was stopped after a model-call
stall; no 256K result exists**. Its complete remote result directory was
copied to
`out/benchmarks/pi256k/quantum-interrupted-stall-20260925` locally; all
49 copied files matched the remote originals by SHA-256, and the originals
remain on `3090box`. The local archive has a per-call table and file guide in
its `README.md`. This was Qwen3.8-27B standard FP4 weights, Q4H target KV,
MTP-4, Pi `xhigh`, one RTX 3090, and a configured 256,000-position context.
Three user turns completed (`hi`, live research, conference-app coding), with
25 completed model calls and a maximum measured populated prompt of 58,664
tokens. The unfinished fourth turn asked for iCalendar import and live RFC
5545 guidance; it began at 14:25:06 UTC, emitted 690 thinking deltas (2,442
characters) through 14:26:37 UTC, then produced no more events, tool call,
visible text, or provider completion telemetry for over two hours. The GPU
was idle at 0% while the process and HTTP connection remained alive. This is
an observed stall, not evidence of a text loop or a known root cause. The
user-requested stop removed the benchmark process tree; port 18080 was free,
GPU memory returned to 292 MiB used, and the task was unregistered. No
`COMPLETE.json` was produced.

Across only the 25 completed calls, 50,323 generated tokens comprised 29,613
reasoning and 20,710 visible tokens. Provider prefill consumed 1,327.951 s
and decode 719.779 s; weighted generated-token decode was 69.915 tok/s.
The 2,081.438 s elapsed through the last completed call gives 9.950 visible
tok/s end-to-end, excluding the unfinished call and the subsequent stall.
All 25 calls had `resumed=false` and full-prompt prefill; all 25 provider logs
said `session_retain_skipped` with `reason=speculative_bonus`. This means no
retained Pi prefix, **not** necessarily cold NVMe reads. Paging measurements
were unavailable (`null`) in these call records. A single 27,230-token
completion (24,434 reasoning) at prompt 4,814 jumped the next prompt to
35,291. Correctness is still `pending_review`; this incomplete run does not
pass the long-context or quality gate.

Earlier Qwen launches were also stopped and archived. The third partial run is preserved on the 3090
host under `out/benchmarks/pi256k/quantum-aborted-checkpoint-redesign-20260925`.
Its first two turns passed: `hi` (2,239 populated
prompt tokens) and the research request (3,362 populated prompt tokens,
42.234 s end-to-end, 2,143 completion tokens, 1,403 reasoning tokens). The
research turn used Pi's built-in PowerShell tool and returned live GitHub
repository URLs and a fetched W3C URL. Pi `xhigh`, compaction-off and the
request-level canonical sampling parameters were confirmed in its state.
The third turn entered a long tool/model loop. The former controller wrote
checkpoints only after a full user turn, so its first reported 32K crossing
was 47,116 tokens after 27,705, not a 32K measurement. Raw Pi and provider
logs can support per-call reconstruction, but cannot retroactively create an
unobserved 32K prompt. The benchmark-owned controller, Pi and server process
tree was stopped; port 18080 was free and the RTX 3090 returned to its idle
allocation. The task was unregistered without deleting the archived evidence.
The first revised run proved per-call persistence but correctly failed its 4K
gate: consecutive prompt sizes were 3,636 and 5,656 around a live research
tool call. The 5,656-token call was recorded as `missed_overshoot`; it is not
a 4K speed result. The first call generated 1,831 tokens (1,267 reasoning)
before invoking tools, explaining most of the jump. The run cleaned its
port/GPU and is preserved as
`out/benchmarks/pi256k/quantum-failed-4k-checkpoint-20260925`. A revised
dialogue inserted brief ordinary follow-ups before live research and coding;
the strict 4K gate was still in force for that run.
The next run measured 4K at 4,466 populated prompt tokens, 6K at 6,191,
and 8K at 8,189. Its first coding call then produced 9,206 completion
tokens, including 9,158 reasoning tokens, before a tool call. The following
actual prompt was 17,422 tokens: 10K was missed by 7,422 and the run stopped
cleanly. The 17,422-token call is a near-16K observation, but it does not
replace the missed 10K gate or establish a complete benchmark. The second
partial result is preserved locally and on the host under
`out/benchmarks/pi256k/quantum-failed-10k-checkpoint-20260925`; port/GPU cleanup
passed. This shows that a fixed, natural `xhigh` Pi dialogue can skip a
requested context size by many thousands of tokens in one model call.

The first Qwen service loaded with the requested FP4/Q4H/MTP-4 formats and
256,000-token context, but Pi 0.87.1 clamped the unlisted `xhigh` level to
`high`. The task failed its state gate before sending `hi`, stopped its own
service, and released its GPU allocation. The partial run is preserved on the
3090 host under `out/benchmarks/pi256k/quantum-failed-xhigh-clamp-20260925`.
The benchmark-local provider definition now declares `thinkingLevelMap.xhigh`
and request-level sampling, including `presence_penalty=0.5`. A second launch
confirmed Pi `xhigh`, disabled compaction and those model sampling parameters.
The `hi` turn completed with 1,756 populated prompt tokens, 27 completion
tokens (15 reasoning, 12 visible), 2.125 s cold prefill, 2.172 s provider TTFT,
72 generated tok/s for that one tiny turn, and 2.625 s total turn wall time.
These numbers are **not** a 32K/64K/128K/256K result. The next turn's custom
`web_search` extension hung after launching a live request despite its declared
20-second timeout; the model had finished its preceding tool call and was idle.
The run was terminated, its exact process tree and GPU allocation cleaned up,
and the partial data preserved under
`out/benchmarks/pi256k/quantum-aborted-web-search-20260925`. No format or model
throughput conclusion follows from this tool failure. The custom extension has
been removed from the active candidate in favor of Pi's built-in PowerShell tool,
whose public-API request succeeded in the host preflight. This is a separate comparative benchmark, not
the project's exact-F16/262,144-token acceptance gate. The benchmark uses one
RTX 3090 and runs backends sequentially. It starts with official
`qwen3.8-27b-fp4:xhigh` (standard FP4 weights, Q4H target KV, MTP-4), then
tests the official [ggml-org GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/tree/main)
Q4_K_M artifact with pinned [llama.cpp v0.5.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.5.0).
GGUF Q4_K_M weights, llama.cpp `q4_0` KV and draft MTP-4 are **not**
numerically equivalent to QPack FP4, Q4H and its MTP-4 implementation. Report
the formats and actual CPU/GPU placement alongside every speed comparison.

The English Pi dialogue begins with exactly `hi`, then grows through a
disposable offline-first conference-schedule app: live research on iCalendar,
time zones and accessibility, implementation, real file/tool operations,
tests, changed requirements and long-range recall. Pi uses its own built-in
PowerShell tool to fetch live public API/source pages over the network; it has
no preloaded answers or custom web-search extension. Research turn 2 must
complete a successful live request whose output includes a source URL. Pi's
built-in PowerShell tool receives a per-call timeout in the opening request;
the controller also fails and cleans up if any tool remains open beyond a
90-second wall guard. Each subsequent user message is short and
gradual; no synthetic 200K-token input block is inserted. Pi runs in an
isolated workspace and config, with `xhigh`, automatic compaction and retries
off. The active model context is **256,000 positions**, not 262K or 265K.

The current measurement policy records **every completed Pi model call at its
actual populated prompt size**. There are no intermediate target sizes,
crossing tolerances, or overshoot aborts. Analysis after the run may derive
context bands or compare nearby observations, but it must report the actual
size of every source call and never relabel a 17,422-token observation as 10K.
The first `hi` request in the coding harness already used 2,239 prompt
tokens, so 1K/2K cannot be directly observed with this harness. The only
terminal context gate is a real prompt of at least 255,000 and strictly fewer
than 256,000 tokens, leaving room for output. Configuring a 256K window is
not evidence that it was populated; if the next prompt exceeds capacity
before the terminal gate, the run ends as incomplete with all prior calls
preserved. Every completed call writes an atomic JSON file immediately after
Pi `message_end`: actual prompt and output/reasoning/visible tokens, client
and provider TTFT, cold/reused prefill seconds and tokens, prefill/decode
tok/s, client wall time and effective output rates, tools completed since
the previous call, provider telemetry and paging evidence where available.
Each turn writes its prompt, visible response, raw Pi events, provider
telemetry, actual input/output/reasoning token counts where reported, cold or
retained-prefix prefill where independently logged, TTFT, decode speed, live
tool count/time, total wall time, and `pending_review` correctness. Correctness
is not inferred from a successful HTTP response; test-command outcomes are
recorded as provisional evidence and source claims still need review. Missing
provider telemetry is saved as `null`, never silently treated
as zero. For GGUF, reported full input less log-reported prompt-eval tokens
is saved as an **estimated** reused-prefix count, not provider-proof of exact
retention; Qwen's own `resumed` telemetry supplies the exact classification.

The GGUF download is pinned to HF revision
`efbb3b1f70a21d97fd4495240648405f7228554f`: main Q4_K_M file
18,973,870,528 B, SHA256
`c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747`;
MTP Q4_0 file 1,680,271,776 B, SHA256
`c5be331f82fb61f5304adfa00ccd60c6743c8ce52255d66fa863446231d49ba4`.
Only the maintained Xet Start/Get scripts transfer and verify these files.
Their HF cache is explicitly relocated beneath `MODEL_ROOT` on the NVMe;
publication makes validated hard links under
`${MODEL_ROOT}/qwen3.8-27b-gguf-q4-k-m`. llama.cpp is pinned to the peeled
v0.5.0 commit `7fe450e19305b828c199d602c23a8337aaa1f03b` and built for
CUDA compute capability 8.6.

The full-GPU capacity lower bound at 256,000 positions is:

```text
GGUF main weights                  18,973,870,528 B
Q4_0 target KV estimate             4,718,592,000 B
mandatory VRAM reserve              1,073,741,824 B
without MTP                        24,766,204,352 B = 23.065 GiB
MTP Q4_0 weights                    1,680,271,776 B
with MTP                           26,446,476,128 B = 24.630 GiB
```

The estimate excludes runtime workspaces and allocator fragmentation. With
MTP-4 it exceeds physical 3090 VRAM by at least 0.630 GiB, so the GGUF
benchmark explicitly **allows RAM/CPU layer placement**. llama.cpp auto-fit
may adjust the number of GPU layers, but the 256,000-token context, Q4_0 K/V,
MTP-4, one visible 3090, 1-GiB fit margin and Pi `xhigh` remain fixed.
`/props` must confirm the effective context is still 256,000; a silently
reduced window fails the gate. The host had 61,014,260 KiB of free RAM at
preparation. RAM capacity is checked again at launch. CPU-executed dense
weights are hot per token, so offload can materially lower decode speed; this
is a measured comparison, not a claimed throughput optimization. The
previously measured pinned PCIe bound is 12.46 GiB/s, or at least 0.080 s
per GiB of weights transferred per token if a configuration streams them.

`placement.json` records each llama.cpp CPU-mapped, CPU-allocated and CUDA
model **and KV** buffer (MiB) and reported GPU/total layer counts, including
both target and MTP draft allocations. CPU model/KV **placement is selected**
at load, before the first populated prompt token (selection threshold 0),
not at a later 32K/64K/128K milestone. This does not claim that every mapped
weight page or reserved KV position was already resident or touched. Per-turn
JSON repeats that fixed
placement beside actual populated context, and records GPU free MiB, host
available RAM, process working set, private bytes, page faults and process
read/write transfer counters before and after the turn, with explicit deltas.
Mapped buffer size is **not** treated as physically resident RAM; working set
is reported separately. RAM growth alone is **not** called new offload, and
process read-transfer bytes are **not** automatically called NVMe traffic. The
Qwen turns also report the provider's exact demand-paging read/upload bytes,
RAM-hit and storage-miss counters, plus the first populated-prompt size where
paging was actually observed; this is distinct from GGUF's fixed CPU layer/KV
placement at startup. The raw startup/server logs are retained for auditing.
Provider-reported TTFT and
Pi-observed time to first stream delta/first visible text are separate fields;
the latter includes client/transport overhead. The same per-call
prefill/TTFT/decode definitions and
tool-loop wall measurements apply to both backends; missing fields remain
explicitly unavailable. The Qwen artifact's thinking sampling defaults are
temperature 1.0, top-p 0.95, top-k 20, min-p 0 and presence penalty 0.5;
the GGUF server is started with those same values, repeat penalty 1.0 and
prompt caching enabled. Both Pi model entries cap an individual completion
at 32,768 tokens as a runaway guard, not a target response length.

The on-host Windows task writes durable results under
`out/benchmarks/pi256k/{quantum,llama}` and survives SSH/PowerShell client
disconnect while the Windows user remains logged in. The scheduled task uses
that user's interactive token because [S4U explicitly lacks network access](https://learn.microsoft.com/en-us/windows/win32/taskschd/principal-logontype),
which would invalidate the live-search dialogue. It never restarts an existing
partial run or stops an occupied
port. `ops/windows/Invoke-Pi256kBenchmark.ps1` exposes `prepare-pi`,
`start-download`, `download-status`, `publish`, `build-llama`, `preflight`,
`launch`, and `status`; `ops/benchmarks/pi256k/fetch.sh` copies a completed result bundle
back to the local repository. Download/build/deploy remain prepared; the
archived Qwen attempts above do not establish a full-context result. The
current dialogue starts with `hi`, then live research and coding without
artificial early questions to steer context sizes; both backends must use
the same script.

The 2026-09-26 GGUF run was stopped at the user's request because the early
decode rate was too low to justify continuing the long-context dialogue. It
used the pinned Q4_K_M model, Q4_0 target K/V, MTP-4, Pi `xhigh`, and one
RTX 3090 with a configured 256,000-position window. Three completed Pi model
calls are preserved under the host's `out/benchmarks/pi256k/llama/calls/`
and copied locally to `out/benchmarks/pi256k/llama-aborted-20260926/`,
along with `PROGRESS.json`, Pi events/session, placement, and server logs.
The first `hi` call populated 2,239 prompt tokens, took 6.061 s of provider
prefill and 6.281 s to the first client delta, and decoded at 6.740 tok/s.
At the second user question (live iCalendar/accessibility research), the
last completed call populated 3,632 prompt tokens and decoded at 6.672
tok/s. The question had not completed when stopped. llama.cpp reported
48/65 target layers and 66/66 draft layers on GPU, 5,727 MiB of CPU-mapped
model buffers, and 1,125 MiB of CPU KV allocation. This is observed
placement, not proof that all mapped pages were RAM-resident or that CPU
placement alone caused the low rate. The scheduled task and its exact process
tree were stopped; port 18081 was free, GPU memory returned to approximately
its pre-run level, and the task registration was removed. There is no
`COMPLETE.json` or 256K-context result from this aborted run. Do not use the
early decode observations as a full-context or head-to-head throughput claim.

The operational sequence uses the existing remote wrapper; `launch` returns
after creating a Windows scheduled task, so the benchmark is not tied to the
SSH session. Run the second backend only after the first has finished and its
GPU/port cleanup gate passes:

```bash
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 -Action prepare-pi
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 -Action preflight -Backend quantum
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 -Action launch -Backend quantum
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 -Action status -Backend quantum
./ops/benchmarks/pi256k/fetch.sh quantum

./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 -Action start-download
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 -Action download-status
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 -Action publish
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 -Action build-llama
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 -Action preflight -Backend llama
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 -Action launch -Backend llama
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 -Action status -Backend llama
./ops/benchmarks/pi256k/fetch.sh llama
```

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

### Direct packed-Q4 batch-five attention, 2026-09-20

The experimental `qwen-q4-bfp` alias uses 166-byte K records and 134-byte V
records. The batch-five kernel reads this payload directly, expands only to
transient INT8 Tensor Core operands and never materializes an F16 KV cache.
Its independent attention oracle remained within `3.72529e-09` maximum
absolute error. The clean native gate measured:

| Populated context | 16-layer attention | Per layer | Packed bandwidth |
|---:|---:|---:|---:|
| 32,768 | 6.10714 ms | 0.381696 ms | 103.018 GB/s |
| 131,072 | 22.8332 ms | 1.42707 ms | 110.216 GB/s |

The exact same terminal coding-prompt windows and real sampled profile used by
the K1 baseline (`temperature=1.0`, `top_p=0.95`, `top_k=20`, seed 1) then
produced 128 tokens through the service path. Session retention and durable
snapshots were disabled; cleanup reported zero allocated and reserved pages.

| Populated prompt | TTFT | Post-first | Target calls / positions | Accepted drafts | Positions/call | Cycle |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 39.390 s | 46.88 tok/s | 48 / 127 | 79 | 2.6458 | 56.437 ms |
| 131,072 | 203.496 s | 33.56 tok/s | 50 / 127 | 77 | 2.5400 | 75.685 ms |

The gate artifacts are
`qwen-q4-bfp-batch5-direct-32768-suffix.json` and
`qwen-q4-bfp-batch5-direct-131072-suffix.json`. Relative to K1 MTP-4, useful
decode improved by 79.0% at 32K and 40.5% at 128K. The candidate passes the
40-50 tok/s target at 32K but not at 128K. Linear extrapolation from only the
two requested contexts is:

```text
d32  = 2.7089760 / 127 = 0.0213305 s/useful token
d128 = 3.7842662 / 127 = 0.0297974 s/useful token
d262 = d32 + (7/3) * (d128 - d32) = 0.0410870 s/useful token
rate262 = 24.34 useful tok/s
```

This is an extrapolation, not a 262K measurement. At the measured 128K
acceptance, 40 tok/s requires `63.50 ms/cycle`; the observed `75.685 ms` cycle
misses by `12.185 ms`. The measured attention call is `22.833 ms`, leaving an
observed `52.852 ms` of proposer, target dense/recurrent/head, sampling and
runtime work per cycle.

The admitted split-K INT8 Tensor Core dense kernel subsequently kept the same
FP4 artifact, Q8 activations and FP32 block-scale accumulation. Eight warps
cover disjoint K blocks for one 16x8 output tile and reduce once in 4 KiB of
shared memory. The independent scalar oracle stayed at `5.34058e-05`; clean
batch-five bandwidth improved from 453.70 to 597.287 GB/s on
17,408x5,120 and from 365.78 to 525.455 GB/s on 5,120x17,408.

The final real service gates used the same prompt windows and sampling profile
as the table above. Both output hashes were byte-identical to the pre-split-K
Q4 runs, and session cleanup again returned allocated/reserved pages to zero.

| Populated prompt | TTFT | Post-first | Target calls / positions | Accepted drafts | Positions/call | Cycle |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 39.559 s | 52.82 tok/s | 48 / 127 | 79 | 2.6458 | 50.089 ms |
| 131,072 | 203.296 s | 36.57 tok/s | 50 / 127 | 77 | 2.5400 | 69.449 ms |

The artifacts are `qwen-q4-bfp-splitk-dense-32768-suffix.json` and
`qwen-q4-bfp-splitk-dense-131072-suffix.json`. Relative to the preceding
direct-Q4 result, useful throughput improved another 12.7% at 32K and 9.0% at
128K. The requested two-point extrapolation is:

```text
d32  = 2.4042503 / 127 = 0.0189311 s/useful token
d128 = 3.4724691 / 127 = 0.0273423 s/useful token
d262 = d32 + (7/3) * (d128 - d32) = 0.0385572 s/useful token
rate262 = 25.94 useful tok/s
```

This is an extrapolation, not a 262K measurement. The 32K target passes. At
128K, 40 tok/s requires 63.50 ms/cycle and the measured cycle is 69.449 ms,
so 5.949 ms/cycle remains.

### Direct per-head-Q4 (Q4H) batch-five attention, 2026-09-20

The separately named `qwen-q4-per-head` profile stores one FP16 scale for
each complete 256-value K or V head and 128 signed-Q4 code bytes. The target
K+V record pair is 260 bytes. The target cache is 4.0625 GiB at the configured
262,144-position ceiling; the unchanged Q8 MTP cache raises the combined
capacity to 4.566406 GiB. This is explicitly lossy and is not an exact-F16
result.

The direct kernel consumes packed codes and scales without materializing F16.
Its independent host encoder/layout check was byte exact, the independent
attention oracle had maximum absolute error `3.72529e-09`, and the final clean
Windows Release/CUDA gate measured:

| Populated context | 16-layer attention | Per layer | Packed bandwidth |
|---:|---:|---:|---:|
| 32,768 | 4.54246 ms | 0.283904 ms | 120.036 GB/s |
| 131,072 | 16.5949 ms | 1.03718 ms | 131.428 GB/s |

The initial 128K result was `16.9247 ms`, 0.1720 ms above the admitted
`16.7527 ms` bound. One local correction decoded all eight Q4 values from one
32-bit word, sharing the nibble mask and shift between both four-value PRMT
expansions. It retained the oracle and reduced 128K attention to `16.5949 ms`.

Real service gates then used suffix windows from the same 262,016-token coding
prompt and generated 128 tokens with `temperature=1.0`, `top_p=0.95`,
`top_k=20`, seed 1. Session retention was disabled. The worker reported
`q4-f16-per-head`, MTP-4 ready, GPU sampling only, and zero allocated/reserved
pages after each request.

| Populated prompt | TTFT | Post-first | Target calls / positions | Accepted drafts | Positions/call | Cycle |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 39.627 s | 54.54 tok/s | 48 / 128 | 80 | 2.6667 | 48.512 ms |
| 131,072 | 203.250 s | 47.16 tok/s | 42 / 128 | 86 | 3.0476 | 64.124 ms |

The gate artifacts are `qwen-q4-per-head-32768-suffix.json` and
`qwen-q4-per-head-131072-suffix.json`. Relative to the selected block-floating
Q4 baseline, measured useful throughput improved by 3.25% at 32K and 28.93%
at 128K. Both requested live contexts pass 40 useful tok/s. The user-owned
real-project Pi fidelity gate remains pending.

Only the requested points are extrapolated; no 262K service gate was run:

```text
d32  = 2.3285583 / 127 = 0.0183351 s/useful token
d128 = 2.6932210 / 127 = 0.0212065 s/useful token
d262 = d32 + (7/3) * (d128 - d32) = 0.0250349 s/useful token
rate262 = 39.94 useful tok/s
```

#### External `llama-benchy` endpoint gate

`llama-benchy` 0.4.0 was run against the real OpenAI-compatible chat endpoint,
not against the provider directly. The service was freshly started with the
`qwen-q4-per-head` alias, session retention and durable session caching were
disabled, and the request used `temperature=1.0`, `top_p=0.95`, `top_k=20`,
concurrency one and cache avoidance. The standard shape was `pp=2048`,
`depth=32768`, `tg=32`; the server reported 34,817 actual prompt tokens after
chat templating.

| External metric | Result |
|---|---:|
| end-to-end time to first content token | 42.131 s |
| post-first decode | 56.99 tok/s |
| peak one-second decode | 58.82 tok/s |
| target calls / verified positions | 12 / 32 |
| accepted drafts | 20 |
| verified positions/call | 2.6667 |

The worker confirmed `q4-f16-per-head`, MTP-4, a 65,536-token draft
vocabulary, Q8 MTP K/V, GPU-only sampling and direct fused multi-query
attention. Cleanup left zero allocated/reserved pages and zero retained
sessions. The raw result is
`out/benchmarks/llama-benchy-qwen-q4-per-head-32k-noretention.json`.

The tool also printed 193,112 prompt tok/s from `TTFR - latency`. That value is
invalid for this endpoint because an early SSE role/metadata chunk arrived at
0.210 s while the first content token arrived at 42.131 s. It is deliberately
not promoted as a prefill measurement. This was one 32-token run, so the decode
number is an external service-path confirmation with MTP acceptance variance,
not a multi-run statistical estimate or a fidelity gate.

#### External GuideLLM single-stream endpoint gate

GuideLLM 0.7.4 used its synchronous profile with one worker, maximum
concurrency one and exactly one synthetic request. The requested shape was
32,768 prompt tokens and 128 generated tokens; chat templating raised the
actual input to 32,820 tokens. The freshly started `qwen-q4-per-head` service
had session retention and durable caching disabled. Sampling came from the
artifact's thinking profile (`temperature=1.0`, `top_p=0.95`, `top_k=20`).

| GuideLLM metric | Result |
|---|---:|
| actual input | 32,820 tokens |
| reasoning / visible output | 128 / 0 tokens |
| first generated reasoning token | 39.861 s |
| request wall | 41.957 s |
| post-first generation window | 2.0963 s |
| post-first generation rate | 60.58 tok/s |
| mean inter-token latency | 16.506 ms |
| end-to-end output rate including prefill | 3.05 tok/s |
| target calls / verified positions | 43 / 128 |
| accepted drafts | 85 |
| verified positions/call | 2.9767 |

The worker again confirmed MTP-4, the 65,536-token draft vocabulary, Q8 MTP
K/V, GPU-only sampling and direct fused multi-query attention. Cleanup left
zero allocated/reserved pages and zero retained sessions. Raw results are
`out/benchmarks/guidellm-qwen-q4-per-head-32k-c1.json` and its CSV companion.

GuideLLM's `time_per_output_token_ms=327.8` is request wall divided by all 128
output tokens, so it includes the entire prefill and is not decode TPOT. The
post-first value above is independently derived from the recorded first/last
token timestamps: `127 / 2.0962884 = 60.58 tok/s`, matching the reported
16.506 ms inter-token latency. The random synthetic prompt consumed the output
budget entirely as hidden reasoning, so this is a serving-performance gate,
not a coherence, visible-answer or fidelity gate. One request is not a latency
distribution.

#### Real Pi long-context repetition failure and mitigation

A Qwen Abliterated Q4H Pi session supplied the missing real-project quality
evidence. Two fresh-prefill requests entered lexical reasoning loops; the user
cancelled both. A normal request completed between them, so this is not evidence
of a permanently corrupted retained session. No matched K1 or F16 replay was
run, and the result therefore cannot isolate the KV codec.

| Prompt | Generated before cancellation | TTFT | Wall | Result |
|---:|---:|---:|---:|---|
| 133,610 | 12,952 | 208.906 s | 449.750 s | repeated `(2026 real)` pattern |
| 148,441 | 79,608 | 240.218 s | 1,533.172 s | repeated multi-clause pattern |

Between those failures, a 133,617-token prompt generated 4,786 coherent tokens
and reached a tool call. Another coherent tool-producing turn in the session
used 23,599 output tokens, so a 16K ceiling would reject observed valid work.
The initial replacement policy declared `presence_penalty=1.5` and a 32,768
token thinking-output ceiling in each operational Qwen artifact. The
262,144-position context and non-thinking output capacity remain unchanged.
This is an intentional sampling change and a runaway bound; its real-project
fidelity gate is pending. Post-promotion `/model-info` reported the manifest as
the sampling source, the 32,768 thinking limit, Q4H, ABI 2, MTP-4, the
65,536-token draft head and Q8 MTP state for both artifacts. Non-thinking `hi`
smokes completed coherently at
52.99 tok/s post-first for Abliterated and 62.23 tok/s for official Qwen. These
short smokes qualify wiring only, not the pending long-context mitigation.

The corrected universal limit contract was then promoted on both Qwen Q4H
artifacts. The live Abliterated service reported `max_context=262144`, normal
`maximum_new_tokens=262143` and `maximum_thinking_tokens=32768`. A direct xhigh
request using Pi's actual `max_tokens=262143` shape completed with `OK`, 57
prompt tokens, 17 reasoning tokens and 22 total completion tokens, with
`finish_reason=stop`. The identical official-Qwen gate also returned `OK` and
`stop`, with 42 reasoning tokens and 47 total completion tokens. This closes
the prior HTTP 400 wiring regression; it does not replace the pending
long-project repetition gate.

The first user-observed Romanian answer under the 1.5 policy showed material
quality degradation: malformed words, inappropriate lexical substitutions and
broken agreement appeared throughout otherwise structured prose. Runtime
inspection confirmed that the artifact policy, rather than the parser, applies
the penalty once to every token already emitted by the response. The Qwen and
Qwen Abliterated policies were therefore corrected to `presence_penalty=0.5`
in both profiles while retaining the 32,768-token thinking ceiling. Ornith,
the third operational Q4H artifact, received the same 0.5 penalty while
retaining its existing sampling defaults and no Qwen-specific ceiling. Real
Romanian replays were still required; configuration validation alone was not
a quality pass.

#### Controlled Romanian long-generation fidelity matrix, 2026-09-22

A fixed 196-token Romanian prompt requested ten detailed AI-startup ideas.
Every controlled request used `reasoning_effort=xhigh`, `top_p=0.95`,
`top_k=20`, seed `314159`, a 12,000-token ceiling and the same native service
path. A new provider-neutral `speculative_decoding=false` request control used
the worker's existing scalar target step. Worker telemetry proved zero
`exact_decode_calls` for every non-speculative row.

| Weights | Target KV | Decode | Temp / presence | Wall | Completion / reasoning | Finish | Result |
|---|---|---|---|---:|---:|---|---|
| Abliterated FP4 | Q4H | MTP-4 | 1.0 / 0.5 | 240.967 s | 12,000 / 9,830 | length | stopped during idea 4; malformed Romanian |
| Abliterated FP4 | Q4H | scalar target | 1.0 / 0.5 | 391.897 s | 10,903 / 7,029 | stop | all ten ideas; malformed Romanian |
| Abliterated FP4 | F16 | scalar target | 1.0 / 0.5 | 459.430 s | 12,000 / 10,296 | length | stopped during idea 5; malformed Romanian |
| Official FP4 | F16 | scalar target | 1.0 / 0.5 | 319.344 s | 8,521 / 4,505 | stop | all ten ideas; severe malformed Romanian |
| Official FP4 | F16 | scalar target | 1.0 / 0.0 | 327.364 s | 8,730 / 4,127 | stop | all ten ideas; severe malformed Romanian |
| Official FP4 | F16 | scalar target | 0.6 / 0.0 | 225.539 s | 6,097 / 2,751 | stop | all ten ideas; shorter but still malformed Romanian |

Representative failures included `persoanelor vârstnici`, `sistemul triage`,
`Prevvedere`, `date longitudinale de vocă`, `frontierile`, `camiona`,
`script care iubește 1000 de dosare`, `gâdiluri uriașe`, `Rani un model`,
`părărire` and `școlile ... nu știe`. The issue therefore persists without
MTP, with exact F16 target KV, on official weights, with the official thinking
presence penalty of zero, and at a lower temperature. Q4H, MTP and sampling
change the sampled trajectory and task completion, but none is an isolated
root cause or a fidelity fix.

One matched-seed MTP/scalar pair is not a distribution-level MTP quality
measurement: exact speculative sampling may consume random values differently
while preserving the target distribution. It is valid evidence that disabling
MTP does not remove the lexical defect. No further sampling adjustment is an
admitted cause-isolation step. The next gate is an independent full-target
numerical oracle for the stored FP4 artifact, followed by a source-precision
behavior comparison if the artifact execution passes.

#### Selective MXFP6 embedding/head gate, 2026-09-22

The same Abliterated source revision was recompiled with only the token
embedding and vocabulary head in OCP MXFP6 E3M2 block-32. All other tensors,
Q4H target KV, MTP-4, sampling values and the fixed Romanian prompt remained
unchanged. This was a selective weight-fidelity experiment, not a claim that
the RTX 3090 executes native FP6.

| Gate | Result |
|---|---|
| compiler/container validation | 1,199 tensors; valid; source checkpoint SHA unchanged |
| MXFP6 records | exactly 2; 993,284,096 B each; ABI 6 |
| artifact delta | +635,699,200 B; 15,411,089,408 B candidate QPack |
| independent source oracle | relative L2 0.0563446; cosine 0.998523; max abs 0.00366211; 0 payload/scale mismatches |
| native CUDA oracle | embedding max abs 0; GEMV max abs 0 |
| Windows clean build | CTest 8/8; canonical Python suite 120/120 |
| live startup | passed on one RTX 3090; 5,877 MiB free after load |
| short `hi` smoke | coherent; 27 completion tokens; 1.211 s; 22.30 generated tok/s |
| Romanian long gate | 180 prompt + 12,000 completion tokens; 11,996 reasoning; `content=null`; `finish_reason=length`; 306.641 s |
| generated throughput | 39.13 tok/s versus 49.06 tok/s on the matched 244.578 s Q4H baseline; -20.24% |
| disposition | failed fidelity and speed; operational Q4H artifact restored and smoke-tested |

The hidden reasoning remained grammatically degraded (`senzoriilor`, `devieri
timpuri`, `linii lunar`, `întoarcii`) despite the better local reconstruction
of both I/O matrices. The result rejects those two tensors as the dominant
cause of the Romanian defect. It does not distinguish internal FP4 weight error
from a source-model behavior problem; that boundary still requires the
independent full-target and source-precision gates.

#### Source-BF16 activation-semantics gate, 2026-09-23

The candidate reused the validated 14,775,390,208-byte MSE-v2 QPack and changed
only artifact-declared activation semantics. Its CUDA path rounded the same
FP32 workspaces to BF16 at source-equivalent operation boundaries; Q4H, MTP-4,
sampling and the fixed Romanian request remained unchanged.

| Gate | Result |
|---|---|
| Windows clean Release/CUDA build | focused Qwen numeric gates passed |
| exact diagnostic request | 317-token prefill + stable-prefix checkpoint + first MTP step; Compute Sanitizer reported zero errors |
| asynchronous direct replay | passed with the same first two generated token IDs |
| real service telemetry | raw `provider_activation_bf16=1` |
| Romanian long gate | 317 prompt + 12,000 completion tokens; 8,619 reasoning; `finish_reason=length` |
| wall / generated throughput | 262.208 s / 45.78 tok/s |
| speed floor | 49.06 tok/s; failed by 6.69% |
| fidelity | malformed `familile`, `progreului`, `moduile`, `meslerii`, `nevoiea`, `alerte mai early`, `bilanț hídric` |
| disposition | failed fidelity and speed; no operational alias |

The first service start ended in `nvcuda64.dll` with status `0xc0000409`, but
the failure did not recur under the exact sanitizer, asynchronous direct or
real-service replays. It is not credited as a deterministic inference defect.
The request-level value `provider_activation_bf16=0` was also misleading: the
server subtracted a configuration gauge before/after the request, while raw
provider stats correctly reported one.

#### Hybrid block-32 Q8 activation gate, 2026-09-23

| Gate | Result |
|---|---|
| offline activation sample | fixed 317-token Romanian request, layer-0 normalized embeddings |
| hybrid Q8 reconstruction | 19.258x lower SSE; 317/317 rows non-regressed; 64.6155% local blocks; zero clipping |
| native CUDA gate | exact Q8 bytes/codes; GEMV, weight-reuse, GEMM and staged-GEMM checks passed after a clean build |
| live request | 317 prompt + 12,000 completion tokens; 11,554 reasoning; stopped after visible idea 1 |
| wall / generated throughput | 285.906 s / 41.97 tok/s |
| speed floor | 49.06 tok/s; failed by 14.45% |
| fidelity | malformed `diailizelor`, `clinicii pierd`, `introdu greutatea`, `Medicianul` |
| disposition | rejected; runtime implementation and service alias removed |

More accepted MTP drafts did not compensate for the per-block activation-scale
work. This is direct evidence that the much better local activation
reconstruction neither fixes the Romanian defect nor preserves the throughput
contract.

#### Exhaustive MSE-v2 scale audit, 2026-09-23

| Metric | MSE-v2 | Exhaustive legal UE8M0 search |
|---|---:|---:|
| FP4 matrices | 666 | 666 |
| deterministic sampled blocks | 42,624 | 42,624 |
| relative L2 | 0.1149280417 | 0.1149280417 |
| aggregate SSE | 4.1241743611 | 4.1241743611 |
| blocks selecting a different exponent | - | 0 |
| additional SSE reduction | - | 0.00% |

The predeclared admission threshold was at least 5% additional SSE reduction
with no block or role-group regression. It failed before any encoder change,
artifact build or live request.

#### Activation-aware complete-output scale gate, 2026-09-23

| Gate | Result |
|---|---|
| calibration coverage | 505/505 eligible dense projections |
| calibration residual SSE | -33.70% versus MSE-v2 |
| disjoint holdout residual SSE | -24.12%; 14/14 role groups improved |
| selected block scales | 105,250,169 / 813,957,120 covering reselections (12.93%) |
| QPack/runtime | ABI 3; 14,775,390,208 B; unchanged CUDA kernels |
| independent source qualification | valid; 269,056 sampled values; zero payload/scale mismatches; relative L2 0.1170506; cosine 0.9931263 |
| clean native gate | Windows Release/CUDA `--clean-first`; dense-FP4 CUDA smoke passed |
| live request | 317 prompt + 25,673 completion tokens; 17,850 reasoning; `finish_reason=stop` |
| prefill / TTFT | 1.000 s / 1.032 s; one prefill batch |
| wall / generated throughput | 500.235 s / 51.32 tok/s |
| MTP | depth 4; 15,151 accepted drafts; 10,523 exact calls |
| speed floor | 49.06 tok/s; passed by 4.61% |
| fidelity | failed: `muncă de liniă`, `modele de anomaliă`, `vau diferențiere`, `decizie de rețu`, `a pierdutului` |
| disposition | rejected as a Romanian-fidelity remedy; no operational promotion |

The artifact passed the no-runtime-cost premise and was faster than the fixed
floor on this completed trajectory, but the large offline residual improvement
did not restore natural Romanian. The candidate remains a recoverable research
artifact; the operational alias was never changed.

#### Activation-v3 Russian long-generation contrast, 2026-09-24

The physical `qwen3.8-27b-abliterated-fp4-activation-v3` artifact used above
was reopened directly on the RTX 3090. Its `COMPLETED` marker matched the
manifest file hash, and the QPack was 14,775,390,208 bytes. The service was
started cold with Q4H target KV, MTP-4, no retained session, and the same
`xhigh`, seed 314159, temperature 1.0, top-p 0.95, top-k 20, presence penalty
0.5, and 32,768-token completion ceiling. Only the 10-AI-startup-ideas prompt
was translated faithfully into Russian; the artifact was not promoted and
the operational alias was unchanged.

| Gate | Russian contrast |
|---|---:|
| prompt / completion / reasoning / visible tokens | 307 / 20,666 / 14,896 / 5,770 |
| finish reason / HTTP status | `stop` / 200 |
| provider / HTTP wall | 486.578 / 486.642 s |
| cold prefill / TTFT / prefill batches | 0.984 / 1.031 s / 1 |
| generated throughput | 42.47 tok/s; below the 49.06 tok/s floor on this trajectory |
| MTP accepted drafts / exact calls | 10,348 / 10,319 |
| visible quality | all 10 ideas completed; mostly fluent Russian, but `Последняя миля часто suffers от опозданий`, malformed technical word `металомах`, and `precast-заводы` |
| cleanup | service and runner exited; port 8080 closed; GPU returned to 218 MiB desktop residency |

The Russian result does not reproduce the Romanian error density, but it does
not establish clean multilingual fidelity either. This is one cold prompt and
one seed, not a source-BF16 contrast. The translated prompt changed token count
and generated trajectory; lower draft acceptance (10,348/10,319 versus
15,151/10,523 in the Romanian gate) accompanies the lower observed throughput
but does not prove a language-specific runtime regression. The raw request,
response, HTTP timing and service telemetry are retained under
`out/gates/qwen-abliterated-activation-v3-russian-*` and
`out/gates/qwen-activation-v3-russian-server.jsonl`.

#### Qwen3.8-27B configuration inventory, 2026-09-24

These are the distinct Qwen 27B mappings currently declared in
`ops/model-aliases.tsv`; alias synonyms share one row. This is a configuration
inventory, not a claim that every artifact is present on the host or that every
cross-product of settings has passed a live quality gate.

| Alias(es) | Weight artifact | Target KV at launch |
|---|---|---|
| `qwen`, `qwen3.8`, `qwen3.8-27b-fp4` | official FP4 | Q4H (`q4-f16-per-head`) |
| `qwen-f16` | official FP4 | exact F16 (`fp16`) |
| `qwen-abliterated`, `qwen3.8-27b-abliterated-fp4` | Abliterated FP4 | Q4H |
| `qwen-abliterated-k1` | Abliterated FP4 | lossy K1 (`fp4-e2m1-ue8m0-block32-key-outlier1`) |
| `qwen-abliterated-f16` | Abliterated FP4 | exact F16 |
| `qwen-abliterated-mse` | Abliterated MSE-v2 FP4 | Q4H |

The activation-v3 and activation-v4 FP4 artifacts are research candidates,
not operational aliases. The Russian contrast above verified activation-v3
physically and launched it directly; its JSON `model` field named the ordinary
Abliterated alias but did **not** change that alias. Activation-v4 passed a
separate Romanian speed gate at 50.48 generated tok/s and failed fidelity.

The last Russian request used the direct activation-v3 artifact, one RTX 3090,
cold startup, no retained session, Q4H, MTP depth 4, 307 prompt tokens,
`reasoning_effort=xhigh`, `enable_thinking=true`, `temperature=1.0`,
`top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.5`, seed `314159`,
`stream=false`, and `max_completion_tokens=32768`. The latter is a ceiling,
not a generation target. The effective policy also has frequency penalty 0
and repetition penalty 1. The exact payload is retained in
`out/gates/qwen-abliterated-activation-v3-russian-request.json`.

Request controls are separate from launch-time artifact/KV selection:
`speculative_decoding=true` uses available MTP by default, while `false`
forces scalar target decoding without changing weights or target KV;
`reasoning_effort` accepts `low`, `medium`, or `xhigh`; thinking-off uses
`chat_template_kwargs.enable_thinking=false`. The server accepts temperature
0--2, top-p (0,1], top-k 0--1,000,000, min-p 0--1, presence penalty -2--2,
and a non-negative signed-63-bit seed. Nonzero frequency penalty and
nonunit repetition penalty are not implemented. The current long-agent
policy defaults to temperature/top-p/top-k/min-p/presence of
`1.0/0.95/20/0/0.5` in thinking and `0.7/0.8/20/0/0.5` otherwise.

**Causal limit:** FP4 weights are not established as the cause of malformed
Romanian. Raw source-BF16 free generation was stopped after 528 reasoning
tokens and already contained malformed Romanian; it was not a full visible
answer. On an exact FP4-generated prefix, two malformed pieces had similar
native and source-BF16 probabilities; a third was excluded by BF16 top-p but
admitted by the native target. That separator has not been assigned to stored
FP4 weights versus provider arithmetic. A separately dequantized FP4 replay
used a reconstructed prefix that failed exact token parity, so it cannot close
this causal question. The earlier source-BF16 and exact-token gates above are
the evidence; no full source-BF16 visible-response fidelity pass exists.

#### English xhigh 12-profile speed matrix, 2026-09-24 (complete)

This new measurement does not rank candidates using any historical speed
number. It compares four physical Abliterated Qwen FP4 artifacts (standard,
MSE-v2, activation-v3, activation-v4) crossed with Q4H, K1 and F16 target KV.
All 12 profiles keep Q8 dense activations, MTP-4, one RTX 3090, a fresh service,
no retained session, `xhigh`, temperature 1.0, top-p 0.95, top-k 20, min-p 0,
presence penalty 0.5 and seed 314159. Each profile runs exactly one
English request using the same fixed ten-startup-ideas prompt in
`ops/benchmarks/qwen27b_english_startups_prompt.json`. The 32,768-token output
ceiling is a runaway guard; any length-terminated or empty-visible round is
excluded from ranking.

Each profile has an immutable run script under
`ops/windows/benchmarks/qwen27b/<profile>/Run.ps1`. The host-local launcher
continues without an SSH session. Each round saves the exact request, raw
response, and a summary matched to provider request telemetry; it also saves
`round.md`, and the orchestrator updates
`ledger.md` and `RESULTS.json` on the host. The result bundle is transferred
only after `COMPLETE.json` exists. Prefill is provider prefill wall time;
decode rate is `(completion_tokens - 1) / (provider_wall - TTFT)`; HTTP
end-to-end rate is `completion_tokens / HTTP_wall`. The primary winner is the
highest HTTP end-to-end rate among the twelve normally stopped requests;
the decode-rate winner is reported separately. Startup time is recorded
separately and is not included in request throughput.

An earlier launcher erroneously repeated the same request ten times per
profile. It was stopped during `fp4_q4h` after four completed requests; its
partial files are archived on the 3090 under
`out/benchmarks/qwen27b-english-speed-invalid-repeats-20260924`. Those files
are excluded from this matrix and from ranking. The corrected matrix uses one
request per profile and starts in a fresh result directory.

The first corrected `fp4_q4h` request completed and its service cleaned up,
but the orchestrator initially stopped before recording the round as complete:
Windows PowerShell rejected `File.Replace` with a null backup path while
updating existing `RESULTS.json`. An explicit backup path passed a two-update
Windows gate. The orchestrator then resumed from the validated first-round
summary and cleanup record; it did not regenerate that response. This was an
orchestration failure, not an inference failure.

The final `COMPLETE.json` reports 12 completed and zero failed rounds. Its
scheduled task exited with result 0; port 18080 was free, and GPU memory use
after cleanup was 309 MiB. The full 114-file result bundle was copied from the
3090 to `out/benchmarks/qwen27b-english-speed`. Each request JSON has the same
canonical SHA-256 (`471619a2f6bcf33e45462a2b3cf50e3ac45daf1daebc28eacaeec74f23c4a281`),
all responses have `finish_reason=stop`, and each contains ten numbered startup
ideas. The prompt had 228 populated tokens in every round.

| Rank | Profile | Generated / reasoning tokens | Prefill s / tok/s | TTFT s | Decode tok/s | HTTP end-to-end tok/s | HTTP wall s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `fp4_q4h` | 20603 / 15160 | 0.531 / 429.38 | 0.578 | 59.55 | 59.44 | 346.61 |
| 2 | `msev2_q4h` | 18851 / 10657 | 0.578 / 394.46 | 0.609 | 57.48 | 57.37 | 328.57 |
| 3 | `v4_q4h` | 17007 / 11721 | 0.563 / 404.97 | 0.609 | 56.54 | 56.41 | 301.47 |
| 4 | `v3_q4h` | 18026 / 9887 | 0.547 / 416.82 | 0.594 | 56.18 | 56.07 | 321.51 |
| 5 | `msev2_k1` | 16743 / 9953 | 0.563 / 404.97 | 0.609 | 51.56 | 51.46 | 325.39 |
| 6 | `fp4_k1` | 18399 / 13143 | 0.531 / 429.38 | 0.578 | 50.81 | 50.72 | 362.73 |
| 7 | `v3_k1` | 21950 / 13818 | 0.547 / 416.82 | 0.594 | 49.88 | 49.80 | 440.75 |
| 8 | `v4_k1` | 20697 / 12285 | 0.562 / 405.69 | 0.609 | 49.06 | 48.98 | 422.59 |
| 9 | `v4_f16` | 23477 / 14893 | 0.594 / 383.84 | 0.656 | 46.58 | 46.51 | 504.76 |
| 10 | `msev2_f16` | 20786 / 14176 | 0.594 / 383.84 | 0.640 | 46.24 | 46.17 | 450.22 |
| 11 | `v3_f16` | 15419 / 10015 | 0.593 / 384.49 | 0.640 | 44.83 | 44.74 | 344.62 |
| 12 | `fp4_f16` | 19305 / 13013 | 0.625 / 364.80 | 0.687 | 44.15 | 44.07 | 438.04 |

The speed-only winner on this **single 228-token prompt** is the standard FP4
artifact with Q4H target KV: 59.55 generated decode tokens/s and 59.44
generated HTTP end-to-end tokens/s. These rates include reasoning tokens; its
visible output was 5,443 tokens, or 15.70 visible tokens/HTTP second. The
responses have different generated and reasoning lengths, so this one-shot
ordering is not a statistical claim about all prompts, long-context prefill,
fidelity, or real coding-harness throughput. The highest visible-output rate
on this request was `v3_q4h` at 25.31 visible tokens/HTTP second, reflecting
different reasoning/output allocation rather than higher generated-token
decode speed. No service alias or model default was changed by this benchmark.

#### Activation-aware adjacent-code gate, 2026-09-23

| Gate | Result |
|---|---|
| target coverage | 497/497 regular VM projections; eight exact-decode-only matrices excluded |
| sampled calibration residual SSE | -55.24% versus activation-v3 |
| disjoint holdout residual SSE | -41.93%; 13/13 role groups non-regressed |
| bounded ordering | top-64 exactly matched unbounded sampled result; 23,590 / 27,172,864 changes |
| complete artifact payload changes | 27,945,498 adjacent nibble reselections; scales fixed to activation-v3 result |
| QPack/runtime | ABI 3; 14,775,390,208 B; unchanged CUDA kernels and MTP-only payload |
| independent source qualification | valid; 269,056 sampled values; zero payload/scale mismatches; FP4 relative L2 0.1181699; cosine 0.9929944 |
| live request | 317 prompt + 19,541 completion tokens; 13,612 reasoning; `finish_reason=stop` |
| prefill / TTFT | 1.125 s / 1.157 s; one prefill batch |
| wall / generated throughput | 387.125 s / 50.48 tok/s |
| MTP | depth 4; 11,335 accepted drafts; 8,207 exact calls |
| speed floor | 49.06 tok/s; passed by 2.89% |
| fidelity | failed: `angajatori care recrutați`, `pașii greșiit`, `model de limba artificială`, `stabilizaască`, `de la o anuire`, `partenери` |
| disposition | rejected as a Romanian-fidelity remedy; no operational promotion |

The candidate proved that representable adjacent-code changes can substantially
reduce the captured complete-output residual while preserving live throughput.
That improvement still did not restore natural Romanian, so it is not an
accepted quality fix and remains outside the operational alias.

#### Raw source-BF16 Romanian separation gate, 2026-09-23

The immutable `huihui-ai/Huihui-Qwen3.8-27B-abliterated` snapshot at revision
`739e3c5b89849f6c238ce1e5b70008612ae42cdd` was loaded directly with
Transformers in `torch.bfloat16`; no QPack or quantum-llm projection kernel was
used. The request retained the fixed 317-token Romanian prompt, `xhigh`
thinking, seed 314159, temperature 1.0, top-p 0.95, top-k 20 and
`presence_penalty=0.5`.

| Gate | Result |
|---|---|
| free source generation | 528 reasoning tokens checkpointed; reinspection found `experiența utilizator` and `irigare/boale` in raw BF16 thinking |
| source contrast | 14,087-token text prefix reconstructed through the first activation-v4 visible defect; original service token IDs were not retained |
| malformed candidate | ` durată` in `prețurile de spitalizare și durată șederii` |
| corrected candidate | ` durata` |
| BF16 log-probability delta | corrected minus malformed = `+3.533903` |
| BF16 probability ratio | corrected / malformed = `34.2574x` |
| matched BF16 first-token top-20, presence 0.5 | corrected token rank 6, malformed first token rank 14; adjusted logit lead `+1.375` for corrected |
| cleanup | reference process stopped; GPU returned to 323 MiB desktop use |

The short free generation is not a visible-response fidelity pass, and the
earlier "no malformed word" description was incorrect. Although BF16 preferred
the corrected continuation, the malformed first token remained eligible under
the fixed top-20 sampler. This contrast alone cannot attribute the emitted
mistake to the FP4 representation or native runtime. The CPU-offloaded reference
run is not a service-throughput measurement.

#### Independent FP4 weight versus BF16 Romanian-token gate, 2026-09-24

The saved activation-v4 response was reencoded with the source tokenizer to
select 19 token pieces across seven malformed expressions. Source BF16 and the
same upstream Transformers implementation with all 546 text-path FP4 QPack
tensors dequantized into BF16 storage scored the same 16,820-token reconstructed
prefix. The native quantum-llm provider was not used in either pass. Cached
causal scoring verified every cache length; logits were finite. Rankings below
include the request's 0.5 presence penalty, before the separate top-p filter.

| Gate | BF16 source | Independent dequantized FP4 |
|---|---:|---:|
| token pieces inside `top_k=20` | 18 / 19 | 19 / 19 |
| final `it` in `greșiit` | rank 23; -0.3125 logit to cutoff | rank 13; +0.625 logit to cutoff |
| reference cleanup | not applicable | 207 MiB GPU desktop residency |

This demonstrates that stored FP4 weights can cross a malformed token's top-k
boundary without the native provider. It does not establish top-p eligibility,
the probability of the full malformed phrase, or numerical parity with the
service: original service token IDs were not captured and the prefix was
reconstructed from text. The other 18 pieces were already inside BF16 top-20.
No operational artifact or setting changed, and no throughput gate was run.

#### Full-sampler reconstructed-prefix prerequisite, 2026-09-24

The independent FP4 replay targeted the final `it` in `greșiit` with all 546
text-path FP4 tensors dequantized in upstream Transformers. It kept the 0.5
presence penalty, temperature 1.0, top-k 20 and top-p 0.95. No native service
or candidate artifact was changed.

| Gate | Result |
|---|---:|
| reconstructed prefix before target | 15,590 tokens |
| malformed `it` after presence penalty | rank 12; 0.875 logit above top-20 cutoff |
| top-p nucleus | 2 tokens; malformed `it` excluded |
| plausible reencoded full completions | 19,529--19,531 tokens |
| service-reported actual completion | 19,541 tokens |
| planned BF16-head/layer substitutions | stopped before execution |
| cleanup | 214 MiB GPU desktop residency |

The earlier rank-13 result used different causal chunks; both runs agree on
top-k inclusion, but top-p exclusion is the decisive new gate. The 10-token
deficit for the previously used separator proves the saved parsed text is not
an exact token-stream replay. Neither the responsible tensor group nor the
native sampler/provider can be identified from this prefix. This is not a
fidelity fix or a throughput measurement.

#### Recurrent A-projection selective-MXFP6 prerequisite, 2026-09-23

The first candidate covered both recurrent scalar-control projections across
48 layers each. It added 5,898,240 bytes but missed the fixed disjoint-holdout
improvement gate (47.8270% versus the required 50%), so it was not built.
Calibration-only selection retained A and rejected B before a third holdout.

| Gate | A-only result |
|---|---:|
| selected matrices | 48 |
| third holdout prompt | 267 tokens; distinct school-energy domain |
| FP4 residual SSE | 11,061.264079 |
| MXFP6 residual SSE | 4,110.853400 |
| improvement | 62.8356% |
| layer regressions | 0/48 |
| added raw bytes / target call | 2,949,120 B |
| declared maximum | 5 MiB and about 394.38 MB throughput-equivalent |
| candidate QPack delta / MXFP6 records | +2,949,120 B / 48 |
| independent source qualification | valid; zero FP4/MXFP6 payload or scale mismatches; MXFP6 cosine 0.9985397, relative L2 0.0541595 |
| live request | 317 prompt + 22,308 completion tokens; 15,825 reasoning; 6,483 visible; `finish_reason=stop` |
| wall / generated throughput | 450.961 s / 49.4677 tok/s |
| speed floor | 49.06 tok/s; passed by 0.83% |
| fidelity | failed: `gerosiatrii`, `oprrire`, `coleriele`, `calificăți`, `demostra`, `consilienți`, `o istoric`, `incompletes`, `reducererea` |
| disposition | rejected as a Romanian-fidelity remedy; no stable-artifact promotion |

The comparison used actual activation-v4 FP4 QPack rows, an independent MXFP6
decoder and source-BF16 weights. Runtime Q8 activation rounding was emulated
exactly. The prerequisite admitted a candidate, but the completed live gate
showed that its large local residual improvement did not restore Romanian.

#### BF16 dense-activation-input gate, 2026-09-23

This artifact-declared experiment retained the activation-v4 FP4 weights and
replaced only the dense projection operand quantization from row-global Q8 to
BF16. The first narrow-only microbenchmark was invalidated by the real service
path. A small-batch BF16 kernel and a wide down-projection measurement were then
added before the single allowed retest.

| Gate | Corrected result |
|---|---:|
| canonical clean build | CTest 8/8; Python contract 120/120 |
| CUDA maximum absolute error | 0.0000534058 |
| batch-five combined Q8 / BF16-input | 76.3036 / 108.2550 GB/s |
| BF16-input minimum | 74.1572 GB/s |
| live request deadline | 675.008 s; no completed response |
| absolute throughput upper bound | 48.5446 tok/s (`32768 / 675.008`) |
| required throughput | at least 49.06 tok/s |
| fidelity | not scored because the request was cancelled |
| disposition | rejected for speed; stable artifact untouched |

The native result proves the corrected kernel in isolation, but not an
end-to-end speed pass. The real request falsified the prerequisite once service
overheads and all dense operations were included.

#### Exact recurrent-convolution gate, 2026-09-24

This candidate replaced all 48 four-tap recurrent convolution tensors with
exact source F32 records. The recurrent CUDA kernel already read an F32 buffer,
so arithmetic and per-call device bytes were unchanged; only offline storage
and startup dequantization changed.

| Gate | Result |
|---|---:|
| source FP4 convolution relative L2 | 0.1119022 aggregate; 0.1058189--0.1148664 by layer |
| candidate source qualification | 48 exact F32 convolutions; zero F32 mismatches; zero remaining FP4 payload/scale mismatches |
| candidate QPack | 14,774,996,992 B; 393,216 B smaller than activation-v4 |
| canonical clean build | CTest 8/8; Python contract 120/120 |
| direct service smoke | coherent `Hi! How can I help you today?`; normal stop |
| live request | 317 prompt + 19,471 completion tokens; 11,153 reasoning; 8,318 visible; `finish_reason=stop` |
| wall / generated throughput | 391.799 s / 49.6964 tok/s |
| speed floor | 49.06 tok/s; passed by 1.30% |
| fidelity | failed: `cardiice`, `utilizatori directs`, `datele greu de replicabile`, `o操are de utilaj`, `Fermere`, `statiche`, `experiența utilizatorul` |
| cleanup | candidate stopped; temporary alias removed; stable artifact untouched |
| disposition | rejected; experimental ABI and default adapter change rolled back |

This is a stronger negative control than another local error reduction: it
removed the entire source error of the selected component and still left the
same defect class. Recurrent convolution quantization is not the dominant cause
of the Romanian corruption.

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

#### RTX 3090 cold Pi layer-major gate, 2026-09-21

This gate used `./ops/pi.sh qwen-flash --no-session -p hi`, the real project
context, default xhigh reasoning, exact F16 KV, the 12 GiB common routed-VRAM
ceiling and one RTX 3090. Before the measured request the service loaded zero
persistent sessions and reported zero session-cache bytes. Model startup is not
included. The server snapshots worker counters immediately after `BEGIN`, so
prefill traffic below excludes all decode tokens.

| Metric | Historical baseline | Final gate |
|---|---:|---:|
| populated prompt | 4,190 | 4,190 |
| provider prefill batches | 10 | 1 |
| program-sequence tiles | not recorded | 1 |
| workspace rows | 512 | 1,024 |
| SSD -> RAM | 158.8 GB | 62,030,667,776 B (62.03 GB) |
| storage wait | 99.7 s | 34.730 s |
| RAM -> GPU | 236.8 GB | 61,982,054,400 B (61.98 GB) |
| upload wait | 92.3 s | 24.617 s |
| prefill wall | not separated | 163.125 s |
| TTFT | 325.3 s | 163.640 s |

TTFT improved by about 49.7%, and both prefill traffic counters passed the
80 GB gate. The runtime selected a 12,288-row sequence tile, so the complete
4,190-row hidden/Hyper stream stayed in one tile. CUDA free memory was
18,724,421,632 bytes before fixed workspace allocation and 14,012,121,088
bytes afterwards, above the mandatory 1 GiB reserve.

The grouped standard-FP4 path processed 1,999,010 primary selections in
533,944 bounded work items across 240 routed-prefill calls. Its independent
native oracle reported maximum absolute error 0 and cosine 1.0. This combined
end-to-end gate does not isolate how much TTFT reduction came from grouping.
It does prove the exact standard-FP4 service path; native NVFP4 was neither
selected nor credited.

The same request generated 331 tokens and finished in 264.406 s. Its complete
request counters reached 77,046,390,784 read bytes and 113,401,804,800 uploaded
bytes. Those totals include decode and are intentionally not compared with the
prefill-only 80 GB gate.

Common-path regressions on the same clean binary passed through
`./ops/model.sh chat`: Qwen returned coherent text at 13 prompt / 11 output
tokens, 1.700 s first-visible and 1.895 s wall; DeepSeek returned coherent text
at 5 prompt / 10 output tokens, 3.350 s first-visible and 9.688 s wall. The
DeepSeek start also exposed and fixed a strict-PowerShell validation bug for an
absent optional artifact sampling policy; Qwen, Qwen Flash and DeepSeek
contract extraction then passed.

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
