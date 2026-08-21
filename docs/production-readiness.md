# Production readiness

Status: rejected for production and Claude Code use as of 2026-08-21.

## Verdict

The official `Qwen/Qwen3.8-27B` checkpoint was compiled and transactionally
published as a custom FP4 artifact, but the resulting service is not
production-ready. The earlier short `hi` and tool-call gates proved protocol
functionality only. Real Claude Code requests failed cold latency, factual
quality and bounded agent behavior, while maximum-context decode remained far
below the requested rate.

An experimental FP8-KV provider later made the complete 262,144-token state fit
on the RTX 3090 and reduced cold prefill substantially, but decoded only 3.064
tok/s at the saturated context. It also failed a direct factual prompt. FP8 was
not promoted. The attempted F16 comparison bypassed the chat service and used
greedy fixed-length diagnostic generation without EOS handling, so it cannot
be used to reject the FP4 weights or runtime. The corrected real F16 service
path subsequently passed both `hi` and the four-stanza prompt through
`model.sh chat`; this removes the earlier weight/runtime rejection inference,
but does not qualify saturated-context throughput or reference quality.

## Published artifact

| Property | Verified value |
|---|---|
| source | `Qwen/Qwen3.8-27B` |
| immutable source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| location | `${MODEL_ROOT}/qwen3.8-27b-fp4` |
| container | one `dense.qpack`, 14,775,390,208 bytes, 1,199 records |
| matrix encoding | FP4 E2M1 with UE8M0 block-32 scales, quant ABI 3 |
| other dense records | raw ABI 0 where declared by the artifact |
| routers/experts | none; Qwen3.8-27B is dense |
| norms/metadata precision | float32 as declared in the manifest |
| runtime program | schema 3, 113,369 bytes |
| runtime program SHA-256 | `d8c331156f4634c5ce7249a0652bc05d1d1fb2fba184865272e4e9f226283e0d` |

The FP4 payload was not rewritten during the response/sampling work; only the
validated runtime program metadata was refreshed and published
transactionally.

## Accepted evidence

| Gate | Evidence | Verdict |
|---|---|---|
| build/tests | Release Windows/CUDA build, 5/5 CTest and 74 Python tests | pass |
| server dependencies | exact `transformers==5.15.0`, `Jinja2==3.1.6` checked before service replacement | pass |
| normal chat | 53 prompt, 44 generated, 2.047 s wall, 0.703 s server TTFT | pass |
| normal decode rate | about 21.5 tok/s end-to-end and 32.0 tok/s after first generated token | short-context pass |
| function call | official parser returned `get_weather({"city":"Chisinau"})` | pass |
| tool continuation | resumed request consumed the tool result and returned a coherent final answer | pass |
| maximum capacity | 262,016 prompt + 128 generated = 262,144 tokens | pass |
| maximum-context latency | TTFT 5,423.422 s; total 5,462.672 s | too slow |
| maximum-context decode | about 3.24 tok/s after first generated token | target not met |
| FP8-KV maximum capacity | 262,016 prompt + 128 generated; 1,024/1,024 pages | capacity pass only |
| FP8-KV maximum-context latency | 1,168.594 s prefill; 41.774 s for 128 decode calls | too slow |
| FP8-KV maximum-context decode | 3.064 tok/s | target not met |
| Claude Code cold `hi` | 29,487 prompt; 82.875 s TTFT; 88.281 s wall | fail |
| Claude Code factual prompt | fabricated Maia Sandu biography after 94.157 s | fail |
| direct FP8 factual prompt | no visible answer after 320 reasoning tokens; factual confusion | fail |
| direct F16 diagnostic | fixed 768 greedy tokens outside the chat service and without EOS | inconclusive |
| real F16 `model.sh chat` `hi` | 53 prompt; 30 generated; 2.125 s server TTFT; 3.079 s wall; 30.40 tok/s after first generated token | pass |
| real F16 `model.sh chat` poem | 68 prompt; 476 generated; 2.812 s server TTFT; 19.047 s wall; 29.26 tok/s after first generated token; four stanzas and no fabricated biography | short-context pass |
| common-runner Qwen F16 regression | `hi`; 53 prompt; 30 generated; 1.703 s server TTFT; 2.672 s wall; 29.93 tok/s after first generated token | pass |
| common-runner DeepSeek regression | `hi`; 5 prompt; 10 generated; 3.579 s server TTFT; 10.735 s wall; honest `session_retention=false` | pass |

The maximum-context gate proves real allocation, prefill and decode, not merely
metadata admission. It does not yet qualify long-document retrieval quality,
tool reliability across a corpus, or concurrent capacity.

## Readiness matrix

| Area | State | Remaining gap |
|---|---|---|
| immutable FP4 artifact | ready | retain source/manifest hashes and rollback artifact |
| real service lifecycle | functional | final stop/process cleanup is required after each gate |
| official chat/reasoning template | functional | artifact must continue declaring the grammar |
| stochastic sampling | functional | distribution-quality comparison against official reference remains |
| function tools | functional pilot | no constrained `required` or named-tool decoding |
| 262,144-token capacity | functional, capacity one | semantic long-context suite and failure/recovery gates remain |
| short-context throughput | direct API can decode quickly | real Claude Code cold TTFT is about 83 seconds |
| maximum-context throughput | not ready | 3.24 tok/s versus >=30 tok/s target |
| FP4 numerical quality | basic F16 chat passes; reference fidelity unresolved | compare logits/output quality with the immutable official reference before a production claim |
| DeepSeek compatibility | real common-runner `hi` passes | callable provider does not yet implement retained-session checkpoint/rewind |
| observability | pilot | server telemetry is authoritative; client visible-TTFT rate is misleading with hidden reasoning |
| reliability/security | not ready | soak, fault injection, edge TLS/auth/rate limits and alerting remain |

The final regression cleanup stopped process IDs 25588, 24060 and 21764. The
task returned to `Ready`, the endpoint was closed, the inventory reported
25,089,277,952 of 25,769,803,776 VRAM bytes free, and no Python or Expert VM
process remained in the CUDA process query.

## Release blockers

1. Compare the real F16 service path against the immutable official reference
   on a bounded numerical and behavioral corpus. Do not reuse the direct
   fixed-length diagnostic as chat evidence.
2. Raise full-context decode from about 3.24 tok/s to the declared SLO without
   reducing context, changing model math, or hiding reasoning tokens.
3. Reduce the 262,016-token prefill wall time while preserving exact causal
   results.
4. Compare FP4 output quality and sampling behavior with the immutable official
   checkpoint on a bounded reasoning, tool and long-context corpus.
5. Pass long-context cancellation, timeout, OOM, restart and repeated-session
   tests without leaked GPU processes or corrupted retained state.
6. Qualify the actual harness request shape, including large histories and
   tool-result continuations, rather than only synthetic token capacity.
7. Put the loopback service behind production supervision, logs/alerts and a
   private authenticated TLS edge.

## Claims policy

- `max_context=262144` is now a demonstrated capacity claim, not a 30 tok/s or
  semantic-quality claim.
- The valid short-run post-first rate is calculated from the first generated
  token in server telemetry. The old `135.91 tok/s` client value is invalid
  because it mixed hidden-token counts with first-visible-content timing.
- A coherent answer is a functional gate, not an FP4 quality evaluation.
- A kernel microbenchmark is not end-to-end service performance.
- No INT8 baseline or mixed-format label may be presented as this FP4 result.
