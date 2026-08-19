# Production readiness

Status: Qwen3.8-27B FP4 production-pilot verdict as of 2026-08-19.

## Verdict

The official `Qwen/Qwen3.8-27B` checkpoint is compiled, transactionally
published and served through the common runtime on the RTX 3090 host. Normal
chat, official reasoning formatting, sampling, a real function call, tool
continuation, retained sessions and the fully populated 262,144-token capacity
path work.

It is not yet production-ready for the requested performance SLO. Short chat
decode reaches approximately 32 tok/s after the first generated token, but the
same rate at maximum populated context is approximately 3.24 tok/s. The
capacity objective is proven; the `>=30 tok/s` maximum-context objective is not.

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
| build/tests | Release Windows/CUDA build, 5/5 CTest; local compiler and 33 server tests | pass |
| server dependencies | exact `transformers==5.15.0`, `Jinja2==3.1.6` checked before service replacement | pass |
| normal chat | 53 prompt, 44 generated, 2.047 s wall, 0.703 s server TTFT | pass |
| normal decode rate | about 21.5 tok/s end-to-end and 32.0 tok/s after first generated token | short-context pass |
| function call | official parser returned `get_weather({"city":"Chisinau"})` | pass |
| tool continuation | resumed request consumed the tool result and returned a coherent final answer | pass |
| maximum capacity | 262,016 prompt + 128 generated = 262,144 tokens | pass |
| maximum-context latency | TTFT 5,423.422 s; total 5,462.672 s | too slow |
| maximum-context decode | about 3.24 tok/s after first generated token | target not met |

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
| short-context throughput | target reached for accepted `hi` run | representative harness workload still required |
| maximum-context throughput | not ready | 3.24 tok/s versus >=30 tok/s target |
| FP4 numerical quality | partially qualified | native kernel checks and coherent output exist; reference-model quality suite remains |
| DeepSeek compatibility | preserved by common contracts/tests | latest response-protocol changes have not been requalified as a new DeepSeek release |
| observability | pilot | server telemetry is authoritative; client visible-TTFT rate is misleading with hidden reasoning |
| reliability/security | not ready | soak, fault injection, edge TLS/auth/rate limits and alerting remain |

## Release blockers

1. Raise full-context decode from about 3.24 tok/s to the declared SLO without
   reducing context, changing model math, or hiding reasoning tokens.
2. Reduce the 262,016-token prefill wall time while preserving exact causal
   results.
3. Compare FP4 output quality and sampling behavior with the immutable official
   checkpoint on a bounded reasoning, tool and long-context corpus.
4. Pass long-context cancellation, timeout, OOM, restart and repeated-session
   tests without leaked GPU processes or corrupted retained state.
5. Qualify the actual harness request shape, including large histories and
   tool-result continuations, rather than only synthetic token capacity.
6. Re-run DeepSeek's real service gate before publishing a new universal-path
   compatibility claim.
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
