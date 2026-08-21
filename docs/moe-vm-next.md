# Current state and next work

Status: canonical implementation handoff, revised 2026-08-21 after the
Qwen3.8 real F16 service correction.

Read this document before resuming. Historical campaigns in `docs/` remain
evidence, not active backlogs.

## Active objective and invariants

The experimental FP8-KV service failed a direct factual-quality gate and the
Claude Code deployment failed latency and agent behavior. The original F16
comparison was invalid because it bypassed chat, sampling and EOS. The
corrected `MODEL_KV_CACHE_DTYPE=fp16` service passed both `hi` and the original
four-stanza prompt through `model.sh chat`; therefore the prior claim that the
FP4 weights/runtime were rejected was unsupported. Preserve DeepSeek
compatibility while qualifying the remaining maximum-context objective.

The acceptance criteria are fixed:

- context is actually tokenized, allocated, prefilled and consumed; metadata
  admission or a sparse placeholder is not completion;
- Qwen3.8 matrices execute through the declared FP4 ABI on SM86;
- no reduced context, reduced reasoning effort, altered model math or quality
  shortcut may be presented as the target result;
- default reasoning effort is `xhigh`;
- common runtime/API/service code remains artifact- and capability-driven;
- DeepSeek may keep an artifact adapter and genuinely different numeric
  providers, but receives no second launcher/API stack;
- benchmarks exist only to answer an acceptance or bottleneck question.

Qwen3.8-27B is dense, so expert demand paging is not involved in its current
hot path. The generic program/provider infrastructure remains shared with the
sparse DeepSeek path.

## Current artifact and deployment

The selected artifact is `${MODEL_ROOT}/qwen3.8-27b-fp4`, sourced from
`Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Its manifest declares:

- 64 text layers: 16 full-attention and 48 linear-attention layers;
- maximum position count 262,144;
- one 14,775,390,208-byte dense pack with 1,199 records;
- FP4 E2M1 matrix payloads with UE8M0 block-32 scales (ABI 3), plus explicitly
  declared raw ABI-0 dense records;
- no routers or experts;
- schema-3 ordered runtime program, SHA-256
  `d8c331156f4634c5ce7249a0652bc05d1d1fb2fba184865272e4e9f226283e0d`.

The Windows project, build and published artifacts are on `D:/quantum-llm`;
the independent Hugging Face cache remains on C:. `MODEL_ROOT` is the only
host-dependent model-storage setting. `qwen` resolves to the Qwen3.8 FP4
artifact through `ops/model-aliases.tsv`.

Reference service configuration:

```text
MODEL_MAX_CONTEXT=262144
MODEL_MAX_OUTPUT_TOKENS=8192
MODEL_MAX_BODY_MIB=16
MODEL_KV_CACHE_MIB=5120
MODEL_VRAM_CACHE_GIB=12
MODEL_WORKER_CAPACITY=1
MODEL_GENERATION_TIMEOUT_SECONDS=14400
```

This remains the rollback configuration. The experimental
`fp8-e4m3-per-head` / 8,528 MiB KV profile was not promoted to `.env`.
The corrected F16 run was also selected transiently and was not promoted.

## 2026-08-21 qualification and corrected F16 evidence

- Full physical FP8-KV context: 262,016 prompt + 128 generated, 1,024 pages,
  1,168.594 s prefill and 3.064 tok/s decode.
- Claude Code cold `hi`: 29,487 prompt tokens, 82.875 s TTFT and 88.281 s
  wall time.
- Claude Code four-stanza Maia Sandu request: 29,660 prompt tokens, 83.454 s
  TTFT, 94.157 s wall time, and a fabricated biography.
- Direct 100-token FP8 prompt: no visible answer in 320 reasoning tokens and
  incorrect biographical claims. The 768-token F16 diagnostic is inconclusive
  because it bypassed the real chat service, sampling and EOS handling.
- Real F16 `model.sh chat` `hi`: 53 prompt tokens, 30 generated tokens,
  2.125 s server TTFT, 3.079 s wall and 30.40 tok/s after the first generated
  token.
- Real F16 `model.sh chat` four-stanza prompt: 68 prompt tokens, 476 generated
  tokens, 2.812 s server TTFT, 19.047 s wall and 29.26 tok/s after the first
  generated token. It returned four stanzas without the fabricated
  mathematician biography.
- The terminal client's initial 85.84/60.44 post-first figures are invalid:
  they combined total output (including hidden reasoning) with the timestamp
  of the first visible content. Server `request_telemetry` is authoritative.
- A cancelled resumed Claude request dropped the retained session and forced
  the next 29,660-token cold prefill. This is a real retention bug, but fixing
  it cannot qualify the failed model output.
- The common-runner retention contract is now capability-driven. Dense FP4
  implements exact checkpoint/rewind and advertises retention; the callable
  DeepSeek provider does not and advertises `session_retention=false`. The
  server no longer sends or executes checkpoints for non-retaining providers.
- The post-change Qwen F16 `hi` regression recorded 53 prompt tokens, 30
  generated tokens, 1.703 s TTFT, 2.672 s wall and 29.93 tok/s after the first
  generated token. The DeepSeek `hi` regression recorded 5 prompt tokens, 10
  generated tokens, 3.579 s TTFT and 10.735 s wall. Both used the common
  artifact VM runner with a 12 GiB VRAM-cache budget.

The RTX 3090 service was stopped after the final common-runner regressions. The
latest inventory reported 25,089,277,952 of 25,769,803,776 VRAM bytes free;
the CUDA process query contained no Python or Expert VM process.

## Implemented in the current cycle

- schema-3 program execution for Qwen3.8's declared full-attention,
  linear-attention, MLP, norm, embedding and head operations;
- paged KV allocation and bounded causal prefill up to the advertised context;
- SM86 FP4 matrix execution and native long-context attention qualification;
- generic provider token selection with temperature, top-k, top-p, min-p and
  deterministic request seeding;
- artifact generation-config defaults instead of a hardcoded greedy client;
- official tokenizer chat template with `xhigh` reasoning by default;
- standard Transformers response parsing for reasoning and XML function calls;
- OpenAI Chat/Responses tool-call output and tool-result continuation;
- exact pinned Python dependency verification before replacing a running task;
- configurable request body ceiling for large serialized histories;
- transactional metadata refresh/publication of the existing FP4 artifact.

The common server does not inspect a Qwen model ID to enable these features.
It consumes tokenizer grammar, artifact program and provider capabilities.

## Proven service results

### Normal chat

The real `./ops/model.sh chat qwen` prompt `hi` returned a coherent answer.
Server telemetry recorded 53 prompt tokens, 44 generated tokens, 0.703 s TTFT
and 2.047 s total: about 21.5 tok/s including TTFT and 32.0 tok/s after the
first generated token.

### Tool round trip

A real Chat Completions request produced
`get_weather({"city":"Chisinau"})`. A resumed request with the tool result
returned the final answer. Hidden reasoning was separated from visible text and
counted in usage.

### Maximum populated context

A real request recorded:

```text
prefill_tokens=262016
generated_tokens=128
ttft_seconds=5423.422
wall_seconds=5462.672
```

This fills exactly 262,144 positions and proves the full service capacity path.
After the first generated token, the remaining 127 tokens took 39.25 seconds,
or approximately 3.24 tok/s. Therefore the maximum-context speed target is not
met.

For this request, accumulated provider telemetry attributed about 3,973.5 s
to full attention, 637.8 s to recurrent/linear attention, 582.1 s to FFN and
259.6 s to MTP. Full attention is the first measured bottleneck; do not start a
different optimization direction without disproving that attribution.

The follow-on architecture calculation is recorded in
[Heterogeneous organ placement](heterogeneous-placement-research.md), with the
reproducible manifest inventory and bandwidth arithmetic in
[Heterogeneous placement benchmark](heterogeneous-placement-benchmark.md).
It proposes algebraic GPU/CPU sharding with compute colocated with its weights
or state; it is research, not implemented functionality. In particular, the
calculation rejects copying whole active KV from RAM/NVMe to GPU and gives CPU
KV execution only a qualification gate, not an assumed speedup.

## Remaining work, dependency order

1. **Lock correctness evidence before another kernel change.** Preserve the
   completed 262,016+128 telemetry, the short chat/tool transcripts, artifact
   hashes and current native tests. Add a bounded official-reference comparison
   for logits/output quality and sampling behavior; coherent text alone is not
   numerical qualification.

2. **Profile full-attention prefill on the real program boundary.** Attribute
   time and bytes by the 16 full-attention layers, sequence length and prefill
   chunk. Confirm whether the dominant loss is repeated KV traffic, matrix
   staging, launch serialization or the attention kernel before changing code.
   The gate is lower end-to-end TTFT for the same populated prompt with matching
   tokens/numerics.

3. **Optimize the generic long-context attention provider.** Integrate only
   evidence-supported improvements such as wider exact staged prefill,
   asynchronous double buffering, fused QKV/position work, or a better online
   softmax/Tensor-Core schedule. Capability and geometry constraints belong to
   the provider, never a Qwen branch. Re-run the same native oracle and full
   populated request after the complete change.

4. **Qualify exact tiered tree verification.** Follow the acceptance/capacity,
   streamed-verifier and recurrent-commit gates in the active research
   document before extending runtime infrastructure. The acceptance rate is
   approximately 15 tok/s after the first generated token at 262,016-token
   prompt length with exact F16 KV semantics.

5. **Qualify real harness histories.** Exercise append-only multi-turn reuse,
   large system/tool schemas, assistant tool calls and tool results near the
   context limit. Verify that only the delta is prefilled and that truncation is
   explicit rather than silent. Use a representative harness transcript, not
   arbitrary phrase suites.

6. **Close reliability.** Test cancellation during long prefill, generation
   timeout, malformed/oversized bodies, KV exhaustion, worker restart and full
   process-tree cleanup. No failure may leak CUDA ownership or retain a corrupt
   session.

7. **Production edge and supervision.** Keep the built-in service loopback;
   add durable supervision, log/metric retention, alerts and an authenticated,
   rate-limited TLS edge for actual deployment.

## Explicit non-goals

- reducing top-k or approximate expert selection for DeepSeek;
- reducing Qwen context or reasoning effort to manufacture a rate;
- INT8 artifacts presented as FP4;
- a Qwen-specific runner, service task or `model.sh` code branch;
- model-family checks disguised as layer-count or geometry checks;
- CPU/GPU activity maximization as an objective by itself;
- benchmarks or kernel experiments without an acceptance question.

## Resume procedure

1. Read this file and `docs/production-readiness.md`.
2. Inspect the current diff and the immutable artifact manifest.
3. Confirm the service is stopped before rebuilding.
4. Select the first unfinished item above and state its invariant and gate.
5. Plan, implement and test the complete change. Allow at most one local
   corrective patch after a failed result; otherwise reassess end to end.
6. Publish no performance number unless model, prompt length, generated tokens,
   TTFT, wall time and rate definition are all stated.
