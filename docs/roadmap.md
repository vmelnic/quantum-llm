# Roadmap

Status: active dependency order, 2026-09-20. Measurements belong in
[Benchmarks](benchmarks.md); failed ideas belong in
[Research decisions](research-decisions.md).

## Baseline to preserve

- one self-contained Windows host, one primary RTX 3090, one service task and
  one artifact-driven VM;
- seven artifacts and the aliases declared by `ops/model-aliases.tsv`;
- exact declared routing/top-k/aggregation and honest FP4/NVFP4/BF16/F16 names;
- 12 GiB common fixed routed-VRAM ceiling; 13 GiB fails DeepSeek preflight;
- `MODEL_ROOT` as the only host-dependent model root;
- request-selectable thinking, default `xhigh`;
- no external model-state owner or family-specific service runner;
- installed P100s are optional capacity experiments, not a performance
  dependency or default.

## 1. Correctness and recovery

- fault-inject long-prefix cancellation, timeout, KV exhaustion and restart;
- make unhealthy DeepSeek cancellation fail readiness and restart cleanly;
- verify committed retained-state rollback for providers that advertise it;
- retain one artifact-declared streaming parser from first delta through final
  text/tool extraction;
- extend independent reference behavior to stochastic sampling, Qwen images,
  Muse tools and Ornith coding histories.

Acceptance: explicit failure states, no orphaned workers, no partial assistant
commit and no unsupported session commands.

## 2. Qwen maximum context

Target: 262,144 populated exact-F16 positions and about 15 useful output tok/s
in a real coding harness. Cold prefill, reused prefill, saturated decode and
tool-loop wall time are separate gates.

Current exact-F16 prerequisite fails:

```text
hot weights + exact 262K KV = 30,805,569,536 bytes
RTX 3090 physical           = 25,769,803,776 bytes
host-KV scan floor          = 1.284 s/scalar call
```

The explicitly lossy `q4-f16-per-head` policy is now the operational default
for compatible Qwen, Abliterated and Ornith artifacts. Qwen measured 54.54 and
47.16 useful tok/s at 32K/128K; the two-point 262K extrapolation is 39.94 tok/s,
not a live maximum-context result. The first Qwen Abliterated real-project Pi
gate failed with lexical loops at 133K/148K prompt length. Qwen and Abliterated
now use an artifact-declared `presence_penalty=0.5` policy and a 32K per-turn
circuit breaker. The initial 1.5 value caused Romanian lexical and grammatical
degradation and was rejected. Ornith declares the same 0.5 penalty while
retaining its prior sampling defaults. Romanian and long-context
requalification remain pending.
ContextBench remains pending. The exact-F16 release target above is unchanged.

Persistent, codec-preserving Qwen session snapshots are implemented behind the
provider capability: immutable content-addressed NVMe chunks plus a
transactional manifest store the selected KV codec, recurrent/hidden state,
exact artifact/tokenizer/template/media identity and token prefix. A real Pi
restart/resume gate restored 443 K1 tokens from 169,443,798 bytes in 0.203 s
and prefilled only a 34-token suffix. Automatic TTL/LRU and explicit
`model.sh clear-cache <model>` cleanup are available.

The K1 deterministic uninterrupted-versus-restored next-token parity gate has
passed. Still required is a real large coding-session measurement. This is a
resume optimization only: it cannot improve the first cold prefill and is not
arbitrary KV injection.

Do not reopen ordinary offload, rejected speculative proposers or P100 KV/
dense sharding without a new prerequisite that changes their failed math.

## 3. DeepSeek settled throughput

Novel routes near 1 tok/s are acceptable; settled target remains 10-15 tok/s.

- close cancellation/recovery first;
- measure novel and settled NVMe/RAM/VRAM traffic on the primary path;
- preserve demand priority, exact router/top-k/merge and 12 GiB preflight;
- optimize only the critical-path byte transfer or synchronization identified
  by request telemetry;
- keep a change only if real direct chat improves without breaking Qwen and
  cleanup.

The best recorded settled primary path is 6.54-6.69 tok/s. The P100 path
reached 3.18 tok/s and is closed as a speed direction; do not patch its Pascal
kernel again without a complete design that first proves <=100 ms/token.

## 4. Qualify callable artifacts

| Artifact | Next meaningful gate |
|---|---|
| Qwen Abliterated | explicit behavioral/coding comparison before any quality claim |
| Qwen Flash | official operation parity, then the bounded QSA+Q4H experiment below and representative coding; P100 route is not an accelerator |
| Mistral | identify and reduce measured TTFT; Q4L below is a long-context capacity experiment, not a short-prompt TTFT fix |
| Muse | populated 131K text and representative tools, then the bounded global/sliding Q4H experiment below; vision remains auxiliary |
| Ornith | long-context per-head-Q4 quality and correct coding completion without routed churn |

Minimal `hi` and valid tool transport do not satisfy these gates.

## 5. Packed attention-state extensions (explicitly lossy)

These experiments generalize the direct-packed-attention principle, not the
current Qwen kernel or its ABI. Weights, recurrent state and routing semantics
remain artifact-native and unchanged. Each candidate must be selected through
artifact-declared geometry and capabilities; common code must not branch on a
model name. The 12 GiB common routed-VRAM ceiling and 1 GiB reserve remain
unchanged unless all required cross-model preflights and chat gates justify a
separate change.

All three artifacts declare zero MTP layers, so they accept one target token
per call and receive no batch-five or proposer/verifier benefit. A candidate
must read its packed state directly in attention, with no F16/BF16 cache
materialization. It also needs an independent numerical oracle, fail-closed
encoding/geometry validation, a populated-context service measurement and an
explicit user-approved quality gate. Passing capacity or a CUDA microbenchmark
alone is not promotion evidence.

Implementation order is QSA+Q4H, MLA+Q4L, then global/sliding Q4H. Preserve the
full wire names for ABI validation; `Q4H` and `Q4L` are documentation names.

### 5.1 Qwen Flash: sparse QSA+Q4H

The cached K/V is headwise and can use Q4H, but the exact QSA index must remain
unchanged. The existing dense contiguous Q4H batch-five kernel cannot serve a
sparse selected-token list. Add an artifact-declared sparse-QSA Q4H capability,
packed page store and kernel that gathers selected Q4H records directly and
expands only transient operands inside the attention tile.

At the active 262,144-position ceiling, with 12 QSA layers, two KV heads and
head dimension 256:

```text
exact F16 K/V       12 * 2 * (K+V) * 256 * 2 B * 262,144 = 6.0000 GiB
exact F16 QSA index 12 * 1 * 128 * 2 B * 262,144         = 0.7500 GiB
Q4H K/V             12 * 2 * 260 B * 262,144            = 1.5234 GiB
Q4H + exact index                                             2.2734 GiB
capacity recovered                                            4.4766 GiB
```

For a 2,048-token selected set, the K/V payload touched across all 12 layers is
48 MiB in F16 versus 12.1875 MiB in Q4H per target query; QSA index/search
traffic is additional and unchanged. Short profiling attributed only about
1.2% of execution to QSA/full attention and 84.7% to routed MoE, so direct
attention compression alone cannot be claimed as a large throughput gain. The
primary hypothesis is that the recovered capacity permits a larger
artifact-specific routed-expert cache. Re-run future-state reservation and
preflight before admission; do not turn that fit into a higher common cache
ceiling.

Acceptance requires exact-index parity, sparse-selection parity, an independent
Q4H attention oracle, measured selected-record/index traffic, and representative
short plus populated-context service runs. Keep the change only if total
service throughput or latency improves and the separate Pi quality gate passes.

### 5.2 Mistral MLA: direct latent Q4L

MLA does not retain separate K and V heads. It retains one BF16 latent record of
`kv_lora_rank + qk_rope_dim = 256 + 64 = 320` values per layer and token, so
literal per-head Q4H is the wrong ABI. Introduce a distinct experimental `Q4L`
encoding with 160 Q4 code bytes plus two FP16 scales per record: one scale for
the 256-value content latent and one for the 64-value RoPE component.

For 36 MLA layers at the active 262,144-position service ceiling:

```text
exact BF16 latent  36 * 320 * 2 B * 262,144       = 5.6250 GiB
Q4L latent         36 * (160 + 2 + 2) B * 262,144 = 1.4414 GiB
capacity recovered                                  4.1836 GiB
```

At the artifact's 1,048,576-position maximum the same representation is
22.5000 GiB exact versus 5.7656 GiB Q4L; this does not raise the active service
ceiling by itself. A full 262K target call reads at least those 5.6250 or
1.4414 GiB of latent payload. At 936 GB/s, their ideal HBM-only floors are about
6.46 and 1.65 ms before index, softmax, projection, routed-weight or launch
costs.

The new MLA kernel must perform the absorbed-query dot product and weighted
latent accumulation directly from Q4L, without reconstructing full K/V or a
BF16 latent cache. Use a new artifact encoding/capability and versioned ABI;
share only validated nibble/scale primitives with Q4H. Acceptance requires an
independent latent-attention oracle covering both scale domains, a populated
262K service gate, and Pi quality. Short Pi prompts retain little latent state,
while current latency is dominated by routed work and weights, so Q4L is not an
admitted short-TTFT optimization.

### 5.3 Muse: global/sliding Q4H

Muse retains ordinary headwise K/V, so Q4H semantics apply, but its geometry and
access pattern do not: head dimension is 128, the query/KV-head ratio is 16,
13 layers use global paged attention and 39 layers use a cyclic 2,048-token
window. Generalize Q4H records to 64 code bytes plus one FP16 scale for each
128-value K or V head, then add direct packed kernels for both global pages and
sliding-window wraparound.

At Muse's 131,072-position ceiling, with two KV heads:

```text
F16 global   13 * 2 heads * (K+V) * 128 * 2 B * 131,072 = 1.6250 GiB
F16 sliding  39 * 2 heads * (K+V) * 128 * 2 B * 2,048   = 0.0762 GiB
Q4H global   13 * 2 heads * 132 B * 131,072             = 0.4189 GiB
Q4H sliding  39 * 2 heads * 132 B * 2,048               = 0.0196 GiB
total                                                    1.7012 -> 0.4386 GiB
capacity recovered                                                1.2626 GiB
```

Reading the full retained K/V once gives an ideal HBM-only saving of about
1.45 ms per target token at 936 GB/s; weights, attention arithmetic and other
runtime work remain. Acceptance requires independent global and sliding
oracles, including first-token, full-window and cyclic-wrap boundaries, plus a
populated 131K service run and representative text/tool quality gate. Do not
infer a 40--50 tok/s result from the capacity calculation.

## 6. Production operations

- unify admission for active/retained KV, sessions and routed RAM ownership;
- add durable supervision, bounded logs/metrics and actionable readiness;
- verify immutable-artifact backup, rollback and failed-start recovery;
- put TLS and rate limiting in a trusted edge before public exposure;
- provide one release command for clean build/tests, required chats, selected
  Pi gates and final process/GPU cleanup.

## Completion

Qwen completes only at the populated exact-F16 real-harness target. DeepSeek
completes only when exact novel/settled traffic and recovery meet their accepted
rates. Other models complete at their numerical, populated-context and harness
gates—not at configured context, parser smoke or `hi`.
