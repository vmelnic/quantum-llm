# Engineering history and decision record

This document preserves what was built, measured, rejected, and learned. It is
historical evidence, not the current runbook. Current operation is documented
in [Deployment](deployment.md), current performance in
[Performance evidence](benchmarks.md), and remaining work in
[Roadmap](roadmap.md).

## Qwen3-Next 80B foundation

The first complete backend targeted `Qwen/Qwen3-Next-80B-A3B-Instruct` on one
RTX 3090 with approximately 64 GiB system RAM. The source checkpoint is about
162.7 GB. The deterministic INT8 Expert Pack is about 81.9 GB and separates a
4.19 GB dense pack from independently addressable expert records.

Implemented and retained:

- strict SafeTensors-to-Expert-Pack conversion and independent validation;
- Windows IOCP storage with bounded RAM and VRAM expert caches;
- exact Qwen attention, DeltaNet, router, MoE, normalization and output head;
- paged FP16 KV, request isolation and continuous decode batching;
- CUDA and all-core CPU expert lanes behind one placement boundary;
- OpenAI-compatible text API, streaming, cancellation, admission and metrics;
- Task Scheduler deployment plus full process-tree stop.

The optimized hot runner reached 47.6703 tok/s single-stream and 42.5976 tok/s
aggregate for four requests. Those gates warmed the exact expert routes in the
same process and observed no expert SSD or H2D traffic during measurement. They
never represented cold or arbitrary chat throughput.

## Hybrid CPU/GPU work

Useful changes that remain:

- compact CPU result transport reduced one qualified batch from 521,830,400 to
  77,996,032 H2D bytes while retaining exact output;
- resident CUDA and CPU lanes can execute selected experts concurrently and
  aggregate in stable router order;
- bounded placement profiles, cache ownership and memory admission fail closed;
- route observations and measured placement costs can guide prefetch/placement.

Experiments that did not solve the user-visible SLO:

- forcing one CPU and five GPU experts was exact but slower than six GPU
  experts on the tested layer (10.54 ms versus 4.40 ms);
- activation-INT8 DP4A regressed the measured path and was removed;
- larger history-only INT8 expert residency exceeded VRAM or still churned;
- disabling prefetch reduced hot aggregate throughput;
- RAM caching removes repeated SATA reads but cannot make unrelated expert
  routes globally hot; the full 81.9 GB pack exceeds the combined hot caches;
- batching unrelated requests improves aggregate capacity, not single-chat
  latency.

## DeepSeek-V4-Flash expansion

The second backend targeted `deepseek-ai/DeepSeek-V4-Flash`, a 284B-class MoE
with about 13B active parameters per token. It forced the runtime beyond the
Qwen INT8 format rather than pretending a new architecture was an input alias.

Implemented and retained:

- pinned mixed FP4/FP8 source contract and authenticated 43-shard compact pack;
- 11,008 routed expert records with independent indexes and checksums;
- resident dense, typed and shared-expert state;
- exact attention/CSA/HCA, routing, FFN, output head and greedy token path;
- compact FP4 RAM/VRAM storage with SM86 compute-ready expansion/compute;
- asynchronous layer scheduler, bounded request ownership and MTP resources;
- persistent worker bundle behind the same HTTP API.

The backend became functionally usable but remained approximately 0.3–0.6
output tok/s on the reference host. Profiling rejected attention alone as the
dominant explanation. Routed working-set movement and expert execution remain
the principal boundary. Moving some work to the six-core desktop CPU did not
approach the 30 tok/s objective.

## Service and lifecycle corrections

The following failures were discovered only by exercising the real lifecycle,
not by readiness-only checks:

| Commit | Failure | Correction |
|---|---|---|
| `1f2fcdd` | Qwen protocol v4 lacked the newer `placement_prefetch_state` field, so the HTTP worker rejected an otherwise valid runner | accept the legacy effective boolean and publish the explicit field in future runners |
| `b514e48` | Qwen chat closed the connection because Jinja was absent | pin Jinja, add `model.sh install`, and preflight dependencies before stopping a running model |
| `3c905c5` | Qwen returned scalar `token`, while the MTP-capable front-end expected `tokens[]` | accept both protocol forms and publish the list form in future runners |
| `86a4b3e` | incomplete UTF-8/byte-fallback tokens produced `�` and caused the entire decoded prefix to be streamed again | hold unstable replacement suffixes until the tokenizer produces a stable Unicode prefix |

Additional lifecycle behavior now retained:

- `model.sh start` synchronizes, preflights dependencies, stops competing
  tasks, installs the selected profile, waits, and verifies identity/limits;
- task death during startup fails quickly instead of consuming the full ready
  timeout;
- chat resolves the model actually served by `/v1/models`, so `start qwen`
  followed by `model.sh chat` cannot send a stale DeepSeek model ID;
- preprocessing failures return structured HTTP errors instead of silently
  dropping the socket.

## Context-window decision

Both deployments currently advertise a 65,536-token operational ceiling and
an 8,192-token output ceiling from `.env`. DeepSeek checkpoint metadata
advertises 1,048,576 positions; Qwen metadata advertises 262,144.

Only the 4,096-token correctness profile has completed the historical
long-context qualification gates. The larger setting fits the current logical
KV budgets for one long request, but it is not a claim of validated quality,
latency or concurrency at 65K.

## Distributed direction retained for later

Multiple machines will not expose transparent combined RAM. The design moves
activation compute to the node that owns an expert. A coordinator retains
dense/attention/router state, sends activations plus expert IDs and routing
weights, and receives strict or weighted partial outputs.

The future protocol uses persistent standard transport with a binary
application framing layer, separate control/data planes, manifest and kernel
ABI negotiation, credits, cancellation, epochs, idempotent operation IDs and
explicit failure semantics. “Expert Pack v1” remains a working placement and
streaming format; the name QMoE was rejected because it collides with an
existing quantization project.

RTX 3090/CUDA remains the first backend. Metal/macOS and remote expert workers
remain later phases after the local runtime has truthful single-host evidence.

## External Memory Expert experiment

The project also tested whether a frozen model can consume newly ingested
knowledge without prompt stuffing, corpus-wide KV materialization, or
per-ingest training. The final Qwen3-4B PoW trained 2,949,120 low-rank gate
parameters and passed both causal and held-out gates: 12/12 contradictory
memory interventions exact, 16/16 oracle held-out exact, and 15/16 through
automatic retrieval.

Several rejected intermediate results were essential:

- one late-layer cross-attention branch matched the no-memory baseline and did
  not transmit arbitrary values;
- the first all-layer version used the output of layer `i` as memory for layer
  `i`, an off-by-one state, generic LayerNorm instead of the frozen RMSNorm,
  and reversed zero/random gate initialization;
- LAMB at the short PoW horizon made one large first move and then nearly
  stalled; AdamW at 2e-4 remained stable and converged in the bounded run;
- ordinary examples allowed a question-to-answer shortcut; same-question,
  same-visible-ID contradictory records made memory content the only causal
  discriminator;
- the neural channel selected codes correctly but reconstructed digits
  approximately; an admitted-source continuation pointer made literals exact;
- unconditional nearest-neighbor admission hallucinated on absent facts; a
  held-out score floor separated known and unknown synthetic queries.

The result and limitations were documented in the External Memory Expert
notes (since extracted to the standalone `memory-expert` project). It is a
feasibility result, not a
claim that the current synthetic index is ready for real private data.

## First real Memory Data slice

The next slice introduced a generic immutable record contract, configurable
JSON/text adapters, article-aware but replaceable segmentation, a pinned
BGE-M3 dense encoder, FP16 vector shards, ACL/language admission, and
authority-plane quote rendering. One Romanian legal document became 731
records in 0.025 seconds and its index was computed in 3.98 seconds. Five
Romanian questions retrieved the correct article at rank 1 in every case.

The existing English synthetic Memory Expert adapter answered only two of five
strictly. Top-2 versus top-1 admission did not change that conclusion. This
localized the next problem to multilingual/extractive adapter capability,
rather than ingestion or dense retrieval. Model-generated synthetic citation
IDs were rejected; exact citations were rendered only from ACL-admitted source
records. See the Memory Data ingestion and retrieval notes (since extracted
to the standalone `memory-expert` project).

## Multilingual capability benchmark audit

The next adapter initially appeared to improve its internal multilingual
metrics, but an independent code-and-data audit invalidated that conclusion.
The corpus was a synthetic codebook: patterned `ZX...` keys, 18 closed span
labels, and citation IDs correlated with record identity. More importantly,
the causal probe sampled training families instead of eval families and
automatic/oracle modes reported different example subsets.

The untouched Romanian law run exposed the mismatch: evidence retrieval stayed
at 5/5, strict text match was 0/5, and manual semantic review found only one
clearly correct answer. No Qwen or DeepSeek production-model artifact was
modified by this experiment.

The invalid generator was removed rather than patched. Its replacement uses
pinned natural XQuAD passages and questions in Romanian, Russian, and English,
same-question counterfactual memories, distractors, opaque identifiers,
absent-evidence cases, atomic duplicate/translation splits, an eval-only causal
probe, paired retrieval/oracle reporting, and a bounded lazy layer-state cache.
Until this replacement passes both its natural eval and the untouched legal
test, the project claims only channel causality—not generic external knowledge.

A first replacement run was stopped before completion when review found that
384-token memory truncation could hide the answer in 162 examples. A later
review caught a deeper corpus defect before promotion: its “counterfactual”
passages were made by substituting answer spans from unrelated examples, which
could produce incoherent language and ambiguous questions. That artifact and
its partial checkpoints do not establish natural memory capability.

The replacement contract now has no substitution fallback. It emits rewrite
jobs from bounded natural passages, requires a coherent same-language rewrite
plus an independent answer-extraction validation, preserves the same question
and source position across each causal pair, and fails closed on missing or
stale rewrites. Neural supervision uses request-local `SOURCES` slots; durable
citation hashes remain metadata and are mapped to exact quotes only by the
authority plane. Full multilingual generation and training remain pending.

That proposed rewrite pipeline was then removed before use. Existing public
datasets were a better source than generating another private benchmark.
`conflictqa-causal-memory-v3` now consumes the pinned Apache-2.0 ConflictQA
artifact through the official Hugging Face CLI and Xet. It retained 7,940 of
7,947 same-question contradictory-memory families, rejected seven whose
authoritative context still contained the sibling answer, and produced 16,674
English capability examples. FaithEval and ParaConflict were downloaded at
pinned revisions as untouched future external evaluations. No Qwen teacher
generation occurred, and no production-model artifact was changed.

## ConflictQA v3 result and implementation audit

The v3 adapter trained 2,949,120 gate parameters while keeping Qwen3-4B
frozen. Its advertised 4,096 steps were microsteps, not optimizer updates:
batch size 2 with gradient accumulation 16 produced 256 updates and consumed
8,192 of 13,267 training examples, or 0.62 epochs. The run completed in
1,607.87 seconds, reduced observed minibatch loss from 2.71209 to 0.02451 and
peaked at 12.21 GiB VRAM.

The mandatory eval-only causal probe failed promotion. All six tested families
were memory-sensitive and contrastive candidate accuracy was 12/12, showing
that the separate channel affected the answer. Exact generation was only 9/12:
`Daniel Mulloy` became `Daniel Mullony`, `Bobby Hebb` became `Bobbie Hebert`,
and the Akari Hayami answer followed parametric knowledge rather than the
admitted evidence. Held-out evaluation was not run after this prerequisite
failed. The checkpoint remains an ignored diagnostic artifact.

The review found the authors' previously missed
[TokenMem repository](https://github.com/iomgaa-ycz/TokenMem). It contains the
reference model modifications, two-phase training and evaluation code, but no
published gate checkpoint or declared repository license. Comparison exposed
curriculum, cross-attention RoPE, memory-compression and target-format
differences. It also exposed a local contract defect: `dataset-label` rows
could pass visible-evidence validation without the exact answer occurring in
the admitted text. A future run must resolve these points structurally rather
than patch the three failed outputs.

The subsequent review separated a proven defect from untested architectural
hypotheses. The proven defect was training geometry: v3 stopped after 0.62
epochs and 256 optimizer updates. Plan A therefore keeps the independent
architecture and fixes full-epoch scheduling, exact token-weighted gradient
accumulation, per-epoch held-out NLL, best/last checkpoints, and resumable
state. With the current corpus and defaults, v4 is exactly three epochs,
39,801 example visits, 19,902 microsteps, and 1,245 optimizer updates. TokenMem
is retained as Plan B rather than treated as a required rewrite.

[KBLaM](https://github.com/microsoft/KBLaM),
[MeMo](https://github.com/arunv3rma/MeMo) and
[delta-mem](https://github.com/declare-lab/delta-Mem) were recorded as adjacent
research, not equivalent products. None of the reviewed systems combined
instant ingest, hundreds-of-GB sharding, latent injection, ACL/version
semantics, fail-closed answers, exact citations and a production API.

## ConflictQA v4 and Romanian external result

V4 removed the undertraining ambiguity. It completed three full epochs,
39,801 example visits and 1,245 optimizer updates. Its in-distribution ConflictQA
evaluation reached 28/30 strict oracle answers and 30/30 source selections,
compared with 6/30 strict answers without memory. Unknown-memory abstention was
perfect. Automatic retrieval found only 3/10 authoritative records, so the
complete PoW gate did not pass.

More importantly, the unchanged Romanian legal test isolated generalization
from retrieval. BGE-M3 admitted the intended evidence for all five questions;
the adapter answered only the 16-year criminal-responsibility question
correctly. Manual inspection confirmed that the other four were semantic
failures rather than harmless matcher misses. No further tuning is permitted
against those five external questions. The next experiment uses the official
Doc-to-LoRA Qwen3-4B checkpoint as a materially different no-per-document-
training baseline. Generic multilingual semantic scoring remains a TODO and
is not allowed to obscure this failed promotion result.

## English out-of-distribution closure

An adversarial code review found that ConflictQA evaluation was not fully
held out: the best epoch checkpoint used a deterministic prefix of the eval
split for selection, and the reported 30 examples came from that prefix. It
also found a serving-format mismatch: capability training encoded bare record
bodies, while real serving prepended a durable ID and document metadata. The
retrieval gate additionally measured a placeholder hashing index rather than
the BGE-M3 serving path. Qwen's tokenizer was checked directly and confirmed
to right-pad, eliminating the suspected padding error. The actual v4 run used
the same 768-token bound in training and serving, so length configuration did
not explain the external failures.

A same-day English control used a wholly invented five-section field dossier
and eight natural questions. The result matrix was:

| Arm | Evidence retrieval | Strict answers | Valid source selection |
| --- | ---: | ---: | ---: |
| V4, serving metadata | 8/8 | 2/8 | 1/8 |
| V4, raw body matching training | 8/8 | 3/8 | 8/8 |
| Frozen Qwen, full prompt context | all five records | 8/8 | 8/8 |

The raw-body ablation proves that metadata noise broke request-local source
binding, but its one-answer gain cannot explain the factual corruption. The
adapter changed 19 hours to 29.82 minutes, 27 minutes to two hours and a 2037
date to 2017 despite receiving the exact evidence. The v4 architecture as
trained is therefore closed as a ConflictQA-specific reader. No further v4
training or matcher tuning is planned; the next baseline must use a materially
different document-internalization mechanism.

## Mechanism probes and the v5 pointer closure

Inference-only instrumentation established the v4 failure mechanism before any
new training. An internal probe (attention mass, entropy, residual norms over
all 36 layers, prefill and decode) showed the gate stays active out of
distribution and that correct and failed cases are internally
indistinguishable; the null key is masked whenever memory exists, so its mass
is zero by construction. A token-level trace at the answer-divergence step
showed the gold literal reaching the final logits at ranks 2–100 — once losing
on an exact logit tie — with diffuse attention and a readable but weak
mid-stack signal. Interventions then closed the cheap fixes: scaling the gate
residual by 1.5–3 degraded both out-of- and in-distribution accuracy into
degeneration (the signal is not amplifiable), and unmasking the null key
changed nothing. An NLL span-ranking test with a no-memory control showed
selection is only marginally memory-driven out of distribution (5/8 gold at
rank 1 versus 3/8 blind; one dramatic swing on the exact quotation).

V5 tested the resulting design: pointer targets (`ANSWER: @slot:sentence`)
with the authority plane rendering the verbatim sentence, so the channel only
selects and never renders literals. The regenerated corpus
(`conflictqa-causal-memory-v4`, 12,664 rows) rejects the 1,912 families
without a confidently derivable gold sentence. Training completed three epochs
(30,153 visits, 945 updates, best validation NLL 0.0147). The causal probe was
repaired for the pointer contract (contrastive NLL on shared sibling
coordinates is degenerate) and passed semantically: 6/6 evaluation families
rendered distinct memory-tracking content. Held-out evaluation reached 28/30
pointer-exact oracle answers, 30/30 sources and 100% unknown abstention.

The pre-registered external bar (≥7/8 English dossier, ≥4/5 Romanian code)
decided: v5 scored 2/8 on the dossier (neighbouring-sentence pointers,
malformed pointers, one abstention) and abstained on all five Romanian
questions, with retrieval at 5/5. V5 is closed. Its failures were fail-closed:
abstentions or verbatim admitted quotes, never invented literals. The
three-version arc now bounds the design space: the synthetic PoW needed a copy
pointer for exact digits, v4 proved a rank-16 residual cannot render literals,
and v5 proved the same gate cannot select finely enough out of distribution
when trained on a single English corpus. The trained capability tracks the
training distribution, not "admitted memory" as a general channel.

## Memory Expert extraction

The KV-attach pivot that followed the v5 closure — frozen native attention
reading a lazily prefilled record K/V prefix, plus a 2.4M-parameter LoRA
reader trained once on ConflictQA — validated its first two milestones on
2026-08-08 (frozen: Nacre 8/8 strict answers; LoRA: 8/8 and 5/5 correct
abstentions on the external sets). The complete experiment — code,
architecture and results documentation, the Nacre demo set, and the
operations runbooks — was then extracted from this repository into the
standalone public `memory-expert` project. This chronology is kept intact
as the design-space evidence that led there; nothing described above
changes, and the memory-expert docs, ops scripts, and experiment code no
longer live in this repository.

## Serving-path campaign (W0–W5)

Executed 2026-08-08 to 2026-08-09 against the plan in
[Path to 30 tok/s](inference-30toks-plan.md); full measurements in
[Performance evidence](benchmarks.md). Landed and retained:

- W0 per-phase request telemetry through the worker STATS command;
- W1 protocol v5 retained conversation sessions (multi-turn prefills only the
  per-turn delta; TTFT flat 3–9 s, was 17–28 s growing with history) and
  prefill chunking decoupled from the decode batch capacity;
- W2 asynchronous directory planning, worker-mode placement freeze after a
  warmup boundary, and vectorized GEMV dispatch — resident route 27.2–27.7
  tok/s with zero disk bytes (was ~1–2);
- W3 event-driven uploads (no stream-wide syncs on the hot path) and a wider
  DeepSeek prefetch pipeline; the Qwen route-ahead prefetch variant was
  measured and reverted (predicted experts were already RAM-resident; 2–7×
  VRAM churn);
- W4 bounded router-aware re-promotion while frozen: settled topics reclaim
  VRAM from dead ones (up to −43% turn wall) with no added storage reads;
  resident route holds 27.8–28.9 tok/s. Two aggressive gating variants were
  measured and discarded (ping-pong churn). The plan's resident-hit ≥ 0.8
  criterion was not met;
- W5 DeepSeek MTP enabled end-to-end and measured throughput-neutral
  (0.47–0.54 → 0.49–0.59 tok/s): a verify pair pays the union of two
  adjacent routes through the same bandwidth-bound pipe. Deployment keeps
  MTP on — correct, retention-compatible and self-suppressing. The 8–15
  tok/s hope for DeepSeek on this host is closed; the 3.21 GiB/token
  arithmetic wall stands.

Final state: Qwen3-Next-80B meets the 30 tok/s target on the resident route;
non-resident chat sits at the SATA first-touch floor (~3 tok/s int8).

## FP4 routed experts (S1)

Follow-on work tracked in [Scaling next steps](inference-scaling-next.md):
the FP4-E2M1/UE8M0 block-32 format already proven by the DeepSeek compact
path was wired for Qwen routed experts as Expert Pack quant ABI 3
(compiler encoder, manifest/schema/validator, uploader/directory dispatch,
packed `__dp4a` GEMV kernels), and the FP4 pack (45.36 GB vs 72.3 GiB int8)
was measured on 2026-08-10 against a same-configuration int8 arm:
resident steady 39.6–45.6 tok/s (int8 7.75), novel routes 5.3–16.9
(int8 1.9–4.0), behavioral probe 10/10 coherent at ~11.8% relative L2
weight error. Two structural bugs were fixed to get there: pinned-allocator
4096-alignment, and an eviction livelock (eviction now skips route-pinned
victims and `try_retire()` never spins). The whole FP4 pack fits the 48 GiB
RAM budget, so the SATA floor is cold-start-only. The host bandwidth
measurements behind these numbers (H2D pinned 12.46 GiB/s over PCIe Gen3
x16, SATA sequential 0.47 GiB/s) are pinned in
[Performance evidence](benchmarks.md).

## S1-DeepSeek: FP4 residency accounting (2026-08-10)

Direct-FP4 execution for DeepSeek routed experts had already landed in
`05b474e` (packed `__dp4a` kernels, compact records kept as-is in VRAM),
but the routed catalog still accounted every slot as the 25,198,592-byte
int8 layout, so the VRAM tier and the census warm set ran at half their
physical capacity (warm set hard-capped at 6 experts/layer). The fix
declares the compact record size (13,369,344 bytes) as the routed device
footprint and derives the per-layer warm cap from the VRAM entry budget.
A route-skew measurement on the persisted census justified the deeper warm
set: top 13% of experts cover 92.5% of cumulative route mass. The 3-turn
probe improved ~7.4x over the W3 baseline (turn-3 decode 3.5 vs 0.47
tok/s), and an MTP ablation showed MTP is now a 2-3.5x decode multiplier
under FP4 residency (the W5 "neutral" verdict belonged to the int8-slot
regime), so MTP stays on. Remaining bottleneck, decomposed in
[Performance evidence](benchmarks.md) §S1-DeepSeek: SATA misses on
fresh-topic routes (storage wait dominates wall time); the next levers
are hardware (NVMe tier, RAM >= 192 GB for a fully RAM-resident pack),
not runtime changes. Quality-risky options (sub-4-bit requantization,
dynamic top-K) were considered and explicitly rejected.
