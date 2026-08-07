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

The result and limitations are documented in
[External Memory Expert](memory-expert.md). It is a feasibility result, not a
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
records. See [Memory Data ingestion and retrieval](memory-data.md).

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
