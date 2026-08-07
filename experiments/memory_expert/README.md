# Memory Expert proof of concept

This experiment tests one claim only: a small trained residual branch can make
a frozen language model answer from external evidence that is never inserted
into its token context.

It does not modify, convert, or load the Qwen3-Next-80B or DeepSeek-V4-Flash
artifacts. The first backbone is the official Qwen3-4B BF16 checkpoint pinned
by commit. Its size leaves enough RTX 3090 memory for adapter training and
downstream activations.

## Information path

```text
external records ─► separate frozen pass ─► layer-input memory states
          │                                     │
          └──────── exact source tokens ────────┼──► continuation pointer
                                                │
question tokens ─► frozen block 0 ─► frozen Q/K/V/O cross-attention + low-rank gate
                  frozen block 1 ─► frozen Q/K/V/O cross-attention + low-rank gate
                                      ...
                 frozen block 35 ─► frozen Q/K/V/O cross-attention + low-rank gate
```

The external records are tokenized only for the separate memory pass. They are
not concatenated with the question, do not receive request positions, and do
not consume the request's context window. The initial PoW caches those JIT
representations only for its tiny generated corpus. It does not propose
materializing KV or hidden states for a production corpus.

The branch is attached after each decoder block. It reuses that layer's frozen
Q/K/V/O projections and trains only an independent low-rank residual gate
(rank 16 by default). It is parallel to the frozen block; it does not replace an
MoE expert or change original router probabilities. The gate's output factor is
zero-initialized, so every branch is an exact identity at initialization. The
layout follows the central TokenMem topology closely enough to test the channel
rather than an intentionally weaker single-layer approximation.

Layer `i` consumes the memory pass state at `hidden_states[i]`, not the output
at `i + 1`, and uses the same frozen RMSNorm as the decoder layer. Exact source
token identities are retained beside those states. Once generation begins a
meaningful span from admitted memory, the continuation pointer favors the exact
next source token. This prevents continuous hidden states from approximating a
digit in an otherwise correctly selected code.

## Corpus and evaluation

`synthetic_memory.py` deterministically creates an English fictional universe
with names, dated transfers, artifacts, locations, access codes, and stable
record IDs. Evaluation entities do not occur in training entities. It includes:

- single-record questions;
- two-record composition questions;
- absent facts that require the exact abstention response;
- irrelevant-record negatives;
- exact citation IDs.

Training also contains several non-indexable counterfactual records for each
question. The question and public citation ID are identical while only the
admitted source content and answer change. Internal cache keys differ but are
never present in text shown to the model. Therefore neither a question nor an
ID shortcut can solve the training objective: the only causal variable that
identifies the target is the memory content. These records never enter the
automatic retrieval index.

Every generated answer has the closed form:

```text
ANSWER: <answer or exact abstention>
CITATIONS: <record IDs or NONE>
```

Three modes are reported separately:

1. `no-memory` tests the required abstention path;
2. `oracle-memory` tests the neuronal channel independently of retrieval;
3. `automatic-retrieval` tests the channel plus the sharded index.

A fourth `full-context-ceiling` mode is an evaluation control only. It shows the
same records to the frozen model as text and disables the adapter. It is never
part of the proposed serving path; it tells us whether a failed memory result
comes from the new channel or from the backbone/task itself.

The PoW passes only when oracle memory reaches 70% joint answer+citation
accuracy, improves by at least 30 percentage points over no-memory, absent
facts abstain at least 80% of the time, and retrieval recall reaches 75%.
Oracle memory must also retain at least 80% of the full-context control score.
These are feasibility gates, not production accuracy targets.

## Validated result

The pinned 1,024-step AdamW run passed every gate. On 16 held-out examples:

- oracle memory: 100% joint answer+citation accuracy;
- automatic retrieval: 93.75% joint accuracy and 100% retrieval recall;
- oracle citations: 100%;
- absent-fact abstention: 100%;
- causal intervention probe: 12/12 exact and 12/12 memory-sensitive;
- peak allocated VRAM: approximately 10.85 GiB;
- adapter parameters: 2,949,120.

The full report and interpretation are in
[`docs/memory-expert.md`](../../docs/memory-expert.md).

## Scale contract

The generated vector index writes a manifest and bounded shards. It opens
adaptively: small vector payloads become resident, while payloads above a RAM
budget are searched one memory-mapped shard at a time with only bounded top-k
candidates retained. Retrieval first finds a credible root and can spend a
second bounded hop on that record's lexical/entity neighborhood; this lets a
dependent source be found even when it does not repeat the entity from the
question. Records have stable IDs, generation, language, ACL, and shard keys.
This proves bounded storage and multi-hop behavior, not production-scale search
latency: exact scans and lexical expansion must later become routed ANN shards
plus typed links and a learned hop budget. The current hashing encoder is a
deterministic test double behind the same vector interface; it must be replaced
by a compact multilingual encoder before real ingestion.

At production scale the tiers are:

```text
cold  raw authoritative records + full-text index
warm  compact/PQ embeddings and metadata shards
hot   learned/JIT memory slots for observed frequent records only
```

No corpus-wide Qwen KV or DeltaNet state is allowed. Admission count, memory
tokens, shards, and hot-cache bytes remain independently bounded and may adapt
to score margin, query complexity, memory pressure, and observed heat.

## Run

From the control host, after configuring `.env`:

```bash
./ops/memory-pow.sh download
./ops/memory-pow.sh download-status
./ops/memory-pow.sh run
```

Generated state is under `work/memory-expert-pow` on the GPU host. The wrapper
syncs tracked sources, validates an isolated Python environment, executes its
structural self-test, requires 18 GiB free VRAM, and runs offline against the
pinned local snapshot. `run` stops before held-out evaluation if the causal
probe fails. `train`, `probe`, and `evaluate` remain available separately for
diagnosis.

New records do not require adapter training. Production follow-up replaces the
test-double index and synthetic corpus, adds a serving API and security/data
lifecycle, then ports the interface independently to larger backbones. Adapter
weights are not portable between residual spaces.

The first generic real-data path now lives beside the PoW:

- `ingest.py` and `data_contract.py` create immutable, model-independent shards;
- `dense_index.py` builds a pinned multilingual index;
- `real_query.py` separates retrieval, neuronal answers, and authoritative
  evidence rendering;
- `configs/` contains replaceable source profiles rather than source-specific
  paths in code.

Its measured Romanian legal result and current adapter limitation are in
[`docs/memory-data.md`](../../docs/memory-data.md).
