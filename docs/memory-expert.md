# External Memory Expert

## Current status

The project has proved that a frozen Qwen3-4B can receive information through
a separate layer-wise memory channel. It has **not** yet proved that the adapter
generalizes to arbitrary natural documents.

The distinction matters:

- the original English synthetic PoW passed its own held-out suite, but the
  suite was small and structurally regular;
- the first multilingual capability corpus was invalid as a generalization
  test: it used a closed label vocabulary, patterned `ZX...` keys, correlated
  citation IDs, and ran its causal probe on training families;
- on the Romanian legal document, retrieval found the intended evidence for
  5/5 questions, while that adapter produced 0/5 strict text matches and only
  one clearly correct answer under manual semantic review;
- those results prove evidence admission, not useful natural-language memory
  capability.

The invalid multilingual checkpoint remains historical data and must not be
promoted. The current gate is `xquad-coherent-counterfactual-memory-v2`; its
full training and evaluation result must be recorded before claiming success.

## Architecture under test

```text
immutable source records
        │
        ├── model-independent shards, ACL, language, generation, source spans
        └── bounded retrieval/admission (records never enter the user prompt)
                              │
                              ▼
                    admitted record set only
                              │
                 frozen Qwen memory encoding pass
                              │
          bounded lazy cache of decoder-layer input states in CPU RAM
                              │
                              ▼
question-only prompt ─► frozen decoder layers
                              │
                 frozen RMSNorm + Q/K/V/O cross-attention
                              │ at every selected layer
                 trainable rank-16 residual gate only
                              │
                              ▼
                    answer + request-local source slots
                              │
             authority-plane slot→record mapping, ACL, exact quotes
```

For decoder layer `i`, the memory pass supplies `hidden_states[i]`, the input
of the same layer. The branch reuses the frozen layer's normalization and
attention projections. Only 2,949,120 low-rank gate parameters are trainable;
the Qwen checkpoint stays frozen.

This is not a prompt-RAG generation path: source text does not consume the
conversation context or causal KV cache. A bounded selector admits records
because hundreds of gigabytes cannot be activated for every question, but the
admitted records reach the decoder only through the separate layer-wise memory
channel. The capability corpus below trains that channel; it is not the user's
knowledge base and it is not rebuilt when knowledge is ingested.

## Generic capability protocol

The adapter is trained once to perform a capability—read admitted natural
memory, follow its current authority, cite it, and abstain—not to memorize a
particular ingest.

The capability corpus is constructed from the pinned `google/xquad` revision
`51adfef1c1287aab1d2d91b5bead9bcfb9c68583` (CC-BY-SA-4.0):

- Romanian, Russian, and English natural questions and passages;
- original and coherently rewritten counterfactual memories for the same
  natural question;
- natural answers, opaque IDs, distractor records, and absent-evidence cases;
- duplicate questions and their translations are assigned atomically to one
  split, preventing train/eval leakage;
- each rewrite must change the answer, remove the original answer, remain in
  the source language, stay coherent, and be answerable by an independent
  extraction pass; blind answer-span substitution has no fallback;
- answer-bearing passages are bounded around exact source spans, and a
  tokenizer-aware preflight requires every answer and local source label to remain
  visible after the configured 384-token truncation;
- the model emits only zero-based request-local slots such as `SOURCES: 1`;
  opaque record IDs, ACL decisions, and exact quotes are mapped and rendered by
  the authority plane after generation, never learned as output tokens;
- the evaluated legal questions are forbidden training material.

The loader rejects the old patterned keys, closed-span markers, citation IDs in
questions, non-extractive targets, family split leakage, missing language
coverage, and single-record-only corpora.

Training mixes original, counterfactual, and abstention examples from the first
optimizer phase. The causal probe now uses eval families only. Automatic
retrieval is reported only on eligible natural records and is compared with an
oracle score over the exact same example IDs. Retrieval recall follows the
authoritative citation, not the deliberately injected distractor record.

## Bounded execution

The training corpus is never pre-encoded into layer states. A lazy LRU cache
encodes only memory sets needed by the current batch, stores BF16 states in CPU
RAM, and evicts cold sets under `MEMORY_CAPABILITY_CACHE_BYTES`. The active
batch may temporarily exceed the cache target, but memory use does not scale
unbounded with total corpus size.

At ingest time no model weights change. Raw records and their compact retrieval
index are durable; only admitted hot records receive temporary model states.
New laws, books, orders, or user data therefore require zero adapter training
unless the desired *capability* changes.

## Evaluation contract

Three different claims are kept separate:

1. **retrieval/admission** — did the correct immutable evidence enter the
   memory channel?
2. **strict text match** — does generated text contain configured expected
   fragments? This is deterministic but is not semantic accuracy.
3. **semantic correctness** — does the answer actually answer the question and
   follow the evidence? Until a separately validated judge exists, this
   requires manual review.

The capability run must pass eval-only contradictory-memory preference,
generation, source selection, abstention, no-memory, full-context-control, and paired
retrieval/oracle gates. Passing the natural suite is necessary but not
sufficient: the untouched Romanian law dataset is the external test.

## Run

Configure `.env` from `.env.example`, then:

```bash
./ops/memory-data.sh capability-prepare
./ops/memory-data.sh capability-rewrite
./ops/memory-data.sh capability-build
./ops/memory-data.sh capability-sync
./ops/memory-data.sh capability-run
```

`capability-prepare` downloads only the pinned public XQuAD rows and writes
versioned rewrite jobs. `capability-rewrite` calls the configured
OpenAI-compatible offline teacher and a separate verification pass; it is not
part of inference. `capability-build` refuses unvalidated or stale rewrites and
writes the ignored corpus plus SHA-256 manifests. `capability-run` synchronizes
and validates that artifact, trains on the GPU host, then runs the eval-only
causal probe and held-out evaluation. Individual train/probe/evaluate actions
remain available for diagnosis.

After a capability checkpoint passes, the untouched external dataset is run
with:

```bash
./ops/memory-data.sh index
./ops/memory-data.sh query
```

Generated corpora, checkpoints, indexes, and reports stay under `work/` and are
not versioned.

## Remaining boundary

- coherent multilingual rewrite generation, training, and evaluation have not
  completed; no natural capability pass is claimed yet;
- Python hooks recompute memory K/V and are a research implementation, not a
  throughput backend;
- exact dense shard scan must become routed hybrid retrieval at large scale;
- semantic scoring, temporal validity, deletion, adversarial prompt injection,
  ACL isolation, serving API integration, and cache observability remain open;
- the adapter is backbone-specific; the interface may port to 80B/284B models,
  but the Qwen3-4B gate weights cannot.
