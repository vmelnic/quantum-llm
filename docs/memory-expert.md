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
promoted. The `conflictqa-causal-memory-v3` run completed on 2026-08-07, but it
also failed its promotion gate. Its configured 4,096 steps were microsteps:
with batch size 2 and gradient accumulation 16, the run made only 256 optimizer
updates and consumed 8,192 of 13,267 training rows (0.62 epochs). Training took
1,608 seconds and its observed minibatch loss fell from 2.7121 to 0.0245.

On six untouched causal families, all six changed answers when authoritative
memory changed, but only 9/12 generations were exact. Two names were corrupted
and one answer followed parametric knowledge instead of the admitted evidence.
The causal channel works; the natural answer contract does not. Held-out
evaluation was deliberately not run after this prerequisite failed.

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

## Research provenance and implementation audit

The neural branch is an independent **TokenMem-style** implementation, not a
claim that this project invented gated layer-wise knowledge injection. The
reference is [TokenMem: Faithful Knowledge Injection for Frozen
LLMs](https://arxiv.org/abs/2607.22625). Its authors publish a
[research repository](https://github.com/iomgaa-ycz/TokenMem) with Qwen, Llama
and OLMo training/evaluation code. That repository currently publishes no gate
checkpoints and declares no code license, so it is a behavioral reference and
its source is not copied here.

The audit found material differences that must be resolved before another run:

- the reference trains a factual phase followed by a counterfactual phase;
  v3 mixed factual, counterfactual and abstention rows from the first step;
- the reference modifies the model implementation directly and applies RoPE
  differently on the cross-attention query; v3 used hooks over stock Qwen;
- the reference compresses memory states and trains multiple-choice CoT
  targets; v3 kept longer states and trained natural answers plus source slots;
- v3 reported microsteps as `steps`, obscuring optimizer updates and epochs;
- v3 allowed `dataset-label` evidence to pass preflight without requiring the
  exact answer in visible source text. This made the evidence validation claim
  weaker than its name suggested.

These are structural questions. The next iteration must reconcile them before
training and must not tune against the three failed examples.

No reviewed public project supplies the complete target system. The closest
components are [KBLaM](https://github.com/microsoft/KBLaM), which injects
knowledge tokens but explicitly remains a limited research system;
[MeMo](https://github.com/arunv3rma/MeMo), which trains a memory model per
corpus; and [delta-mem](https://github.com/declare-lab/delta-Mem), which stores
compact interaction history rather than a large immutable factual corpus. The
unproved integration sought here adds generic ingestion, bounded selection,
sharding, ACL/version semantics, fail-closed answers, exact authority-plane
citations, caching and serving.

## Generic capability protocol

The adapter is trained once to perform a capability—read admitted natural
memory, follow its current authority, cite it, and abstain—not to memorize a
particular ingest.

The capability corpus is constructed from the pinned Apache-2.0
[`osunlp/ConflictQA`](https://huggingface.co/datasets/osunlp/ConflictQA)
revision `056384049e63c1ddae853891c24610fa07d85744`:

- 7,940 English causal families retained from 7,947 source rows;
- 16,674 examples: original and counterfactual memories for the same natural
  question plus absent-evidence cases;
- natural answers, opaque IDs, distractor records, and absent-evidence cases;
- normalized duplicate questions are assigned atomically to one split,
  preventing train/eval leakage;
- seven ambiguous source families that retained the sibling answer were
  rejected automatically;
- the corpus builder downloads no source itself: the official Hugging Face CLI
  with `hf_xet`, a pinned revision, and a pinned file SHA-256 owns acquisition;
- a tokenizer-aware preflight requires every local source label to survive the
  configured 768-token memory bound;
- the model emits only zero-based request-local slots such as `SOURCES: 1`;
  opaque record IDs, ACL decisions, and exact quotes are mapped and rendered by
  the authority plane after generation, never learned as output tokens;
- the evaluated legal questions are forbidden training material.

The loader rejects the old patterned keys, closed-span markers, citation IDs in
questions, family split leakage, changed distractors or source positions inside
a causal pair, ambiguous sibling answers, and single-record-only corpora.

The failed v3 run mixed original, counterfactual, and abstention examples from
the first optimizer phase. This is recorded behavior, not the accepted design
for the next run. The causal probe uses eval families only. Automatic
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
./ops/memory-data.sh capability-download
./ops/memory-data.sh capability-prepare
./ops/memory-data.sh capability-sync
./ops/memory-data.sh capability-validate
./ops/memory-data.sh capability-run
```

`capability-download` uses `hf download --repo-type dataset` with Xet and pinned
revisions for ConflictQA, FaithEval, and ParaConflict. `capability-prepare`
verifies the selected ConflictQA file hash and deterministically writes the
ignored corpus plus its SHA-256 manifest. `capability-run` synchronizes and
validates that artifact, trains on the GPU host, then runs the eval-only causal
probe and held-out evaluation. `capability-validate` performs the complete
manifest, corpus, family, and anti-leak validation on the GPU host without
loading a model or changing weights. Individual train/probe/evaluate actions
remain available for diagnosis. FaithEval and ParaConflict stay untouched
external benchmarks and are not training rows.

After a capability checkpoint passes, the untouched external dataset is run
with:

```bash
./ops/memory-data.sh index
./ops/memory-data.sh query
```

Generated corpora, checkpoints, indexes, and reports stay under `work/` and are
not versioned.

## Remaining boundary

- ConflictQA v3 completed training but failed the mandatory generation gate at
  9/12 exact despite 6/6 memory-sensitive families; no natural capability pass
  is claimed and held-out evaluation was not run;
- the next run must align its mechanism, evidence contract, curriculum and
  epoch/update accounting with the public TokenMem reference before training;
- v3 initially validates the mechanism in English; Romanian and Russian remain
  external generalization work after the causal gate passes;
- Python hooks recompute memory K/V and are a research implementation, not a
  throughput backend;
- exact dense shard scan must become routed hybrid retrieval at large scale;
- semantic scoring, temporal validity, deletion, adversarial prompt injection,
  ACL isolation, serving API integration, and cache observability remain open;
- the adapter is backbone-specific; the interface may port to 80B/284B models,
  but the Qwen3-4B gate weights cannot.
