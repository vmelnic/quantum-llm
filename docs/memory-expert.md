# External Memory Expert

## Current status

The project has proved that a frozen Qwen3-4B can receive information through
a separate layer-wise memory channel. It has also disproved the current gate's
ability to generalize reliably beyond ConflictQA, in both Romanian legal data
and an invented English dossier. The checkpoint is diagnostic and must not be
promoted.

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

The v4 training path fixed that scheduling defect without changing the memory
architecture. It completed three deterministic epochs: 39,801 example visits,
19,902 microsteps and 1,245 optimizer updates. On its ConflictQA evaluation,
oracle memory produced 28/30 strict answers and 30/30 correct source selections;
no-memory produced 6/30 strict answers. Unknown-memory abstention was 100%.
Automatic retrieval recovered the authoritative record in only 3/10 paired
cases, but that arm used the placeholder hashing index rather than the BGE-M3
serving index.

The 30 examples were not a truly untouched test: the best of three epoch
checkpoints was selected by validation NLL on a deterministic 256-example prefix
from the same eval split, and the reported 30 are a prefix of that selection.
No gradients used them, but the result has checkpoint-selection bias and is now
described as in-distribution evaluation rather than independent held-out proof.

The decisive external test used five Romanian questions about an ingested
criminal code. Retrieval recovered the intended evidence in 5/5 cases, but
the v4 adapter answered only 1/5 correctly under both strict and manual review.
The four failures included ignoring an explicit quote, incomplete extraction,
malformed source output and token degeneration. This is a model/adaptation
failure after correct evidence admission, not a retrieval failure.

An English out-of-distribution control then removed language and legal register
as explanations. A new five-section fictional dossier contained names, dates,
measurements and an exact quotation that could not exist in pretraining. BGE-M3
retrieved the intended section for 8/8 questions. The default Memory Expert
serving format answered only 2/8 and emitted a valid source slot for 1/8. An
inference-only ablation removed the durable ID, title, section metadata and slug,
feeding the same bare record-body format used in training. Source binding rose
to 8/8, but strict answers rose only to 3/8. The same frozen Qwen3-4B, with all
five records in its prompt and hooks disabled, answered and cited 8/8.

This matrix closes the v4 direction: metadata formatting caused a real source-
slot mismatch, but does not explain factual failures such as changing 19 hours
to 29.82 minutes, 27 minutes to two hours, or 14 October 2037 to 24 October
2017. V4 learned a ConflictQA-specific latent reader, not the generic memory
capability required by the project. It will not receive another training run.

## Mechanism probes and the v5 pointer run

A series of inference-only instruments (`mechanism_probe.py`,
`mechanism_trace.py`, `mechanism_intervene.py`, `mechanism_span_rank.py`;
artifacts under `work/memory-mechanism-*`) established why v4 fails:

- the gate does not shut down out of distribution: residual amplitude on the
  fictional dossier is ~78% of the in-distribution level, and failed and
  correct cases are internally indistinguishable;
- token-level traces show the gold literal reaches the final logits at ranks
  2–100 (one failure lost on an exact logit tie), but the signal is not
  separable: scaling the gate residual by 1.5–3 degrades both out-of- and
  in-distribution accuracy monotonically into degeneration;
- the masked null key is irrelevant to the failure: unmasking it at inference
  changes nothing;
- an NLL span-ranking test showed weakly positive selection: the gold answer
  ranked first in 5/8 cases versus 3/8 with empty memory, with only one
  dramatic memory-driven swing. Reading out of distribution is marginal, not
  absent.

V5 tested the resulting hypothesis directly: if the channel can select but
cannot render, train pointer targets (`ANSWER: @slot:sentence`) and let the
authority plane render the verbatim sentence. The corpus became
`conflictqa-causal-memory-v4` (12,664 rows; 1,912 of 7,940 families rejected
because no confident gold sentence exists). Training completed three epochs
(30,153 visits, 945 optimizer updates, best validation NLL 0.0147). The causal
probe was adapted for the pointer contract — contrastive NLL on coordinates is
degenerate because siblings share them — and passed semantically: 6/6
evaluation families rendered different, memory-tracking content. Held-out
evaluation: 28/30 pointer-exact oracle answers, 30/30 sources, 100% unknown
abstention.

The external gate then decided. On the fictional English dossier the v5
checkpoint passed only 2/8: pointers landed on neighbouring sentences, two
pointers were malformed, one case abstained. On the Romanian criminal code it
abstained on all five questions (retrieval again 5/5). Both results miss the
pre-registered bar (≥7/8 and ≥4/5), so v5 is closed. Its one product-relevant
property is that failures are fail-closed abstentions or verbatim quotes from
the admitted record, never invented literals.

The combined conclusion: a rank-16 residual gate trained on a single English
corpus learns a ConflictQA-specific reader. It cannot render literals (v4) and
cannot select them finely enough out of distribution (v5). Both external
verdicts converge on the same boundary: the trained capability is bound to the
training distribution, not to "reading admitted memory" in general.

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

## What is actually trained

The model is not trained to memorize ConflictQA and it is not retrained for
each new law or database. Qwen already knows how to read and answer. We train a
small set of gates to make Qwen consult a second, non-prompt memory channel.

```text
                         frozen forever
question ───────────────► Qwen3-4B ───────────────► answer
                              ▲
                              │ correction at each decoder layer
                              │
admitted record ─► frozen Q/K/V/O attention ─► trainable rank-16 gate
                                                 2.95M parameters
```

A training pair is conceptually:

```text
same question:  "Who wrote the work?"

memory A:       "The author is Ana."  ─► expected answer: Ana
memory B:       "The author is Boris." ─► expected answer: Boris
no evidence:    unrelated record       ─► expected answer: I don't know
```

Because the question remains the same while authoritative memory changes, the
small gate cannot solve the pair reliably by memorizing the question alone. It
must learn three operations: use admitted memory, prefer it over conflicting
parametric knowledge, and abstain when it has no support.

After that capability is learned, ingest is only data work:

```text
new law/book/order ─► immutable chunks + index ─► admitted records at query time
                                                   │
                                                   └─► same trained gate
                                                       no weight update
```

This is not a prompt-RAG generation path: source text does not consume the
conversation context or causal KV cache. A bounded selector admits records
because hundreds of gigabytes cannot be activated for every question, but the
admitted records reach the decoder only through the separate layer-wise memory
channel. The capability corpus below trains that channel; it is not the user's
knowledge base and it is not rebuilt when knowledge is ingested.

## Research provenance and Plan B

The neural branch is an independent **TokenMem-style** implementation, not a
claim that this project invented gated layer-wise knowledge injection. The
reference is [TokenMem: Faithful Knowledge Injection for Frozen
LLMs](https://arxiv.org/abs/2607.22625). Its authors publish a
[research repository](https://github.com/iomgaa-ycz/TokenMem) with Qwen, Llama
and OLMo training/evaluation code. That repository currently publishes no gate
checkpoints and declares no code license, so it is a behavioral reference and
its source is not copied here.

The audit found material differences from TokenMem:

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

Those differences are research hypotheses, not evidence that every
layer-wise-memory architecture is invalid. Full-epoch training isolated the
known scheduling defect but did not close the external generalization gap.
Further tuning of this checkpoint against the five legal questions is stopped:
it would turn the external test into training data. The next baseline is the
official Doc-to-LoRA Qwen3-4B checkpoint, which generates temporary LoRA
weights from a document without per-document gradient descent. TokenMem remains
a separate Plan B rather than an unlicensed code dependency.

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

The v3 run mixed original, counterfactual, and abstention examples from the
first optimizer phase. V4 deliberately preserves that joint-grounding
curriculum so the schedule fix is tested without an architectural or dataset
change. The causal probe uses eval families only. Automatic
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

TODO: replace the normalized-substring answer matcher with a generic,
multilingual semantic scorer validated against labeled positive and negative
pairs. The current matcher can reject correct shorter formulations, so its
strict score remains diagnostic and must be reported separately from manual
semantic review; do not add answer-specific aliases or dataset-specific rules.

The external English failures above were also manually inspected. Their large
numeric substitutions, refusals and invented locations are semantic failures,
not matcher artifacts.

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

The frozen-base full-context ceiling for any ingested dataset is available as:

```bash
./ops/memory-data.sh control
```

`MEMORY_RECORD_FORMAT=raw` performs the inference-only canonical-format
ablation. It must not be mistaken for a new trained checkpoint.

Generated corpora, checkpoints, indexes, and reports stay under `work/` and are
not versioned.

## Remaining boundary

- ConflictQA v4 completed full-epoch training and reached 28/30 oracle answers,
  but its test subset participated in checkpoint selection; it then failed the
  Romanian external test at 1/5 despite 5/5 evidence retrieval;
- the invented English control failed at 2/8 with serving metadata and 3/8 with
  canonical raw records, while the frozen full-context control passed 8/8;
- mechanism probes closed the v4 hypotheses: no gate suppression, no null-key
  effect, no amplifiable signal; literal transport is lossy and entangled;
- v5 (pointer targets plus authority-plane verbatim rendering) trained cleanly
  and passed the semantic causal probe and held-out suite (28/30 oracle), but
  failed the external bar: 2/8 on the English dossier, 0/5 abstentions on the
  Romanian code; neither v3, v4 nor v5 is promotable and no further training
  of this gate-on-ConflictQA design is planned;
- the answer matcher and generic multilingual paraphrase scoring remain a
  documented TODO, but cannot explain the manually verified legal failures;
- Romanian and Russian capability remains unproved;
- Python hooks recompute memory K/V and are a research implementation, not a
  throughput backend;
- exact dense shard scan must become routed hybrid retrieval at large scale;
- semantic scoring, temporal validity, deletion, adversarial prompt injection,
  ACL isolation, serving API integration, and cache observability remain open;
- the adapter is backbone-specific; the interface may port to 80B/284B models,
  but the Qwen3-4B gate weights cannot.
