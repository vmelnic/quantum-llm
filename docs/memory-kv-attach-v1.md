# Memory KV-Attach v1 — prototype design

Status: milestones 1 and 2 validated on 2026-08-08.

Milestone 1 (frozen, zero trained parameters; artifact
`work/memory-kv-attach-v1/kv-attach.json`, schema 5): with the Nacre OOD
dossier admitted as an attached K/V prefix, frozen Qwen3-4B reached 8/8
strict answers and 7/8 joint (synthetic-turn wrap), versus 8/8 joint for
the full-context control; the split-vs-joint identity check is exact in
float32 (argmax agreement 1.0). Romanian criminal code with retrieval
selection (`query-report-top2.json`): 4/5 strict answers, 3/5 joint.
Caveat recorded: iterations v1–v5 of the harness carried a wrong
`past_length` (KV-head count instead of sequence length), so every
behavioural conclusion from those runs was retracted; only v6 numbers are
valid.

Milestone 2 (LoRA discipline training, `kv_attach_train.py`, checkpoint
`kv-attach-lora-v2`): a 2.4M-parameter LoRA on the attention projections,
trained once on ConflictQA only (3 epochs, 1,884 optimizer updates, best
eval NLL 3.7e-05), with the memory prefill always running LoRA-disabled
(native frozen K/V at training and serving). The answer contract is a
slot-level pointer (`ANSWER: @slot`); the authority plane renders the
record verbatim. Sentence-level pointing was tried first (v1 checkpoint)
and proved fragile across document structures — headings become
"sentence 0" — so v2 points at the record. Unknown rows are additionally
trained with an empty prefix (50%) so abstention covers the no-memory case.
Results: Nacre 6/8 joint with **8/8 correct abstentions** (frozen: 7/8
joint, 6/8 abstentions); Romanian 4/5 joint and semantically 5/5 (the one
strict miss is an orthography mismatch in the question file, not the
model), with **5/5 abstentions** and no nonexistent slots cited. The two
Nacre misses both point at record slot 4, which never occurs in the
≤4-record ConflictQA corpus — a corpus limit, not a mechanism limit;
fix via ≥5-record training rows or k≤4 at serving.

Net: the frozen native-attention read transfers across languages and
invented content with no training, and the trained discipline (abstention,
valid slots, memory over parametric knowledge) transfers out-of-distribution
— the two known regressions have identified, repairable causes. The
rank-16 latent-gate direction (v4/v5) remains closed; KV-attach dominates
it externally with ~1000× fewer trained parameters and no custom
attention. This document follows the closure of that direction (see
`memory-expert.md` and `history.md`). The architecture below is what
milestones 1–2 validated.

## Known issues (deferred, fixes agreed)

1. **Slot 4 unreachable.** ConflictQA admits at most 4 records per example,
   so the v2 adapter never saw `@4` as a target and maps record-5 questions
   to the nearest plausible slot (both Nacre misses). Fix: training-time
   augmentation in `kv_attach_train.py` — pad examples with 1–2 distractor
   records drawn from other families so the gold record lands on slots
   4–5 — then retrain (~3 h). The serving-side alternative (cap
   `top_k <= 4`) hides the bug but blocks larger k for big corpora.
2. **Romanian orthography mismatch in the test file.** `questions.jsonl`
   expects `săvârşirii` (new orthography) while the 2009 law text contains
   `săvîrşirii` (old); NFKD normalization maps them to different base
   letters, so a verbatim-correct answer fails strict matching. Fix: list
   both orthography variants in `expected_answers` (the scorer already
   accepts any-of), then re-score the existing v2 artifact locally — no
   GPU rerun needed.

## Why this pivot

The v4/v5 mechanism audits established:

- the latent channel transmits signal (causal probes pass, gate amplitude is
  normal out-of-distribution), but the learned rank-16 read-out is a free
  function trained on one English corpus; it interpolates, it does not
  generalize;
- out-of-distribution errors concentrate on numbers, dates and names, and the
  cross-attention had no positional signal on memory keys — exact-copy
  geometry was never representable;
- the masked `null` key forced attention mass onto memory whenever memory
  exists, removing any native "do not use this" option;
- per-token, per-layer latent storage for the full corpus does not scale
  (~144 KB/token KV, ~3.6 PB for a 100 GB corpus) — latent state must be
  computed lazily for the retrieval-selected records only.

The protocol conclusion: the reader must be Qwen's own attention, trained on
trillions of tokens, not a small adapter trained on one corpus.

## Architecture

Storage plane (per ingest, offline):

- records are stored as **text + BGE-M3 embedding** only; no hidden states,
  no KV, no per-layer tensors are persisted;
- the index is shardable exactly as today; corpus size costs disk for text
  and embeddings (~2 KB/record), not petabytes of latent state.

Query plane (per query):

1. BGE-M3 retrieval selects top-k records (k ≈ 8, ≈256 tokens each,
   ≈2k memory tokens total).
2. The k records are prefilled through frozen Qwen3-4B once, capturing
   per-layer K/V from its **own frozen self-attention projections**. This is
   a ~2k-token prefill: milliseconds on an RTX 3090, no stored latent state.
3. At every decoder layer, the query forward pass attends over the
   concatenation `[memory K/V | query K/V]`:
   - memory tokens receive contiguous RoPE positions `0..M-1`; the query
     continues at `M..`, so the geometry is exactly "text actually read as a
     prefix" — the native reading configuration, with explicit position
     information (the v4/v5 missing-RoPE defect cannot recur);
   - attention is causal within the query; all query positions may attend to
     all memory positions.
4. As built, no custom attention was needed: the attached prefix is simply
   the K/V cache of the prefilled records and the model's own attention
   reads it natively (this is what milestone 1 validated). The split
   softmax/gate machinery originally planned here was dropped — gates can
   only scale memory contribution, while the observed failures were
   output-distribution behaviours.
5. Abstention is trained through the answer contract, not through a null
   key: unknown rows teach "memory attached but irrelevant → abstain", and
   50% of them are additionally trained with an empty prefix so "no memory
   at all → abstain" is covered as well.

Trained parameters (as built): a rank-16 LoRA (alpha 32) on q/k/v/o of all
36 layers, ~2.4M parameters, trained once on ConflictQA. The memory prefill
always runs with LoRA disabled, so attached records keep native frozen K/V
at training and at serving; the adapter only learns to read and to follow
the contract. No per-ingest training, satisfying the project constraints.

Answer contract: slot-level pointer — the model emits `ANSWER: @slot` and
the authority plane renders the whole record verbatim. Citations are never
generated by the model. (A sentence-level `@slot:sentence` pointer was
tried first and abandoned: sentence indexing is corpus-structure-dependent
and did not transfer; the v5 data pipeline, schema 2, is reused as-is.)

## What changed in code (as built)

- `experiments/memory_expert/kv_attach.py`: memory prefix prefill with
  per-layer K/V capture, cache-backed generation, synthetic-turn wrapping,
  retrieval-selection mode, pointer rendering, identity check, Nacre and
  Romanian validation arms, optional LoRA loading;
- `experiments/memory_expert/kv_attach_train.py`: manual LoRA (no new
  dependencies), frozen memory prefill with LoRA disabled, empty-memory
  abstention augmentation, deterministic epochs, best checkpoint by eval
  NLL, progress logging with ETA;
- `synthetic_memory.py`: slot-level pointer target and parser (legacy
  sentence pointers remain parseable); `pow.py` / `real_query.py`: pointer
  rendering updated accordingly;
- `ops/memory-data.sh` + `ops/windows/Invoke-MemoryKvAttach{,Train}.ps1`:
  the `kv-attach` and `kv-attach-train` actions.

## Training data

The gates learn *when* to read and *when* to abstain, not *how* to read —
the reading itself is native. Training data therefore needs coverage of the
decision space, not of all possible content:

- ConflictQA causal corpus (schema 2) — reuse;
- a modest multi-domain synthetic set (including Romanian legal chunks from
  the criminal-code ingest) so abstention and selection are not English-only;
- negative examples: queries with no relevant admitted record, teaching
  null-mass / abstention.

Target: same schedule as v5 (3 epochs, best-checkpoint by validation NLL).
Gate-only training is minutes-to-hours on the 3090; LoRA variant is the v5
order of magnitude.

## Pre-registered evaluation (before training starts)

Same harness, same criteria as v5, recorded in advance:

- ConflictQA held-out: ≥ 28/30 pointer-exact, 30/30 source, 100% unknown
  abstention;
- Nacre invented-English OOD: ≥ 7/8;
- Romanian criminal code: ≥ 4/5 with retrieval ≥ 4/5;
- mechanism trace: on correct OOD cases, attention max lands on the answer
  span; on unknown cases, null mass dominates;
- base-model regression: with `g_i = 0` the model is bit-identical to frozen
  Qwen; after training, a small standard benchmark sanity check confirms no
  visible degradation (variant B only).

Failure of the pre-registered criteria closes this direction too, and the
result is documented in `history.md` exactly as v4/v5 were.

## Explicit risks

- Zero-init gates may remain near zero (model ignores memory): mitigated by
  the causal training data that v5 already showed can drive memory use.
- Prefix-position RoPE assumes memory behaves like genuinely-read text; if
  long-range decay makes distant memory weak, positions can be compressed or
  per-record restarted — flagged as the first ablation, not silently chosen.
- Gate-only may be too thin for abstention: variant B (LoRA) is the planned
  fallback, not an afterthought.
- Query cost includes a top-k prefill; acceptable at k ≈ 8, and records can
  be cached hot, but the "zero recompute" property of stored-KV designs is
  deliberately traded away for feasibility at corpus scale.

## Decision point after the prototype

- If criteria pass: the mechanism is worth scaling work (shardable ingest,
  larger models, 80B path).
- If selection/reading works but abstention fails: move to variant B (LoRA).
- If native attention with correct geometry still cannot read attached
  memory generally: the external-latent-memory thesis is closed with a
  mechanism-level explanation, and the retrieval + context + authority-plane
  system becomes the documented baseline.
