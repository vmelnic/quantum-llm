# External Memory Expert

## Result

The bounded proof of concept demonstrates that a frozen language model can
answer from newly attached records without placing those records in the
question prompt and without retraining for each ingest.

The validated configuration uses the official Qwen3-4B BF16 checkpoint, keeps
all backbone parameters frozen, and trains 2,949,120 low-rank gate parameters.
On the deterministic held-out suite it achieved:

| Mode | Joint exact answer + citation | Citation accuracy | Retrieval recall |
|---|---:|---:|---:|
| no memory | 18.75% | 18.75% | n/a |
| oracle memory | 100% | 100% | n/a |
| automatic retrieval | 93.75% | 100% | 100% |
| full-context control | 62.50% | 81.25% | n/a |

The strict full-context score is not a general claim that the adapter is better
than prompting. The control often contained the right facts but used prose or
citation ordering rejected by the exact evaluator. Its purpose is diagnostic.

The causal intervention gate also passed:

- three question families, each with four contradictory memories;
- the question and visible citation IDs remain identical inside a family;
- target preference was correct for 12/12 memory interventions;
- generated answer, exact literals, and citations were correct for 12/12;
- changing only the admitted memory changed the answer in every case.

This establishes a causal external-memory channel. It is not yet a production
knowledge service or evidence that arbitrary real corpora will reach the same
accuracy.

## Architecture

```text
                               infrequent ingest
authoritative source ─► stable record + ACL + generation ─► sharded index
                                                           │
                                      query-time top-k/hops │
                                                           ▼
                                            admitted raw records
                                                   │
                                      separate frozen model pass
                                                   │
                              layer input states + exact source tokens
                                                   │
question-only prompt ─► frozen decoder blocks ─────┤
                          │                        │
                          └► frozen Q/K/V/O cross-attention at every layer
                                      │
                              trained low-rank residual gates
                                      │
                         semantic answer and citation selection
                                      │
                exact extractive continuation + authorization check
                                      ▼
                         answer + verbatim cited records
```

For decoder layer `i`, the memory pass supplies `hidden_states[i]`, the input
to that same layer. The cross-attention branch reuses its frozen RMSNorm and
Q/K/V/O projections. Only a rank-16 residual fusion is trained. Its output
matrix is zero-initialized, so the attached branch is an exact identity before
training.

The extractive continuation path solves a distinct problem. Continuous hidden
states can select the correct code or quotation while reconstructing a digit
incorrectly. Once the decoder begins a meaningful span that occurs in admitted
memory, the pointer favors its exact next source token. It cannot introduce
tokens from a non-admitted record. Citation quotes are rendered from the
authoritative record, never regenerated from model text.

The implementation is informed by the separate-passage, layer-wise memory
topology described by [TokenMem](https://arxiv.org/abs/2607.22625) and the
public [DecoupledRAG](https://github.com/Deriq-Qian-Dong/DecoupledRAG) source,
but the sharded ingest contract, causal gate, abstention policy, and exact
continuation path are local design work.

## Ingest does not retrain the model

Training teaches the adapter how to use the memory channel. New content follows
a data path, not an optimization path:

1. preserve authoritative bytes and assign stable record metadata;
2. chunk/index the record and commit its generation atomically;
3. retrieve a bounded set for a request;
4. encode only those admitted records through the frozen memory pass;
5. reuse or discard their bounded hot representation according to cache policy.

A new law, order, user record, or book does not update model weights. Additional
training is justified only when the required capability changes—for example a
new language, document structure, reasoning operation, or compliance policy.

Corpus-wide layer states are forbidden: hundreds of gigabytes of source must
not become hundreds of gigabytes multiplied by model layers. Cold storage keeps
raw records and indexes, warm storage keeps compact embeddings/metadata, and
only observed hot records may hold bounded JIT model states.

## Retrieval and abstention

The PoW index is deliberately replaceable. It writes bounded shards, can open
small indexes resident or large indexes one memory-mapped shard at a time, and
retains only bounded top-k candidates. A root hit may trigger one bounded
lexical/entity expansion for a dependent record.

The synthetic calibration measured a minimum known root score of 0.310 and a
maximum unknown score of 0.187. The PoW therefore uses a 0.25 admission floor;
the final unknown queries admitted no irrelevant records and abstained 3/3.
This number is not portable to real embeddings or corpora. Production requires
held-out calibration per encoder/domain, score margins, typed links, ACL checks,
and an explicit no-evidence outcome.

## Reproduce

Configure the control host from `.env.example`, then run:

```bash
./ops/memory-pow.sh download
./ops/memory-pow.sh download-status
./ops/memory-pow.sh run
```

`run` synchronizes sources, validates the isolated environment and model
snapshot, runs structural/retrieval self-tests, trains, requires the causal
probe to pass, and finally evaluates held-out examples. Generated checkpoints,
indexes, and JSON reports remain under `work/memory-expert-pow` on the GPU host
and are intentionally ignored by Git.

The validated run used 1,024 microsteps, batch size 2, accumulation 16, AdamW
at 2e-4, rank 16, alpha 32, and 0.2 knowledge dropout. Training took about 306
seconds and peaked at 10.85 GiB allocated VRAM. These are reproduction values,
not production sizing guarantees.

Individual phases are available for diagnosis:

```bash
./ops/memory-pow.sh train
./ops/memory-pow.sh probe
./ops/memory-pow.sh evaluate
./ops/memory-pow.sh status
```

## Known limits and next boundary

- the corpus is synthetic, English, small, and structurally regular;
- the synthetic PoW still uses its hashing test double; the real-data path now
  uses BGE-M3, although its exact dense shard scan is not yet a
  hundred-gigabyte retrieval backend;
- memory encoding is JIT and not yet exposed through the serving API;
- the Python hooks recompute cross-attention K/V during decode and are not a
  throughput implementation;
- the exact pointer is a validated continuation mechanism, not a general
  learned span/copy head;
- the automatic mode missed one answer when an extra retrieved record acted as
  a distractor, despite retrieving the correct evidence;
- ACL enforcement, immutable provenance, updates/deletes, temporal validity,
  prompt-injection resistance, batching, cache admission, and observability
  require production implementation and adversarial evaluation;
- adapters are backbone-specific; the interface can port to Qwen3-Next and
  DeepSeek, but these Qwen3-4B gate weights cannot.

The ingestion and retrieval portion of the real-data slice is now implemented and
measured in [Memory Data ingestion and retrieval](memory-data.md). BGE-M3 found
the correct Romanian legal article at rank 1 for 5/5 questions, but the existing
English synthetic adapter answered only 2/5 strictly. The next model-side gate
is therefore one corpus-independent multilingual/extractive capability adapter,
not per-document training.
