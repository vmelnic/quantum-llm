# Memory Expert experiment

This directory contains two deliberately separate paths:

- `pow.py` and `synthetic_memory.py`: the historical English synthetic channel
  PoW. Its scores prove that the layer-wise channel can causally affect a frozen
  Qwen3-4B, not that it generalizes to natural data.
- `capability.py`, `capability_corpus.py`, and
  `prepare_capability_data.py`: the current pinned ConflictQA English
  capability gate.

The old procedural closed-vocabulary corpus was removed after audit. It must
not be recreated as a fallback.

Capability training is epoch-based. Every training row is visited exactly once
per deterministic epoch; microsteps and optimizer updates are derived values,
not user-selected aliases. Validation NLL, best/last checkpoints, and an
optimizer-bearing resume state are written at each epoch boundary. TokenMem is
a documented Plan B, not the active implementation path.

## Current data flow

```text
pinned ConflictQA JSONL from Hugging Face/Xet
  └─► source SHA + causal ambiguity validation
       └─► natural original/counterfactual/unknown JSONL
       └─► strict corpus + anti-leak validation
            └─► frozen Qwen memory-pass layer states
                 └─► bounded lazy CPU LRU
                      └─► frozen layer Q/K/V/O + trained low-rank gate
                           └─► eval-only causal and held-out gates
```

There is no locally generated counterfactual fallback or teacher path. The
neural target contains request-local `SOURCES` slots, while durable IDs and
exact quotes stay in the authority plane. `real_query.py` connects a checkpoint
that has passed the capability gate to generic ingested records. Admission,
strict literal matching, and semantic correctness are separate report concepts;
the code does not claim an automatic semantic score.

Run through `ops/memory-data.sh`; see
[the public design](../../docs/memory-expert.md) for commands, invariants, and
the current result status.
