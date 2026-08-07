# Memory Expert experiment

This directory contains two deliberately separate paths:

- `pow.py` and `synthetic_memory.py`: the historical English synthetic channel
  PoW. Its scores prove that the layer-wise channel can causally affect a frozen
  Qwen3-4B, not that it generalizes to natural data.
- `capability.py`, `capability_corpus.py`, and
  `prepare_capability_data.py`: the current natural Romanian/Russian/English
  capability gate.

The old procedural closed-vocabulary corpus was removed after audit. It must
not be recreated as a fallback.

## Current data flow

```text
pinned XQuAD rows
  └─► versioned counterfactual rewrite jobs
       └─► teacher rewrite + independent extraction/coherence validation
            └─► natural original/counterfactual/unknown JSONL
       └─► strict corpus + anti-leak validation
            └─► frozen Qwen memory-pass layer states
                 └─► bounded lazy CPU LRU
                      └─► frozen layer Q/K/V/O + trained low-rank gate
                           └─► eval-only causal and held-out gates
```

There is deliberately no blind span-substitution fallback. The neural target
contains request-local `SOURCES` slots, while durable IDs and exact quotes stay
in the authority plane. `real_query.py` connects a checkpoint that has passed
the capability gate to generic ingested records. Admission, strict literal
matching, and semantic correctness are separate report concepts; the code does
not claim an automatic semantic score.

Run through `ops/memory-data.sh`; see
[the public design](../../docs/memory-expert.md) for commands, invariants, and
the current result status.
