# Research decisions

Status: canonical negative-results and active heterogeneous-placement record,
2026-08-23. Rejected implementations and raw generated artifacts were removed
from the production repository so they cannot be mistaken for supported paths.

## Fixed acceptance contracts

Qwen target:

- official Qwen3.8-27B artifact;
- FP4 E2M1/UE8M0 matrices executed by the declared ABI;
- 262,144 actually populated positions;
- exact F16 target KV semantics;
- default `xhigh` reasoning;
- at least approximately 15 useful output tok/s on a real coding harness.

DeepSeek target:

- exact router and top-k;
- model may exceed RAM+VRAM but fit VRAM+RAM+NVMe;
- self-contained host, no remote expert owner;
- about 1 tok/s is acceptable on novel routes and 10-15 tok/s is the useful
  settled target.

## Capacity and bandwidth gate

For Qwen at 262K:

```text
target-call weights       13,625,700,352 bytes
exact F16 target KV       17,179,869,184 bytes
subtotal                  30,805,569,536 bytes
RTX 3090 physical         25,769,803,776 bytes
```

The subtotal already exceeds VRAM by about 5.04 GB before recurrent state,
MTP, workspaces and reserve. Exact KV in host RAM incurs an approximately
1.284-second PCIe scan per scalar call at the measured 12.46 GiB/s. A valid
15 tok/s design must either amortize one exact scan over many accepted tokens
or add local high-bandwidth memory with compute colocated beside each shard.

## Rejected directions

| Direction | Quantitative result | Decision |
|---|---|---|
| FP8 target KV | full 262K fit; 3.064 tok/s; direct factual-quality failure | rejected: wrong fidelity and too slow |
| ordinary Q4 KV | user-confirmed degradation | rejected: violates exact F16 |
| AirLLM-style dense layer streaming | 12.689922 GiB target weights / 12.46 GiB/s = at least 1.018 s/call before KV/compute | capacity mechanism only, not decode solution |
| static lossless weight+KV coding | ideal weights + ideal KV + recurrent state = 26,893,647,848 bytes, 1,123,844,072 bytes over 24 GiB before codec/runtime overhead | rejected |
| measured reversible F16 KV transforms | 13.265926 bits/value; practical residency threshold 11.617612 | rejected |
| Huffman/Zstd/zlib/LZ4 family | best held-out raw-F16 Huffman 13.719539 bits/value | rejected |
| exact tiered MTP tree | 4.565 mean accepted tokens/cycle; 0.667 s proposer + 0.963 s transfer floor; optimistic 2.8 tok/s | rejected |
| FP4-KV block-Jacobi proposer | 3.25 accepted/cycle; corrected necessary ceiling 1.55 tok/s | rejected |
| prompt n-gram proposer | 3.8485 useful tokens/cycle, no cycle reached required 24 | rejected |
| pagewise exact KV refinement | one-head lower bound 2,029.014 MiB/call versus 272.822 MiB budget at 15 calls/s | rejected by 7.437x |
| staged single-GPU F16 attention tuning | 5.73516 ms/layer; 16 layers = 91.76256 ms, at most 10.8977 attention calls/s before all other work | no more tuning of that provider as the complete solution |
| CPU F16 V-tail split | 226.279 ms/call at 12 threads; even perfect 2-token amplification caps at 8.84 tok/s | rejected |
| all-K-resident/selective-V | estimated 25,922 MiB before V staging, beyond 24,576 MiB physical | rejected on capacity |
| Neural CPU replacement | failed its pre-registered structural/correctness boundary and had no path to exact transformer semantics | experiment removed |

These results remain in this document specifically to prevent repeating the
same mechanism under a new name.

## Why the architecture is still useful

The dense Qwen and sparse DeepSeek problems are different:

- Qwen's dense weights and all full-attention KV are hot. Demand paging does
  not create sparsity.
- DeepSeek's unselected experts are genuinely cold. Demand paging is valid and
  already makes a 147 GB routed pack executable on the self-contained host.

The artifact VM, exact page identity, tier telemetry and provider boundary are
therefore useful infrastructure. The rejected part is the assumption that the
same single-GPU placement can also deliver exact 262K Qwen at 15 tok/s.

## Active multi-GPU direction: compute follows state

The offered rig provides three Tesla P100 PCIe 16 GB cards and one Tesla P40
24 GB in addition to the existing RTX 3090. The P100 has 732 GB/s HBM2 and
strong native FP16, but is SM60 and cannot execute the current SM86 FP4
provider. Its useful Qwen role is exact F16 KV plus attention, not generic FP4
weights.

Proposed Qwen placement:

```text
RTX 3090
  FP4 weights, recurrent layers, vocabulary head
       | Q/K/V activations
       v
P100 #0      P100 #1      P100 #2
KV-head      KV-head      KV-head
shard +      shard +      shard +
attention    attention    attention
       | small per-head attention outputs
       +---------------> RTX 3090 continues the layer
```

At 262K, three P100s hold about 5.73 GB of F16 target KV each. Ideal
head-parallel memory floors are:

```text
RTX 3090 weights: 13.626 GB / 936 GB/s = 14.6 ms/call
each P100 KV:       5.727 GB / 732 GB/s =  7.8 ms/call
combined ideal floor                         22.4 ms/call
```

This is not a prediction. Kernel efficiency, per-layer dependency, PCIe/PLX
latency, P2P availability, reductions and recurrent work are omitted. The
fail-fast gate is the exact production attention operation:

- below roughly 150-200 GB/s effective per P100 including orchestration: stop;
- above roughly 200 GB/s: a 15 tok/s design remains credible;
- around or above 400 GB/s with acceptable P2P latency: evaluate the 30 tok/s
  path.

For DeepSeek, P100 lacks the current packed DP4A execution path. The P40 is
SM61 and is the more plausible Pascal expert-page executor, but it still needs
an independently qualified provider. The RTX 3090 remains the primary model
GPU. No Pascal card is part of the current implementation claim.

## Why this is not ordinary tensor or layer parallelism

Whole-layer pipeline placement makes the slow GPU's time additive and moves
activations at every boundary. The proposed split follows the model organ:

- Qwen KV heads remain permanently beside the GPU that computes their
  attention;
- DeepSeek expert pages remain beside the GPU that executes them;
- only hidden/QKV/output activations and explicit reductions cross PCIe;
- placement is declared by tensor role and shard semantics, not hardcoded
  layer ranges.

The mechanism must remain optional and capability-driven so the single-3090
path and future GPUs use the same artifact and service.
