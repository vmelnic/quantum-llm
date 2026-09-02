# Research decisions

Status: canonical rejection and prerequisite ledger, 2026-09-02.

This document exists to prevent repeating failed mechanisms under new names.
Only new measured evidence that changes capacity, bandwidth, fidelity or an
acceptance inequality can reopen a rejected direction.

## Fixed targets

Qwen3.8-27B:

```text
host                 one RTX 3090
artifact             official model compiled to declared FP4 ABI
context              262,144 actually populated positions
KV                   exact IEEE F16 semantics
reasoning default    xhigh
harness              real coding agent
target               approximately 15 useful output tok/s
```

DeepSeek:

```text
host                 same self-contained machine
placement            NVMe + RAM + VRAM allowed
routing              exact router, top-k and stable aggregation
novel routes         about 1 tok/s acceptable
settled target       10-15 useful tok/s
```

## Governing capacity gate

For Qwen at 262K:

```text
hot target weights             13,625,700,352 bytes
exact F16 target KV            17,179,869,184 bytes
subtotal                       30,805,569,536 bytes
RTX 3090 physical              25,769,803,776 bytes
measured pinned PCIe link       12.46 GiB/s
full host-KV scan floor             1.284 s/call
```

Ordinary paging cannot reach 15 tok/s. A new mechanism must first prove one of:

- enough compute-local high-bandwidth capacity for complete hot state and
  reserve;
- lossless residency below the practical bit threshold;
- at least 20 useful exact accepted tokens per amortized full-KV traffic unit,
  including complete proposer/verifier cost and state.

Fail this equation before adding runtime infrastructure.

## Accepted, bounded mechanisms

### Progressive exact-F16 device mirror

Implemented for Qwen shorter prefixes. Host F16 pages remain authoritative; a
bounded exact device mirror removes unnecessary PCIe scans while populated KV
fits, then spills atomically to the exact host path. It improved the matched
Todo gate from 1,645 s to 503.240 s. It does not change the 262K inequality.

### Exact sparse expert paging

Valid for MoE because unselected experts are cold. DeepSeek exact paging makes
its 147 GB routed pool executable beyond RAM+VRAM. Demand priority, retention
classes, exact CPU/GPU split and stable merge remain useful. Novel routes still
pay storage/PCIe traffic.

### FreeToken-derived CPU/GPU split

The runtime adopted logical all-layer caching, measured CPU/GPU assignment,
concurrent exact partial execution and stable merge. Ornith can hold its entire
routed pool in the host bank; DeepSeek cannot. The first DeepSeek service gate
proved execution but did not improve short post-TTFT rate (1.29 versus
1.42 tok/s), so no paper-level throughput claim is made.

### Exact QSA tiering

Qwen Flash keeps the exact raw index on GPU and stores exact F16 QSA payload in
progressive host pages under balanced/capacity placement. Selected payload is
staged without changing the model's selected set. CUDA parity reported zero
maximum absolute difference. Short profiling showed routed MoE at 84.7% and
QSA/full attention near 1.2%, so further QSA optimization is gated by a real
populated-context trace that makes it dominant.

## Rejected mechanisms

| Direction | Measured/derived prerequisite | Decision |
|---|---|---|
| FP8 target KV | 262K prefill 1,168.594 s, decode 3.064 tok/s and factual-quality failure | rejected: wrong fidelity and too slow |
| Q4 target KV | user-confirmed quality degradation | rejected for exact-F16 target |
| dense layer streaming/AirLLM path | 12.690 GiB hot weights / 12.46 GiB/s >= 1.018 s/call before KV/compute | capacity only, not decode throughput |
| static lossless weight+KV coding | ideal total still 1,123,844,072 bytes over physical VRAM before runtime overhead | rejected |
| reversible F16 transforms | 13.265926 bits/value measured versus 11.617612 practical threshold | rejected |
| Huffman/Zstd/zlib/LZ4 | best held-out raw-F16 Huffman 13.719539 bits/value | rejected |
| exact tiered MTP tree | 4.565 accepted/cycle with proposer + transfer floor; optimistic 2.8 tok/s | rejected |
| FP4-KV block-Jacobi proposer | 3.25 accepted/cycle; corrected ceiling 1.55 tok/s | rejected |
| prompt n-gram/copy proposer | 3.8485 useful/cycle; none reached required 24 | rejected |
| pagewise exact KV refinement | one-head lower bound 2,029.014 MiB/call versus 272.822 MiB budget | rejected by 7.437x |
| staged single-GPU exact attention | 5.73516 ms/layer x 16 = 91.76256 ms before other work | cannot be the complete 15 tok/s solution |
| CPU F16 V-tail | 226.279 ms/call at 12 threads | even ideal 2-token amplification below target |
| all-K-resident/selective-V | about 25,922 MiB before V staging | exceeds physical VRAM |
| pageable-to-pinned expert bounce | extra 1.385 GB copy and worse upload wait, no cold gain | removed |
| Neural CPU replacement | failed structural/correctness boundary | experiment removed |

## Rejected DeepSeek paths

- Whole-prompt exact layer-major state required about 176 GiB for the routed
  selection streams at 1,048,576 tokens.
- Eight 131,072-token blocks reduced routed-weight traffic but still required
  about 23.516 TiB exact activation traffic.
- A real 1M-token attempt remained in block one after roughly 6.5 minutes.
- Exact two-row batching left one 131K block incomplete after more than
  14 minutes; width tuning was stopped.

`layer_major_prefill`, `causal_layer_major` and row-width tuning are not active
maximum-context solutions.

The shared-expert INT8 block-128 experiment is also rejected. One projection
improved numerically, but the runtime attempt generated incoherent text and the
artifact oracle still described the old layout. It was rolled back. Reopening
requires a complete artifact-derived CPU/CUDA hot-slot oracle before any
service edit.

## Parked model-changing directions

### Native binary/ternary models

Bonsai demonstrates trained/transformed 1.125-bit and ternary models, not a
lossless Qwen FP4-to-Q1 conversion. A binary 27B geometry could leave room for
262K exact F16 KV on 24 GiB, but reported quality is materially below FP16 and
no RTX 3090 262K exact-F16 coding gate exists. Keep it as an optional different
model with explicit quality approval, not an optimization of Qwen3.8.

### Local/fast-weight/episodic memory

Exact local attention plus gradient-free fast-weight state plus routed episodic
pages changes model semantics and requires training or substantial continued
training. Existing checkpoints cannot be losslessly converted. There is no
training budget or causal page-selection quality gate in this project, so the
hypothesis is parked and removed from the active documentation set.

### Nemotron generic FP4 conversion

The BF16 Nemotron-H generic FP4 conversion passed a weak `hi` smoke but failed a
normal instruction after correcting no-position attention. NVIDIA's deployment
recipe uses materially different NVFP4/FP8 assignments. The adapter/artifact
path was rejected; reconsider only an explicitly approved official-NVFP4
integration with its real encodings.

## Qwen Flash behavioral decision

Corrected sampling/effort propagation and a raw native-to-structured parser
trace removed the two suspected transport errors. Flash still produced zero
edits and prolonged reasoning-to-action stalls on the bounded Todo task. Do not
spend more runtime work tuning thinking or the parser as a quality fix without
new checkpoint, quantization or numerical evidence.

## Hardware boundary

Multi-GPU organ placement may become useful only after compatible hardware is
actually installed. Compute must follow state: KV beside attention compute and
expert pages beside expert compute. P100/P40 and remote owners are not present
and are not part of the active one-3090 architecture.
