# Exact tiered tree verification for 262K context

Status: rejected at prerequisite gates, 2026-08-21. MTP self-rollout,
full-context-FP4 block-Jacobi, prompt-copy continuation, static lossless
residency, staged resident attention, and exact CPU/GPU attention splitting
all fail measured acceptance, capacity, or throughput inequalities. None may
be promoted into the service behind these failed prerequisites.

This document replaces ordinary KV offload as the active maximum-context
decode direction. The target is Qwen3.8-27B on one RTX 3090 with 262,144
actually populated tokens, exact F16 KV semantics, and approximately 15
generated tokens/s on the real coding harness. It must remain artifact-driven
and usable by other architectures through declared capabilities rather than a
Qwen branch.

## Immutable acceptance contract

- The target model remains the published Qwen3.8-27B FP4 artifact. An auxiliary
  proposer may be approximate because it cannot emit a token.
- The authoritative target KV is F16. No Q4/FP8 KV, permanent token eviction,
  reduced context, summarization, or approximate attention may decide output.
- Every emitted token is accepted by the exact target verifier. Greedy output
  must match serial target decoding; sampled output must use rejection sampling
  that preserves the target distribution.
- The context must be physically populated. A 262K admission limit or sparse
  placeholder is not evidence.
- The complete real harness path is the release gate. Tiny direct prompts are
  diagnostic only.
- Cold prefill, reused-prefix prefill, and decode are reported separately.

The official model card declares 64 text layers arranged as 48 Gated DeltaNet
layers and 16 full-attention layers, with four KV heads of dimension 256 and a
native 262,144-token context. These values explain the current Qwen geometry;
runtime code must consume equivalent roles and dimensions from the artifact.
See the [official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B).

## Failures this direction must not repeat

| Failed direction | Evidence | Consequence |
|---|---|---|
| Fit the GGUF weights and full F16 KV in 24 GiB | About 16.35 GiB of weights plus 16 GiB KV, before MTP and buffers | CPU placement was forced; this was not full-GPU 262K serving. |
| Stream the full KV for every token | 16 GiB / 12.46 GiB/s = 1.284 s before compute | At most 0.78 target calls/s; paging restores capacity but destroys throughput. |
| Quantize the authoritative KV | Quality degradation was already observed | Invalid for the exact target. |
| Replace the server/runtime | The real 26,193-token Claude request still took about 88.4 s and decoded at 2.16 tok/s | Wrappers do not change the hot bytes. |
| Treat short-prompt decode as harness performance | A 57-token direct request reached 6.28 tok/s while the real harness did not | Prompt size and populated KV must accompany every rate. |
| Treat NVMe as the decode fix | Moving storage fixed load-path behavior, not hot-KV traffic | NVMe is suitable for persistence and cold data, not active full-attention KV per token. |
| Apply a BF16 codec ratio to F16 | SplitZip reports about 1.32x for BF16; F16 has a different exponent/mantissa layout | F16 lossless compression is optional and must be measured on an actual Qwen capture. |
| Assume static lossless compression makes weights plus F16 KV resident | The real 262K capture and artifact tensor scan total 25.0467 GiB under idealized predictors, before codec/runtime overhead | This static coding family cannot fit on a 24 GiB device. |

The previous maximum-context FP4-KV service result remains historical evidence,
not the new fidelity baseline: 262,016 prompt tokens, 128 generated tokens,
5,423.422 s TTFT, and about 3.24 tok/s after the first token. Its provider
telemetry attributed about 3,973.5 s to full attention.

## The physical equation

For the requested F16 KV:

```text
KV/token = 16 layers * 2 (K,V) * 4 heads * 256 values * 2 bytes
         = 65,536 bytes

KV(262,144) = 16 GiB
```

The measured pinned host-to-device bandwidth is 12.46 GiB/s. Let:

- `c` be the fraction of exact F16 KV retained in VRAM for drafting;
- `K = 16 GiB` be the authoritative full KV;
- `A` be the mean number of target-accepted output tokens per exact verify;
- `T_other` include candidate construction, target block compute, recurrent
  bookkeeping, scheduling, and non-overlapped codec work.

The required decode inequality is:

```text
T_cycle = ((1 - c) * K / 12.46 GiB/s) + T_other
A / T_cycle >= 15 tok/s
```

No runtime optimization can bypass this inequality. The useful operating
points are:

| Exact F16 draft subset `c` | VRAM subset | F16 bytes streamed per verify | Transfer floor | Accepted tokens needed if `T_other=0` |
|---:|---:|---:|---:|---:|
| 0.25 | 4 GiB | 12 GiB | 0.963 s | 14.45 |
| 0.375 | 6 GiB | 10 GiB | 0.803 s | 12.04 |
| 0.50 | 8 GiB | 8 GiB | 0.642 s | 9.63 |

The zero-overhead column is not the implementation gate. With `c=0.375` and
`A=24`, the complete candidate and verification machinery has at most about
0.797 s per cycle:

```text
24 / 15 - 0.803 = 0.797 s
```

This is the first operating point with credible capacity and latency margin.
The runtime must measure the actual optimum between `c=0.25` and `c=0.375`;
`c=0.50` is likely too close to the 24 GiB capacity edge.

## Proposed mechanism

The proposal is **exact tiered tree verification**. Approximation is confined
to candidate construction. The target verifier sees all F16 KV values before
it accepts output.

```text
                    one exact populated prefix
                              |
             +----------------+----------------+
             |                                 |
      pinned RAM / NVMe                  RTX 3090 VRAM
      full F16 KV, 16 GiB                FP4 target weights
             |                           exact F16 KV subset c
             |                           recurrent target state
             |                           bounded block proposer
             |                                 |
             |                      candidate lattice / tree
             |                         H <= 32, N <= 128
             |                                 |
             +---- missing F16 KV, once per block ----+
                                                       |
                                         streamed tree verifier
                                         layer/chunk double buffer
                                                       |
                                      exact target logits at every node
                                                       |
                                 accept exact path; discard all other nodes
```

### Tier 0: block-parallel candidate proposer

A small block proposer generates candidate tokens in parallel from target
hidden features. DFlash is the relevant published design: it uses a lightweight
block-diffusion drafter, generates a block in parallel, and leaves final output
to target-model speculative verification. Its paper reports exact/lossless
speculative output and shows why parallel drafting removes the serial draft
cost of EAGLE-style autoregression. It also shows that long-context acceptance
must be trained or adapted explicitly; its published 32K result is not evidence
for 262K. See [DFlash](https://arxiv.org/abs/2602.06036).

The proposer is a capability, not part of target correctness. Its weights,
context representation, and KV may be aggressively compressed or windowed as
long as measured tree coverage remains above the gate. It receives a strict
VRAM budget; a large draft cache is not allowed to evict target weights or the
exact resident subset.

### Tier 1: approximate target tree construction

The target model runs against only the resident exact-F16 subset (`c`) to score
and prune proposer candidates. This target invocation is intentionally lossy
because missing tokens are absent, but it still cannot emit output.

Instead of retaining only one draft path, it keeps a bounded tree. This is the
key change from plain VeriCache:

- a single path fails at the first compressed/full-KV disagreement;
- a bounded tree can retain several high-probability alternatives at each
  depth;
- the expensive prompt KV is common to all nodes, so exact verification can
  reuse one streamed KV tile across many query rows;
- PCIe, not tensor compute, is the scarce resource on this host. Spending spare
  matrix/attention compute to increase accepted depth is the correct exchange.

The initial limits are a horizon of at most 32 and at most 128 total nodes.
They are experiment bounds, not hardcoded architecture geometry. The artifact
and placement policy supply them.

### Tier 2: one exact streamed tree verification

The full target verifies all retained nodes. For each full-attention layer:

1. read the missing F16 KV chunks from pinned RAM into a double buffer;
2. combine them with the exact resident subset using online-softmax state;
3. reuse each KV tile across all candidate queries;
4. apply the tree causal mask so each node sees the prompt and its ancestors;
5. retain target logits and exact candidate KV only until acceptance is known.

The full KV is never materialized in VRAM. Only one or two transport tiles are
resident. This changes the dominant traffic from 10--12 GiB **per output
token** to 10--12 GiB **per accepted block**.

VeriCache establishes the correctness pattern of drafting on lossy KV and
verifying on full KV. It reports approximately 19 accepted tokens at draft
length 30 for Qwen-32B with 4x KV compaction. Its production throughput,
however, relies on multiple-request staggering and PCIe 5.0, so those speed
figures do not transfer to this single-request PCIe 3.0 host. Only its
verification semantics and acceptance precedent are used here. See
[VeriCache](https://arxiv.org/abs/2605.17613).

### Hybrid recurrent-state commit

Qwen3.8 also has 48 Gated DeltaNet layers. Copying the approximately 151.5 MiB
recurrent state for every tree node would consume about 19 GiB at 128 nodes and
is invalid.

Verification therefore processes one model layer at a time and keeps only that
layer's branch states. For final commit it records the compact transition
operands produced for every candidate node (decay/gate/key/value and convolution
updates), not a complete recurrent-state snapshot per node. After the exact
path is selected, the runtime replays only those state-update operators from
the saved pre-cycle checkpoint. This must reproduce the serial state exactly;
otherwise the mechanism is rejected.

This is a generic `recurrent_commit_log` provider capability. Models without
recurrent state do not use it; models with different recurrence must supply a
mathematically valid state-transition implementation.

## Capacity budget

The current executable target-call scan is 12.689922 GiB and the full text
weights including the embedding are about 13.319 GiB. The embedding can be
outside the steady decode set because a token lookup transfers one hidden
vector, while the LM head remains hot.

An initial budget for `c=0.375` is:

| Resident organ | Budget |
|---|---:|
| Hot target FP4 text weights, embedding excluded from steady scan | 12.69 GiB |
| Exact F16 draft subset | 6.00 GiB |
| Block proposer weights and state | at most 1.25 GiB |
| Target recurrent state/checkpoints | about 0.45 GiB |
| Existing staged-weight workspace | about 0.50 GiB |
| Remaining for tree state, transport tiles, allocator and CUDA runtime | about 3.11 GiB |

The proposer budget is a gate, not a claim about a particular downloaded
checkpoint. If the selected proposer and its context state exceed it, the
configuration does not fit and must not be hidden by target-weight offload.

## Why a tree can fit the compute window

One exact attention query over the populated prompt has the following
full-attention arithmetic before implementation overhead:

```text
4 * 16 layers * 24 query heads * 256 * 262,144
= 103.08 GFLOP/query
```

For 128 candidate nodes this is about 13.19 TFLOP. The 3090 must also execute
the block target matrices and recurrent operations, but the calculation shows
why a many-query verifier is qualitatively different from 128 serial tokens:
the KV crosses PCIe once and is reused by a tensor-core-friendly query tile.
The real streamed kernel, not this peak calculation, decides the gate.

## Measured static-lossless residency gate

The gate used the same real coding prompt populated to exactly 262,144 tokens:
262,142 prompt tokens plus two generated tokens. The authoritative target F16
KV is exactly 17,179,869,184 bytes. The profiler sampled 1,048,576 values from
each of the 32 layer/K-or-V streams, uniformly over the populated sequence.

The best independent per-stream raw, token-XOR, token-delta, or channel-XOR
H0 result was:

```text
measured KV entropy       = 13.5851715504 bits/F16 value
ideal KV payload          = 14,586,966,880 bytes = 13.585172 GiB
ideal KV ratio            = 1.177755x
```

The weight gate selected tensors from the artifact operation graph. It
excluded only the sparse embedding-lookup capability from permanent residency,
included the output head, and did not use an architecture name, layer count,
or family tensor-path rule. For every selected FP4 data and scale section it
gave the coder the favorable minimum of raw, XOR, delta, order-1 conditional,
and order-2 conditional entropy independently per tensor. Fractional payload
bits were allowed and metadata cost was zero.

```text
target tensors                         = 850
raw target weight allocations          = 13,622,736,896 bytes
ideal static-lossless weight payload   = 12,147,821,704 bytes
ideal weight ratio                     = 1.121414x
required recurrent state               =    158,859,264 bytes

ideal weights + ideal KV + state       = 26,893,647,848 bytes
RTX 3090 physical capacity             = 25,769,803,776 bytes
absolute deficit                       =  1,123,844,072 bytes
```

The total is 25.046661 GiB. It already exceeds physical capacity by
1.046661 GiB before codebooks, block indexes, decoder workspace, CUDA context,
allocator overhead, or the Windows display reservation. Adding the exact-decode
MTP tensors increases the deficit to 1,324,975,702 bytes.

Consequently the measured static raw/XOR/delta/order-1/order-2 lossless family
is rejected. This is not an information-theoretic impossibility proof for
every reversible transform. A new structural transform has a fail-fast target:
with the measured ideal weight payload and recurrent state, its KV payload must
be no more than 13,463,122,808 bytes, or 12.538510 bits/value, even under the
same impossible zero-overhead assumptions. A practical codec needs additional
margin for all omitted allocations.

The complete evidence is stored in:

- `artifacts/exact-kv-profile-262144.json`;
- `artifacts/exact-kv-residency-gate-262144.json`;
- `artifacts/exact-weight-kv-residency-gate-262144.json`.

## Reversible structural KV and resident attention rejection

A second gate used the real provider to preserve 1,024 uniformly sampled
token rows plus the actual preceding row for every one of the 32 target K/V
streams. The reversible candidates included exponent/channel transforms,
temporal prediction, per-head exponent palettes, exception coding, and
conditioned high/low-byte models. The favorable best-per-stream aggregate was
13.2659264431 bits/value. The zero-overhead full-residency threshold was
12.5385101959 bits/value; after the observed 431 MiB device reservation and a
minimum 512 MiB runtime reserve it was 11.6176117584 bits/value. The structural
family therefore failed even the impossible zero-overhead gate and was not
promoted to a full-context codec implementation. Evidence is retained in
`artifacts/exact-kv-structural-gate-8192.json`.

The next independent gate measured the current device-resident IEEE-FP16
attention provider at the complete 262,144-token geometry. One target
full-attention layer reads exactly 1 GiB of K/V; all 16 layers read 16 GiB per
target call. Fifteen scalar target calls/s require at most 4.166667 ms/layer,
or at least 257.698 GB/s, before every other model operation.

| staged split | measured ms/layer | raw K/V GB/s |
|---:|---:|---:|
| 2,048 | 7.79366 | 137.771 |
| 4,096 | 6.66138 | 161.189 |
| 8,192 | 6.60838 | 162.482 |
| 16,384 | 5.75872 | 186.455 |
| 32,768 | 5.73798 | 187.129 |
| 65,536 | **5.73516** | **187.221** |

The one allowed correction extended the staged workspace from 16K through
64K tokens. Throughput plateaued rather than reaching the required bound. The
best result implies 91.76256 ms per target call across the 16 attention layers,
or at most 10.8977 scalar target calls/s before weights, recurrent layers,
sampling, and service overhead. The staged resident-FP16 provider is therefore
rejected for the 15 tok/s target. A future direct/fused provider may be
considered only together with a capacity mechanism; a faster attention kernel
alone cannot make weights plus full F16 KV fit in 24 GiB. The gate is recorded
in `artifacts/resident-fp16-attention-gate-262144.json`.

## Exact heterogeneous K/V split rejection

Held-out Huffman coding was evaluated before implementing a device codec. A
codebook trained on the first half of every sampled stream and tested on the
disjoint second half required 13.7195388675 bits/value, including literal
escapes for unseen F16 symbols and 2,019,319 bytes of codebooks. Generic
lossless codecs were weaker on the same rows: zlib was about 14.85,
Zstandard about 14.80, and LZ4 about 16 bits/value.

That result still permits a narrow heterogeneous layout: keep all losslessly
encoded K and a V prefix on the GPU, compute the online-softmax probabilities
there, and retain the remaining V tail in host RAM for an exact CPU `P dot V`.
After measured unavailable device memory, a 512 MiB runtime reserve, target and
MTP weights, two recurrent-state copies, and MTP KV, the GPU has
10,329,616,384 bytes for encoded target KV. The held-out Huffman rate represents
12,044,249,784 raw KV bytes, leaving 156,727 V tokens per layer, or
5,135,630,336 F16 value bytes, for every target call on the CPU.

The executable AVX2/F16C kernel used the full 5.135 GB working set and six
queries per KV head. It matched the scalar operation exactly. More threads did
not improve the memory-bound scan:

| CPU threads | ms/target call | F16 V GB/s | maximum absolute error |
|---:|---:|---:|---:|
| 12 | **226.27925** | **22.695984** | 0 |
| 24 | 229.60965 | 22.366788 | 0 |

Even granting impossible perfect native-MTP amplification of two tokens per
target call and zero time for GPU attention, weights, recurrence, probability
transfer, sampling, and service work gives only 8.8386 generated tok/s. This
architecture therefore fails the 15 tok/s target. The complete Windows CUDA
build, five CTests, and 73 integration tests passed after the kernel's single
allowed compiler-portability correction. The evidence is retained in
`artifacts/exact-heterogeneous-kv-gate-262144.json`; the probe remains only as
a reproducible rejection tool, not a serving provider.

Basic cross-layer reversible prediction does not recover the missing margin.
On the captured K streams, raw entropy was about 13.6312 bits/value versus
14.3395 for previous-layer XOR, 14.1721 for delta, and 13.60825 for conditioned
bytes. V streams gave 13.5601 raw, 14.1841 XOR, 14.1685 delta, and 13.5517
conditioned. Same-layer V-from-K XOR/delta was worse still. All remain far
above the practical 11.6176117584-bit threshold. The evidence is retained in
`artifacts/exact-kv-cross-layer-gate-8192.json`.

## Prompt-copy proposer rejection

An unlimited n-gram oracle searched every prior occurrence in the real
262,016-token coding prompt, charged zero lookup/proposer cost, copied up to 32
tokens, and granted one exact target token per cycle. Across the known
127-token exact continuation it achieved only 3.8485 useful tokens/cycle,
maximum 17; none of 33 cycles reached the necessary 24. A practical causal
implementation can only be worse, so a runtime prompt-copy proposer is
rejected without implementation. Evidence is in
`artifacts/exact-ngram-oracle-262016.json`.

## F16 lossless transport is optional

SplitZip is useful evidence that bit-exact KV transport codecs can be fast, but
its representation stores BF16's sign/mantissa byte and compresses its 8-bit
exponent. It reports about 1.32x realized compression on Qwen-family BF16 KV
and bitwise reconstruction. IEEE FP16 has a 5-bit exponent and a 10-bit
mantissa, so the same ratio cannot be assumed. See
[SplitZip](https://arxiv.org/abs/2605.01708).

The core `c=0.25--0.375` design does not require lossless F16 compression. A
codec may be added only if an actual Qwen3.8 F16 capture demonstrates all of:

- useful ratio after metadata, separately for K and V and by layer;
- encode/decode throughput high enough to stay outside the critical path;
- bitwise F16 reconstruction;
- a lower complete verify-cycle time, not merely fewer bytes.

## Prefill and production reuse

Tree verification attacks decode. It does not make the first exact 262K
prefill disappear. The current cold run spent most of its time in full
attention, and exact full-attention prefill remains quadratic in sequence
length.

The production design therefore has two independent prefill paths:

1. use an Ampere-supported exact FlashAttention-2-style provider to remove the
   current custom-kernel gap on a fresh prefix; FlashAttention-2 explicitly
   supports RTX 3090, FP16/BF16, and head dimension 256
   ([official implementation](https://github.com/Dao-AILab/flash-attention));
2. persist the exact token-hashed prefix image (full F16 KV plus recurrent
   state) and reload/reuse the longest exact prefix for repeated harness turns
   and service restarts. Prefix reuse is exact only for identical token
   prefixes; arbitrary file chunks cannot be composed as if they were a
   prefix. The hash contract follows the same correctness principle described
   by [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/).

Cold and reused-prefix TTFT remain separate release numbers. Persistence may
remove repeated work but cannot be presented as faster cold prefill.

## Fail-fast implementation order

No service, pack format, or deployment work starts before the first two gates
pass.

0. **Oracle rejection pre-gate.** Measure the rank of the serial exact path
   under the bounded approximate target. This is allowed to reject a
   configuration, but cannot prove that a causal proposer/tree will preserve
   that path after pruning.
1. **Acceptance/capacity trace.** On a representative populated coding-harness
   trace, measure `c=0.25` and `c=0.375`, candidate-tree coverage, proposer
   memory, proposer latency, and exact accepted depth. Required result: a
   configuration fitting the capacity budget with mean `A >= 24` at horizon
   at most 32 and node count at most 128. One bounded parameter correction is
   allowed; failure then returns to architecture review.
2. **Streamed verifier kernel.** Feed captured F16 KV and 64/128 real query
   rows through layer/chunk double buffering. Compare logits/tokens with the
   serial exact provider and measure the complete transfer plus attention
   time. Required result: the measured values satisfy `A / T_cycle >= 15`,
   using the acceptance from gate 1. A microbenchmark is valid here only
   because it answers this exact inequality.
3. **Recurrent commit log.** Verify accept-all, first-token mismatch, middle
   mismatch, and sampled rejection. Replayed recurrent and convolution states
   must match serial execution and remain inside the declared memory budget.
4. **Generic block executor.** Add artifact-declared proposer, candidate-tree,
   streamed-full-KV, and recurrent-commit capabilities. No model name, fixed
   layer count, or Qwen tensor path may enter common runtime code.
5. **Complete numerical gate.** Greedy output must match serial decoding over
   long coding/tool traces. Sampling must preserve the target distribution and
   deterministic seeded replay contract. Draft-only approximation is never
   reported as target output quality.
6. **Real service gate.** Populate 262,016 prompt tokens, generate enough
   tokens for a stable rate, and report cold/reused-prefix TTFT, accepted depth,
   tree nodes, F16 H2D bytes, verifier cycles, and output tok/s. Acceptance is
   approximately 15 tok/s after the first token through the real harness.
7. **Compatibility and cleanup.** Re-run the common Qwen/DeepSeek lifecycle
   only after the generic provider is frozen, then stop and prove GPU/process
   cleanup. This is compatibility validation, not a substitute performance
   benchmark.

## Decision before the causal gate

Ordinary paging is closed. Pure lossy KV is closed. Lossless F16 compression
alone is not assumed to be large enough. Exact tiered tree verification is the
active candidate because it is the first direction whose capacity and traffic
equations can reach the revised target without changing target output.

At this point it was not yet proven. Gate 1 decides whether Qwen3.8's real 262K coding traces
retain enough exact-path depth in a bounded tree. Gate 2 decides whether the
3090 can verify that tree while streaming the missing F16 KV once. If either
fails after one bounded correction, this direction stops rather than growing
another runtime around a failed premise.

## Measured oracle pre-gate

The first exact capture completed on 2026-08-21 against the published
`${MODEL_ROOT}/qwen3.8-27b-fp4` artifact. The prompt was produced with the
artifact's official tokenizer and `xhigh` thinking template from 388 UTF-8
files in the real repository snapshot. Runtime/build state and `.env` were
excluded. The prompt-token artifact contains exactly 262,016 IDs:

```text
source bytes:             4,897,985
included source chars:    1,016,665
prompt token IDs:         262,016
token IDs SHA-256:        50a6c2ec1ec9a25c2bc59e439f0b380762bd9fe147b2b991714f24b5fa90c986
rendered prompt SHA-256:  5f3a135c52dcdbd37df1fc84c89e147ea6974962ce10942859b20c2e2fe88793
```

The output path was not degenerate: the 127 decoded exact-path tokens contain
81 distinct token IDs, no ID occurs more than six times, and the text begins a
coherent review of the repository and the stated production goal.

Physical runtime evidence:

| Measurement | Result |
|---|---:|
| Populated prompt tokens | 262,016 |
| Generated worker calls | 128 |
| Exact rank positions | 127 |
| Authoritative KV dtype | FP16 |
| Authoritative KV bytes | 17,179,869,184 (16 GiB) |
| Resident tensor bytes | 14,523,879,424 |
| Peak observed GPU allocation | about 18,754 MiB |
| Cold target prefill plus MTP synchronization | 1,134.524 s |
| Instrumented serial rank capture | 298.341 s |

The prefill result is not decode throughput. The rank capture deliberately ran
three serial target calls per position: a 25% preview, a 37.5% preview, and the
complete exact target. Its purpose was to obtain the exact-path ranks.

| Exact-FP16 preview fraction | Top-1 | Rank p95 | Rank max | Oracle mean depth | Oracle mean nodes | Necessary `A >= 24` |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 92.913% | 2 | 3 | 31.75 | 34.75 | pass |
| 37.5% | 92.913% | 2 | 4 | 31.75 | 35.00 | pass |

The oracle used `H=32` and `N=128`; its four consecutive blocks retained
depths `[32, 32, 32, 31]` for both fractions. At `A=31.75`, the 15 tok/s cycle
budget is 2.117 seconds. The measured PCIe floor leaves the following budget
for all tree construction, target block compute, recurrent bookkeeping, and
non-overlapped work:

| Fraction | Missing F16 bytes per verify | Transfer floor | Remaining cycle budget |
|---:|---:|---:|---:|
| 25% | 12 GiB | 0.963 s | 1.154 s |
| 37.5% | 10 GiB | 0.803 s | 1.314 s |

### Decision after the pre-gate

The necessary oracle bound passes, so the direction is not rejected. The 25%
configuration is selected for the real proposer gate: it has the same top-1
rate and accepted-depth upper bound, a lower maximum rank, and a 2 GiB smaller
resident-KV requirement.

This does **not** complete Gate 1. The oracle knows the exact parent at every
depth; a real causal beam/tree does not. The next implementation must run a
bounded proposer/tree with the 25% exact-FP16 subset, count all nodes actually
retained, measure proposer state/latency/VRAM, and test whether the exact path
survives real pruning with mean accepted depth at least 24. Only that result
can promote the design to the streamed-verifier kernel gate.

## Measured causal Gate 1

The artifact's own MTP layer was used as a self-rollout proposer. It generated
a cumulative-log-probability beam with horizon 32 and a total bound of 128
retained prefix nodes. Approximation remained draft-only: the serial target
with authoritative F16 KV selected every output token.

The first trace implementation incorrectly retained only paths ending in the
four final beam leaves. That trace was discarded. The one allowed corrective
patch retained every prefix node constructed at every depth and guarded
variable-length paths during exact comparison. The corrected implementation
then passed the Windows CUDA build, all five CTests, all 73 integration tests,
and a real-provider lifecycle check before the maximum-context rerun.

Corrected maximum-context result:

| Measurement | Result |
|---|---:|
| Populated prompt tokens | 262,016 |
| Maximum context | 262,144 |
| Exact decode positions | 127 |
| Authoritative KV format/bytes | FP16 / 17,179,869,184 |
| Cold prefill plus MTP synchronization | 1,138.452 s |
| Instrumented decode wall time | 241.342 s |
| Completed proposer cycles | 23 |
| Exact tokens accepted in those cycles | 105 |
| Mean accepted depth | **4.565** |
| Minimum / maximum accepted depth | 1 / 11 |
| Mean / maximum retained nodes | 112.348 / 128 |
| Mean proposer latency | 0.667 s/cycle |
| Peak proposer host payload | 4,691,104 bytes |
| Missing 25%-resident F16 transfer floor | 0.963 s/cycle |
| Optimistic ceiling before verifier compute | **2.800 tok/s** |
| Required Gate 1 result | mean depth at least 24 and at least 15 tok/s necessary ceiling |
| Verdict | **fail** |

The 241.342-second decode capture is not production throughput: it includes a
25% approximate target preview, a complete serial exact target call, and the
proposer on every position. The rejection uses the more favorable bound that
removes those serial diagnostic target calls and still reaches only 2.800
tok/s before any streamed-verifier compute.

### Final decision for this proposer

The oracle's 31.75-token upper bound did not survive causal construction. The
corrected MTP tree missed the required acceptance by more than fivefold and
already exceeded the cycle budget before verifier compute. A beam/horizon
parameter sweep cannot repair the mechanism under `H <= 32`, `N <= 128`, and
required mean depth 24: it merely redistributes the same bounded nodes while
the measured tree dies by depth 11 at the latest.

MTP self-rollout is therefore closed. Gate 2 (the streamed verifier kernel),
recurrent commit logging, and service integration are not justified for this
proposer. A future cycle may reconsider exact block verification only with a
materially different parallel proposer whose real 262K causal trace first
passes the same acceptance, capacity, and latency inequalities.

## Full-context FP4 block-Jacobi rejection

The next cycle replaced the MTP-only draft cache with a complete FP4 copy of
all target full-attention K/V while retaining the authoritative F16 cache in
pinned RAM. FP4 remained draft-only. The bounded proposer used one greedy
native-MTP trajectory followed by three 32-position FP4-target block-Jacobi
corrections; the union of their prefixes was capped at 128 nodes.

This configuration has to stream all 16 GiB of authoritative F16 KV during a
future exact verification. Its measured PCIe floor is therefore 1.284 seconds
per cycle. A short-context real-provider gate was deliberately run before the
expensive 262K prefill because dense proposer cost alone can disprove the
latency inequality.

The first implementation took about 1.11 seconds per proposer cycle. The one
allowed correction activated the existing staged BF16/Tensor-Core GEMM path
for 32-row target calls. The complete Windows CUDA build, five CTests, and 73
integration tests passed before the corrected gate.

Corrected gate result:

| Measurement | Result |
|---|---:|
| Prompt / maximum context | 2 / 64 tokens |
| Horizon / correction passes | 32 / 3 |
| Mean / maximum retained nodes | 120.5 / 125 |
| Mean accepted depth | **3.25** |
| Minimum / maximum accepted depth | 0 / 6 |
| Mean proposer latency | **0.972 s/cycle** |
| Full-F16 transfer floor | 1.284 s/cycle |
| Observed necessary ceiling before verifier compute | **1.551 tok/s** |
| Impossible 32/32-acceptance ceiling before verifier compute | **14.18 tok/s** |
| Required result | about 15 tok/s with mean accepted depth at least 24 |
| Verdict | **fail** |

The short context makes this a favorable latency lower bound: a 262K FP4
attention scan can only add work. Even perfect acceptance cannot reach the
target before verifier compute, while measured acceptance misses the gate by
more than sevenfold. The 262K run and streamed verifier are therefore rejected
without further iteration tuning. The retained evidence is
`artifacts/exact-tier-block-jacobi-short-corrected.tsv` and its gate JSON.
