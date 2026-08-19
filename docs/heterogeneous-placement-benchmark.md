# Heterogeneous placement benchmark

Status: calculation and evidence record, 2026-08-19. This document contains no
claim that heterogeneous CPU/GPU placement is implemented.

## Evidence sources

Artifact inventory:

```text
D:/quantum-llm/work/models/qwen3.8-27b-fp4/manifest.json
source: Qwen/Qwen3.8-27B
source revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
dense.qpack: 14,775,390,208 bytes
records: 1,199
```

CUDA evidence:

```text
D:/quantum-llm/out/build/windows-msvc-release/Testing/Temporary/LastTest.log
test: expert-dense-fp4-cuda-smoke
run: 2026-08-19 12:36 EEST
result: pass=true
```

Service evidence is the completed 262,016-token prompt plus 128 generated
tokens recorded in [Current state and next work](moe-vm-next.md). Host PCIe
measurements are recorded in [Performance evidence](benchmarks.md#host-bandwidth-pinned-measurement-2026-08-10).

All GB/s values printed by the CUDA smoke are decimal GB/s. Artifact capacities
are also shown in binary GiB. The calculations do not mix these units.

## Exact artifact inventory by organ

The table is the sum of `stored_bytes` in the published manifest, classified by
tensor role/name. It includes per-record storage alignment.

| Organ | Records | Stored bytes | GiB |
|---|---:|---:|---:|
| Embedding | 1 | 675,434,496 | 0.629047 |
| LM head | 1 | 675,434,496 | 0.629047 |
| Text MLP/FFN | 192 | 9,091,940,352 | 8.467529 |
| Recurrent/linear-attention weights | 432 | 2,963,472,384 | 2.759949 |
| Full-attention weights | 96 | 891,682,816 | 0.830444 |
| Text norms/other | 129 | 3,170,304 | 0.002953 |
| MTP | 15 | 225,771,520 | 0.210266 |
| Vision encoder/projector | 333 | 248,483,840 | 0.231419 |

The target-call weight scan excludes the embedding lookup, vision and MTP but
includes the LM head:

```text
9,091,940,352  text FFN
2,963,472,384  recurrent-attention weights
  891,682,816  full-attention weights
  675,434,496  LM head
    3,170,304  norms/other
─────────────
13,625,700,352 bytes = 12.689922 GiB per target call
```

This is an optimistic traffic model. Hardware caches may reuse small records,
while additional intermediates, MTP and kernel traffic add bytes not counted as
target weights.

## Exact active-state geometry

### FP4 KV

The pack/runtime FP4 block contains 16 data bytes and one scale byte for every
32 values:

```text
17 / 32 = 0.53125 bytes/value
```

Qwen3.8 has 16 full-attention target layers, four KV heads and head dimension
256. K and V are both retained:

```text
target KV bytes/token
  = 16 layers * 2 (K,V) * 4 heads * 256 values * 17/32
  = 17,408 bytes/token

target KV at 262,144
  = 17,408 * 262,144
  = 4,563,402,752 bytes
  = 4.25 GiB
```

The one MTP full-attention layer adds:

```text
1,088 bytes/token * 262,144
  = 285,212,672 bytes
  = 0.265625 GiB
```

Total target plus MTP KV at maximum context is 4.515625 GiB.

### Recurrent state

Per one of the 48 recurrent layers:

```text
conv values   = 10,240 * 4 = 40,960 float32
matrix values = 48 * 128 * 128 = 786,432 float32
bytes/layer   = (40,960 + 786,432) * 4 = 3,309,568
```

The active state is therefore exactly 151.5 MiB. Active state plus the current
rollback and retention checkpoint allocations is 454.5 MiB.

The provider also allocates a 510 MiB BF16 staged-weight workspace. Other CUDA
workspaces, allocator overhead and the Windows display reservation are not
included in the placement subtotal below.

## Measured CUDA kernel evidence

Relevant fields from the passing CUDA smoke:

| Quantity | Measured |
|---|---:|
| FP4 weight GEMV effective bandwidth | 779.326 GB/s |
| Batch selected decode bandwidth | 636.33 GB/s |
| Tensor-core 262K attention | 1.73747 ms/layer probe |
| Tensor-core attention effective KV bandwidth | 164.154 GB/s |
| Scalar 262K attention | 3.63533 ms/layer probe |
| Scalar attention effective KV bandwidth | 78.4558 GB/s |
| Two-position microbatch | 1.84858 ms/layer probe |
| Two-position prefill path | 9.89824 ms/layer probe |
| Maximum tensor-core attention difference | 1.81049e-6 |

These are isolated native probes. They are bandwidth evidence and lower-bound
inputs, not service tok/s predictions.

## Maximum-context decode lower bounds

Using decimal bytes with the decimal GB/s reported by the smoke:

```text
target weight floor
  = 13,625,700,352 / 779.326e9
  = 17.484 ms/target call

target KV floor
  = 4,563,402,752 / 164.154e9
  = 27.800 ms/target call

combined lower bound
  = 45.284 ms/target call
  = 22.08 target calls/s
```

This omits recurrent arithmetic, quantization, reductions, MTP work, launches
and orchestration. Consequently 22.08 is an upper bound on scalar target-call
rate under these measured kernel throughputs, not an expected result.

The completed maximum-context run emitted 128 tokens from 71 target calls:

```text
measured greedy-MTP amplification = 128 / 71 = 1.802817
optimistic output ceiling          = 22.08 * 1.802817
                                   = 39.8 output tok/s
```

This acceptance belongs to that run. It must not be applied to production
sampling until exact speculative sampling is implemented and measured for the
production distribution. With amplification `A = 1`, the same isolated
bandwidth inputs cannot reach 30 output tok/s.

The real service result remains:

```text
prefill tokens:       262,016
generated tokens:     128
TTFT:                 5,423.422 s
wall:                 5,462.672 s
post-first-token:     approximately 3.24 tok/s
```

The gap between this result and the isolated lower bound proves that the bound
is not an end-to-end prediction.

## Whole-KV offload bounds

### RAM copied to GPU

Measured pinned H2D bandwidth is 12.46 GiB/s:

```text
4.25 GiB / 12.46 GiB/s = 341 ms/target call
```

This bounds target calls below 2.94/s before compute. Whole active KV copied
from RAM is incompatible with the 30 tok/s target.

### CPU direct from RAM

The Ryzen platform's DDR4-3200 dual-channel peak is 51.2 GB/s. This is a
theoretical interface ceiling, not a measured CPU-kernel result:

```text
4,563,402,752 / 51.2e9 = 89.13 ms/target call
```

Even at that unattainable-perfect memory rate, whole-KV CPU attention permits
only 11.22 target calls/s. At the historical greedy-MTP amplification it would
be 20.2 output tok/s before all other work.

### NVMe

The active D: NVMe was not benchmarked for this calculation. Even granting a
generous illustrative 7 GB/s ceiling:

```text
4,563,402,752 / 7e9 = 652 ms/target call
```

Therefore active full-attention Qwen KV is not an NVMe candidate. This is a
bound, not a measurement of D:.

## Concurrent GPU/CPU KV partition

Let `B_gpu = 164.154 GB/s` be the measured GPU attention bandwidth and give the
CPU the optimistic `B_cpu = 51.2 GB/s` DDR ceiling. Equal completion time for
two concurrent sequence ranges requires:

```text
CPU fraction = B_cpu / (B_gpu + B_cpu)
             = 0.237748
```

Candidate placement:

| KV partition | GiB |
|---|---:|
| GPU/VRAM | 3.239571 |
| CPU/RAM | 1.010429 |

The optimistic split-stage floor is:

```text
4,563,402,752 / (164.154e9 + 51.2e9)
  = 21.190 ms/target call
```

This could reduce the isolated KV stage by at most 23.8% and release about
1.01 GiB of VRAM. No gain is claimed until a packed CPU attention kernel and
the concurrent exact merge are measured.

## Concurrent GPU/CPU weight partition

Under the same optimistic CPU bandwidth and the measured 779.326 GB/s GPU
weight rate:

```text
CPU weight fraction = 51.2 / (779.326 + 51.2)
                    = 0.061648
CPU weight shard    = 0.782304 GiB
combined floor      = 16.406 ms/target call
```

The theoretical improvement is only about 6%. Real CPU FP4 GEMV compute may be
below the DDR ceiling, so KV partitioning is the stronger first candidate.

For BF16 activation exchange, sending one hidden vector to the CPU and one
partial result back for both attention and MLP over 64 layers is approximately:

```text
2 directions * 5,120 * 2 bytes * 64 layers * 2 organs
  = 2.5 MiB/token

2.5 MiB / 12.46 GiB/s = 0.196 ms/token
```

This byte cost is small; per-layer dispatch and synchronization still need to
be eliminated or amortized in the implementation.

## Combined optimistic candidate

Combining the theoretical CPU weight share and CPU KV range:

```text
weight stage = 16.406 ms
KV stage     = 21.190 ms
total floor  = 37.596 ms/target call
```

This is 26.60 scalar target calls/s, still below 30 before other work. At the
historical greedy-MTP amplification it is 47.95 output tok/s before overhead.
The arithmetic therefore says:

- heterogeneous CPU placement alone does not prove 30 tok/s;
- exact speculative amplification is required under the measured isolated
  rates unless the GPU kernels improve materially;
- the CPU KV range is a credible capacity/bandwidth experiment;
- NVMe is not an active full-attention KV tier for Qwen3.8.

An illustrative persistent subtotal after both candidate splits is about
16.565 GiB:

```text
11.9076 GiB  GPU share of target-scan weights
 0.2103 GiB  MTP weights
 3.2396 GiB  GPU target KV
 0.2656 GiB  MTP KV
 0.4438 GiB  recurrent state plus two checkpoints
 0.4980 GiB  staged BF16 weights
```

This excludes all other workspaces and is not a VRAM allocation guarantee.

## Model-independent implications

- Embeddings and inactive modality organs are capacity candidates because they
  are not fully scanned per text token.
- Full-attention KV can be split across compute tiers using an exact online
  softmax merge; copying the RAM range to GPU defeats the purpose.
- Dense MLPs and recurrent heads can be tensor-sharded, but CPU share is bounded
  by its measured packed-kernel rate.
- Routed experts can be assigned whole to VRAM/GPU or RAM/CPU; only activation
  vectors need cross PCIe.
- NVMe can serve inactive or conditionally selected data only while active
  selected bytes satisfy `bytes/token <= bandwidth/rate`.

The associated design and qualification gates are in
[Heterogeneous organ placement](heterogeneous-placement-research.md).

