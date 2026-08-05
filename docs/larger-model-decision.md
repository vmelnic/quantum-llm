# Larger-model decision record

This record compares candidate architectures for the next runtime backend. It
contains no deployment inventory and does not authorize deletion, download,
conversion, or source-shard reclamation.

Selection considers active parameter mass, checkpoint format, routing and
attention changes, license, implementation reuse, and the disk required to
retain both source and atomic conversion output. Operators must run a local
preflight against their own storage and memory budgets.

## Candidate checkpoints

Sizes below are the sum of repository blobs returned by the Hugging Face API,
not marketing parameter estimates.

| Candidate | Total / active | Geometry | Context | Download | License | Decision |
|---|---:|---|---:|---:|---|---|
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | 122B / 10B | 48 layers, 256 experts, top-8, hidden 3072 | 262K | 233.014 GiB BF16 | Apache-2.0 | Safest source dtype, but largest Qwen source. |
| [Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) | 122B / 10B | same | 262K | 118.460 GiB | Apache-2.0 | Lower-risk Qwen-family integration after FP8 input support. |
| [Qwen3.5-122B-A10B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4) | 122B / 10B | same | 262K | 73.486 GiB | Apache-2.0 | Smallest, but introduces a second quantization ABI and quality provenance. |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | 284B / 13B | 43 layers, 256 routed experts, top-6 + shared, hidden 4096 | 1M | 148.667 GiB mixed FP4/FP8 | MIT | Selected next backend; exercises a materially larger and newer architecture. |
| [Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) | 397B / 17B | 60 layers, 512 experts, top-10, hidden 4096 | 262K | 378.302 GiB | Apache-2.0 | Requires a large atomic conversion workspace. |
| [Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code) | 1T / 32B | 61 layers, 384 routed experts, top-8 + shared, hidden 7168 | 256K | 554.328 GiB | Modified MIT | Future distributed and storage-sharded target. |

DeepSeek-V4-Pro at 1.6T total / 49B active is also a future distributed target,
not a sensible first compatibility step.

## Runtime compatibility gap

The current compiler accepts only BF16, F16, and F32 source tensors and only
the `olmoe` and `qwen3_next` adapters. The CUDA runner also validates the exact
Qwen3-Next 80B geometry. Therefore none of the candidates is runnable merely
by downloading it.

Qwen3.5-122B is the smallest useful architecture step:

- add a `qwen3_5_moe` adapter and explicitly select the language model from the
  multimodal wrapper;
- reject or intentionally preserve vision and MTP tensors rather than silently
  dropping them;
- support hidden 3072, 48 layers, 256 experts, top-8 and the Qwen3.5 hybrid
  attention layout;
- for the recommended checkpoint, decode block-scaled FP8 while streaming into
  the existing per-row INT8 Expert Pack, or define a new compute-ready ABI;
- qualify tokenizer/chat template, exact routing, dense kernels, KV layout and
  output against a trusted reference implementation.

DeepSeek-V4-Flash requires a `deepseek_v4` adapter, mixed FP4/FP8
source decoding, CSA + HCA attention, manifold-constrained hyper-connections,
and the model's routing/shared-expert semantics. It is a new backend, not a
small repack of the existing Qwen implementation.

## Pack space and time estimate

The existing 80B conversion is measured, not inferred:

- 162,659,161,528 source tensor bytes;
- 81,903,198,208 output pack bytes;
- 3,371.421 seconds (56.19 minutes);
- no source shards reclaimed.

Scaling the same INT8-per-row format by parameter count gives a planning range
of 115–125 GiB and 85–100 minutes for Qwen3.5-122B after its adapter and source
decoder exist. This is an engineering estimate; the first representative
dense tensor and one complete routed layer must measure decode rate, output
size, and numerical error before a full conversion starts.

DeepSeek-V4-Flash cannot use that Qwen scaling factor directly: converting
mixed FP4/FP8 source tensors into an INT8 compute-ready ABI can make the output
larger than the downloaded checkpoint. Its preflight must derive output mass
from the pinned tensor index and a representative converted layer.

Disk preflight must reserve the complete source checkpoint, estimated pack,
atomic `.partial` output, validation overhead, logs, and a safety margin. Source
reclamation is never part of the default capacity calculation.

## Recommendation and approval boundary

1. Use `deepseek-ai/DeepSeek-V4-Flash` as the next backend target because its
   284B total / 13B active geometry exercises a genuinely larger directory and
   a different modern architecture.
2. Pin every source download to an immutable revision and verify the published
   SafeTensors index before compiler work begins.
3. Implement mixed-source decoding and one representative routed layer before
   starting a full Expert Pack conversion.
4. Produce packs atomically, validate every record independently, and pass
   correctness/API gates before making performance claims.
5. Retain Qwen3.5-122B as the lower-risk fallback if the DeepSeek dense backend
   cannot initially meet bounded implementation or memory requirements.

Cleanup and download remain separate operator actions. In particular,
`DELETE_CONSUMED_SHARDS` stays disabled until repacking and rollback are
independently proven.

## DeepSeek implementation status

- Complete: pinned download and independent index/shard verification.
- Complete: read-only dtype inventory and exhaustive source metadata contract.
- Complete: FP4 E2M1 and UE8M0 numerical decoding with exhaustive value tests.
- Complete: one real routed expert slice, bitwise PyTorch reference comparison,
  INT8 error metrics, and metadata-derived representation/space measurements.
- Complete: platform-neutral compact-source and 25,198,592-byte SM86 hot-slot
  ABI, with exact layout enforced by native runtime tests.
- Complete: compact H2D plus SM86 admission for one real expert; the complete
  hot slot is byte-identical to the independent candidate and takes 9.73 ms.
- Complete: gate/up/SILU/down CUDA execution directly from the admitted slot;
  one real expert takes 0.493 ms and matches the CPU output within `4.47e-8`.
- Complete: representation-aware cache capacity; compact bytes and expanded
  device bytes are budgeted independently before I/O or publication.
- Complete: compact-record validation and CUDA admission run behind the existing
  IOCP, pinned-buffer, single-flight cache, atomic directory publication, and
  eviction lifecycle. Two concurrent requests produced one read and one upload.
- Complete: direct bounded SafeTensors extent gathering replaces copied expert
  fixtures; the descriptor is about 3 KiB and the source checkpoint remains the
  sole authoritative weight copy.
- Complete: the real FP8 shared expert is bitwise-equal to the independent
  PyTorch decode, admits into the common SM86 slot through the cache, and runs
  gate/up/down CUDA from that slot.
- Complete: all 43 shared slots are atomically described, loaded through one
  bounded staging slot, pinned as model state, directory-verified, and released
  at teardown; measured startup was 3.59 seconds.
- Complete: a generic FP8 dense-matrix source/admission/GEMV ABI passes on the
  real layer-0 query projection with an exact candidate hash.
- Complete: all 236 main-model FP8 matrices load atomically as 4.78 GB of
  resident, named device state through one 33.6 MB staging slot.
- Complete: the real layer-0 attention HCA site runs its exact F32
  RMS-rescale, 24-way projection, sigmoid controls, 20-step Sinkhorn stream
  mixer, collapse, and expansion on CUDA against an independent oracle.
- Complete: all 834 main-model BF16/F32/I64 tensors stream transactionally
  into dtype-preserving device state through one fixed 64 MiB staging slot;
  tensors larger than the slot are hashed and uploaded incrementally.
- Complete: dense and typed sets publish as one 7,772,173,404-byte model
  transaction; fail-closed pointer bindings cover real ratio-4 and ratio-128
  attention layers without hot-path name lookup or reupload.
- Complete: the real CSA decode compressor passes for both checkpoint
  schedules—overlap ratio 4 and ordinary ratio 128—including BF16 projections,
  persistent state, gated pooling, and normalization.
- Complete: compressed KV publication applies RoPE64 and the block-64 MXFP8
  QAT boundary, then stores a 1,024-byte BF16 cache slot; both schedules matched
  their independent cache oracle bit-for-bit.
- Complete: sparse attention consumes the BF16 cache with online softmax,
  sentinel handling, and the learned attention sink; the real layer-2 sink
  passed its independent output oracle.
- Complete: the ratio-4 index path runs its real 128-wide overlap compressor,
  RoPE64, scaled Hadamard transform, block-32 E2M1 QAT boundary, learned
  per-head scoring projection, and stable top-k selection against independent
  oracles.
- Complete: the first real attention sublayer composes HCA, resident dense and
  typed state, Q/window KV, both compressors, index selection, sparse
  attention, grouped output projection, and HCA post. Four sequential tokens
  passed the independent full-graph oracle; token three emitted and consumed
  the first compressed/indexed slot with `4.02e-4` maximum composed error.
- Next: optimize the measured 7.22 ms layer path, then connect attention to
  routing/shared/streamed experts in the first complete transformer block.
