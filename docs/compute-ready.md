# Compute-ready representations

`Compute-ready` is a contract between stored bytes and one concrete kernel. It
does not merely mean "quantized" or "packed into one file". A record is
compute-ready only when the selected kernel can consume it without a full
transpose, reorder, requantization, or expanded persistent copy.

The term is always relative to a kernel ABI. The same FP4 tensor may be
compute-ready for a scalar/DP4A kernel and unsuitable for a Tensor Core kernel
that requires tiled, interleaved operands and different scale placement.

## Four distinct states

| State | What is guaranteed | What may still be required |
|---|---|---|
| Checkpoint/source | Correct model tensors and dtype | Gather extents, validate, reorder, quantize |
| Placement-ready | One expert is independently addressable, aligned and authenticated | Kernel-specific conversion may remain |
| Compute-ready | Exact dtype, tile/order, scales, padding and metadata expected by a named kernel ABI | Copy into RAM/VRAM and lightweight publication |
| Resident/device-ready | Allocation is on the executing device and published in the expert directory | Only activation-dependent execution |

These states must not be conflated. Contiguous I/O removes scattered reads but
does not automatically make arithmetic fast. Likewise, compute-ready bytes on
SSD still incur storage latency before they become resident.

## What the ABI fixes

A useful compute-ready record fixes all of the following:

- projection order and whether gate/up are fused;
- logical dimensions and physical tile dimensions;
- quantized value encoding, signedness and nibble/bit order;
- scale dtype, block geometry and scale interleave;
- row-, column-, tile-, or swizzled-major order;
- padding and alignment for direct I/O, vector loads and device copies;
- the CUDA/Metal/CPU kernel family allowed to consume the record;
- numerical accumulation/output types and required hardware features;
- record hash, version, endianness and fail-closed compatibility rules.

The runtime should choose the kernel from the declared ABI. It must never guess
an interpretation from tensor dimensions.

## Current project formats

Expert Pack v1 is compute-ready for the existing INT8-per-row Qwen/OLMoE
kernels. Its cost is size: converting lower-bit source weights to INT8 can
roughly double routed-weight traffic. Since 2026-08 the same container also
carries an FP4-E2M1/UE8M0 block-32 profile (quant ABI 3, kernel ABI
`expert-pack-sm86-fp4-block32-v1`) consumed directly by packed `__dp4a`
selection-batch kernels; the measured FP4 Qwen pack halves routed bytes per
token (0.7 GiB vs 1.41 GiB) and reached 39.6–45.6 tok/s on resident routes
(docs/benchmarks.md §S1b).

DeepSeek compact pack v1 is placement-ready and preserves the checkpoint's
13,369,344-byte FP4/UE8M0 expert records. The current SM86 DP4A path consumes
those compact records directly, so they are compute-ready for that specific
kernel. They are not yet a high-throughput Tensor Core ABI; a naive direct FP4
kernel was correct but too slow.

Dense and shared tensors currently use their own admitted device ABIs. Their
resident representation consumes VRAM that could otherwise retain routed
experts.

## The smaller representation direction

The proposed next representation is not "compress a file and decompress it
before every token". It must remain compressed through storage, RAM and VRAM,
then be consumed directly by a fast kernel. Candidate work includes:

- 2--3-bit or mixed-bit routed weights with separately gated quality loss;
- compact scale encodings and larger or adaptive quantization blocks;
- offline tile/swizzle layout matching the selected CUDA MMA kernel;
- fused dequantization inside grouped GEMM, amortized over several rows;
- a distinct ABI per hardware/kernel family when layouts genuinely differ.

For DeepSeek, routed experts are already FP4. Merely repacking the same four-bit
values cannot provide the missing order-of-magnitude bandwidth. A materially
smaller ABI, better route residency, or row batching must accompany the kernel.
At the measured roughly 3.45 GB of arbitrary routed weights per token, 30 tok/s
would require more than 100 GB/s of weight delivery. Even halving the record to
two bits still requires roughly 52 GB/s if every selected expert is cold.

Therefore compute-ready is necessary but not sufficient. The production goal
is: smaller bytes, direct fast execution, and enough reuse/batching that those
bytes do not move again for every single token.

## Acceptance boundary

A new ABI is useful only when it passes all of these gates:

1. representative layers and full-model tokens meet declared error/identity
   tolerances against the pinned checkpoint;
2. stored, RAM and VRAM bytes are measured rather than estimated;
3. the selected kernel is faster end-to-end than the current DP4A path;
4. cold admission performs no hidden full-size persistent expansion;
5. record addressing, hashes, cancellation and eviction remain bounded;
6. warm and cold API measurements improve a real target, not only a microbench.
