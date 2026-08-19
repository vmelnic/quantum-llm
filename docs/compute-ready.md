# Compute-ready representations

Status: current representation contract as of 2026-08-19. Physical-container
unification is not complete; see [the MoE VM handoff](moe-vm-next.md).

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

The active Qwen3.8 Expert Pack carries dense FP4-E2M1/UE8M0 block-32 matrices
(quant ABI 3) plus separately declared raw ABI-0 records. The SM86 dense
provider consumes those packed matrices directly; Qwen3.8 has no routed
experts. Historical Qwen3-Next and LFM Expert Packs use the same quant ABI for
routed records and remain performance evidence rather than the active target.

DeepSeek compact pack v1 is placement-ready and preserves the checkpoint's
13,369,344-byte FP4/UE8M0 expert records. The current SM86 DP4A path consumes
those compact records directly, so they are compute-ready for that specific
kernel. They are not yet a high-throughput Tensor Core ABI; a naive direct FP4
kernel was correct but too slow.

Dense, typed and shared tensors use their explicitly admitted device ABIs.
Their resident representation consumes VRAM that may otherwise hold KV pages
or sparse-model expert records.

Both formats now bind a checksummed `runtime-model.tsv` program. This unifies
the execution description, not the physical payload container: DeepSeek still
enters through the worker-bundle/compact-pack storage adapter.

## Performance direction

Any future representation must remain compressed through storage, RAM and
VRAM, then be consumed directly by a fast kernel. Current work does not include
a lower-bit or reduced-quality conversion. Representation work is limited to:

- offline tile/swizzle layout matching the selected CUDA MMA kernel;
- fused dequantization inside grouped GEMM, amortized over several rows;
- a distinct ABI per hardware/kernel family when layouts genuinely differ.

For DeepSeek, routed experts are already FP4. Merely repacking the same four-bit
values cannot provide the missing order-of-magnitude bandwidth. A materially
smaller ABI, better route residency, or row batching must accompany the kernel.
At the measured roughly 3.45 GB of arbitrary routed weights per token, 30 tok/s
would require more than 100 GB/s of weight delivery when every selected expert
is cold. The current plan therefore focuses on exact working-set reuse,
placement and direct execution rather than trading away model quality.

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
