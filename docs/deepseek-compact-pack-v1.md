# DeepSeek compact pack v1

Status: current DeepSeek-specific storage format as of 2026-08-11. It is
accepted by the common MoE VM runner through a storage adapter, but it is not
yet the same physical container as Expert Pack v1. The unification work is in
[the MoE VM handoff](moe-vm-next.md).

DeepSeek compact pack v1 is a placement and I/O representation for routed
experts. It preserves the checkpoint's authenticated FP4/UE8M0 bytes; it is
not another quantization and does not contain expanded SM86 INT8 slots.

The current V4-Flash artifact declares 43 routed layers and therefore has 43
layer shards:

```text
manifest.json
catalog.tsv
extents.tsv
experts-00.dsc
...
experts-42.dsc
experts-00.dsc.commit.json
...
```

Each `experts-LL.dsc` contains 256 records in expert-ID order. Every record is
13,369,344 bytes and begins on a 4,096-byte boundary; a layer shard is exactly
3,422,552,064 bytes. Projection order is:

```text
w1.weight, w1.scale, w3.weight, w3.scale, w2.weight, w2.scale
```

These bytes are identical to the six logical SafeTensors extents recorded by
`deepseek-routed-catalog-v1`. The per-expert SHA-256 is therefore unchanged.
The runtime accepts either six checkpoint extents or one packed extent and
publishes the same `deepseek-fp4-e2m1-ue8m0-block32-v1` source ABI.

## Publication and recovery

Packing never modifies or deletes the checkpoint. A layer is written to a
temporary file, every source record is checked against the catalog, the shard
is flushed, and only then is it renamed and given an atomic commit marker.
Interrupted runs retain the `.partial` directory. `--resume` reuses only
complete layer shards with commit markers and canonical byte size. The final
directory is published only after every config-declared layer and both
catalogs are durable. For the current artifact that count is 43; the publisher
derives it from the pinned checkpoint configuration rather than a common
runtime constant.
A process-scoped file lock rejects overlapping pack writers while still
releasing automatically if a process exits or crashes.

## Build

On Windows use:

```powershell
.\ops\windows\Invoke-DeepSeekCompactPack.ps1 `
  -ModelId "deepseek-ai/DeepSeek-V4-Flash" `
  -Revision "<pinned-revision>" `
  -Output (Join-Path $env:MODEL_ROOT "deepseek-v4-flash/compact-pack-v1") `
  -Resume
```

The loader detects the packed headers and resolves payload shards relative to
the pack, while legacy extent catalogs still resolve against the checkpoint.
Publish the complete `worker-bundle-v3` afterward; the common launcher consumes
that immutable artifact directory, not loose launcher arguments.

The older asynchronous `Start-DeepSeekCompactPack.ps1` wrapper still rejects
every destination under repository `work/`, which conflicts with the current
`MODEL_ROOT=D:/quantum-llm/work/models` convention. It is therefore not the
documented publication path until that guard is made root-aware. The direct
packer above remains transactional/resumable and never deletes or modifies the
source checkpoint.

`Publish-DeepSeekWorkerBundle.ps1` binds the compact catalog, dense/shared/MTP
resources and tokenizer into `deepseek-v4-flash/worker-bundle-v3`. Its manifest
also authenticates `runtime-model.tsv` schema 2. The operation and compression
schedules are derived from the pinned config; adding that program metadata did
not rewrite the compact expert payload.

## Current compute boundary

Version 1 removes six-way scattered reads and creates the stable input layout
for a direct compressed CUDA kernel. That kernel exists: since `05b474e`
("Execute DeepSeek experts directly from packed FP4") the worker uploads the
13,369,344-byte compact record unchanged (`CudaCompactExpertAllocation`),
the directory publishes it as `DeviceExpertFormat::deepseek_fp4_block32`,
and gate/up/down run the packed `__dp4a` selection-batch kernels — the same
geometry-generic kernels the Qwen FP4 pack (ABI 3) reuses. The SM86 expansion
path remains only for FP8 shared experts.

A direct-FP4 qualification prototype kept routed records compact in VRAM and
performed FP4 dequantization inside gate/up/down. It was numerically correct,
but increased a real FFN block from 6.94 to 38.64 ms and the five-token prompt
from 3.74 to 26.75 seconds, so the execution branch was removed. The lesson
carried into `05b474e`: direct compact execution pays off only with packed
vectorized (`__dp4a`) kernels, not per-value dequantization.

Accounting note (fixed 2026-08-10): the routed catalog originally kept
declaring `device_bytes = 25,198,592` (the int8 slot) after direct-FP4
landed, so VRAM budgets, preflight and the census warm set sized every FP4
slot at ~2x its real footprint and `warm_from_census` was hard-capped at 6
experts/layer. The catalog now declares the compact record size
(13,369,344) for routed records; the warm cap derives from the VRAM entry
budget (clamped 6..32/layer). Measured effect and the route-skew analysis
behind the cap choice: docs/benchmarks.md §S1-DeepSeek.

## Qualification result

The complete pack contains 43 shards, 11,008 records, and 147,169,738,752
payload bytes. A real five-token chat prompt retained the expected `Hello`
token. The cold run took 29.05 seconds and read 12.57 GB; an immediate warm
run took 3.61 seconds versus the previous 3.74-second warm baseline. This modest
warm improvement does not satisfy the throughput objective. The evidence
requires persistent RAM/VRAM placement and a faster grouped compute path;
compact files are a prerequisite, not the final optimization.
