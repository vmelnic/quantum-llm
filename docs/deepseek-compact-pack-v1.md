# DeepSeek compact pack v1

DeepSeek compact pack v1 is a placement and I/O representation for routed
experts. It preserves the checkpoint's authenticated FP4/UE8M0 bytes; it is
not another quantization and does not contain expanded SM86 INT8 slots.

The container has 43 layer shards:

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
directory is published only after all 43 layers and both catalogs are durable.
A process-scoped file lock rejects overlapping pack writers while still
releasing automatically if a process exits or crashes.

## Build

On Windows use:

```powershell
.\ops\windows\Invoke-DeepSeekCompactPack.ps1 `
  -ModelId "deepseek-ai/DeepSeek-V4-Flash" `
  -Revision "<pinned-revision>" `
  -Resume
```

Pass the resulting directory as `-RoutedCatalog` to the model launcher. The
loader detects the packed headers and resolves payload shards relative to the
pack, while legacy extent catalogs still resolve against the checkpoint.

## Current compute boundary

Version 1 removes six-way scattered reads and creates the stable input layout
for a direct compressed CUDA kernel. Until that kernel is selected by the
device ABI, admission still expands a cold record into the existing
25,198,592-byte INT8 SM86 slot on the GPU. The pack alone improves I/O shape
but does not claim to remove conversion cost.

A direct-FP4 qualification prototype kept routed records compact in VRAM and
performed FP4 dequantization inside gate/up/down. It was numerically correct,
but increased a real FFN block from 6.94 to 38.64 ms and the five-token prompt
from 3.74 to 26.75 seconds, so the execution branch was removed. Direct compact
execution becomes a production candidate only with admission-time GPU
expansion or grouped multi-row/Tensor Core kernels.

## Qualification result

The complete pack contains 43 shards, 11,008 records, and 147,169,738,752
payload bytes. A real five-token chat prompt retained the expected `Hello`
token. The cold run took 29.05 seconds and read 12.57 GB; an immediate warm run
took 3.61 seconds versus the previous 3.74-second warm baseline. This modest
warm improvement does not satisfy the throughput objective. The evidence
requires persistent RAM/VRAM placement and a faster grouped compute path;
compact files are a prerequisite, not the final optimization.
