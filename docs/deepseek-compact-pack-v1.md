# DeepSeek compact pack v1

Status: deployed routed-expert and worker-bundle contract, 2026-08-23.

## Why it is separate

DeepSeek-V4-Flash stores routed experts as compact block-scaled FP8 source
extents. The runtime's SM86 expert ABI repacks each complete expert into one
authenticated FP4/UE8M0 record. This physical layout is different from QPack,
but both artifacts publish the same model-program/provider and logical-page
contracts.

The physical formats must not be described as unified until the source ABI,
record authentication and demand-paging geometry can be preserved without a
model-specific execution stack.

## Geometry and files

The main decoder contains 43 routed layers × 256 experts = 11,008 logical
pages. Each compact record is 13,369,344 bytes; all routed payloads total
147,169,738,752 bytes before indexes and bundle metadata.

`pack-deepseek-routed` publishes:

- one `experts-*.dsc` shard and commit descriptor per layer;
- `catalog.tsv`, mapping layer/expert keys to one packed extent;
- `extents.tsv`, physical path/offset/length information;
- `manifest.json`, source identity, geometry and hashes.

Packing is resumable by committed layer. A layer is accepted only after all
256 records and hashes are durable. The main MTP layer, when enabled, is a
separate namespace and compact pack.

## Worker bundle

`Publish-DeepSeekWorkerBundle.ps1` validates and combines:

- dense FP8 descriptor set;
- typed BF16/F32/I64 descriptor set;
- always-active shared-expert descriptor set;
- main routed catalog/compact pack;
- tokenizer/config/encoding assets;
- optional MTP dense/typed/shared/routed resources;
- `runtime.tsv`, `runtime-model.tsv` schema 3 and an authenticated manifest.

Large immutable weights may remain referenced in the pinned checkpoint while
routed compact shards live on NVMe. All locations are captured in the
published bundle; the common service receives only the stable bundle path
below `MODEL_ROOT`.

## Build outline

1. inspect and validate the pinned source with the `deepseek_v4` contract;
2. export dense, typed, shared and routed descriptors;
3. pack the routed catalog with `Invoke-DeepSeekCompactPack.ps1`;
4. build and qualify optional MTP resources as one complete namespace;
5. run independent numeric oracles for FP8 admission, attention, HCA, routing,
   experts and I/O;
6. publish `worker-bundle-v3` only after every dependency is complete.

The repository deliberately keeps the direct compact-pack command instead of
the former background wrappers: publication state already provides the
transaction boundary and one operator-visible command is less ambiguous.

## Runtime semantics

The router selects exact top-6 experts. Demand acquisition waits for all six,
uses stable route order and weights, and never reduces top-k. Routed records
move through SSD, protected/probationary RAM and transient/protected VRAM.
Telemetry attributes hits, reloads, reread bytes, storage wait and H2D wait.

The current bundle is executable on the SM86 RTX 3090 provider. P40/P100
providers and multi-device ownership remain roadmap work, not properties of
this format.
