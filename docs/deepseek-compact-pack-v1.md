# DeepSeek compact pack v1

Status: deployed compact expert and self-contained bundle contract,
2026-09-02.

## Purpose and geometry

DeepSeek's source layout and paging geometry differ from QPack. The compiler
repackages each complete routed expert into one authenticated FP4 E2M1/UE8M0
record without changing router selection or top-k semantics.

The main decoder contains:

```text
43 routed layers x 256 experts = 11,008 logical pages
13,369,344 bytes per compact record
147,169,738,752 routed payload bytes
```

`pack-deepseek-routed` writes committed layer shards, `catalog.tsv`,
`extents.tsv` and a manifest. Packing resumes by complete committed layer; a
partial layer is never published. MTP routed state, when present, uses a
separate authenticated namespace.

## Worker bundle

`Publish-DeepSeekWorkerBundle.ps1` validates and publishes below
`${MODEL_ROOT}/deepseek-v4-flash`:

- dense FP8 and typed BF16/F32/I64 descriptor sets;
- always-active shared-expert descriptors;
- main routed compact catalog and payloads;
- tokenizer/config/encoding assets;
- complete optional MTP dense/typed/shared/routed resources;
- `runtime.tsv`, schema-3 `runtime-model.tsv`, manifest and completion state.

The active alias points at `deepseek-v4-flash/worker-bundle-v3`. Runtime
dependencies are self-contained below `MODEL_ROOT`; Hugging Face cache paths
are not serving dependencies.

## Publication

1. inspect the pinned source with the `deepseek_v4` contract;
2. export dense, typed, shared and routed descriptors;
3. pack and commit every routed layer;
4. publish MTP only as a complete namespace;
5. run independent FP8/admission, attention, HCA/CSA, routing, expert and I/O
   oracles;
6. publish `worker-bundle-v3` only after all dependencies validate;
7. start through the common VM and run the real DeepSeek chat gate.

The direct pack command is the supported transaction boundary. Removed
background wrappers are not an alternate operator path.

## Runtime semantics

The router selects exact top-6 experts. Acquisition waits for every selected
record, preserves route order/weights and never substitutes zero or reduced
top-k. Pages move through NVMe, retained RAM and routed VRAM under leases and
priority classes. Telemetry attributes tier hits, reload/reread bytes, storage
wait, uploads and CPU/GPU expert decisions.

The 147 GB routed pool exceeds the current 48 GiB host cache. Demand paging is
therefore genuine: cold routes still touch NVMe, while retained RAM-ready pages
may execute on CPU or upload to GPU. The validated common VRAM-cache ceiling is
12 GiB; 13 GiB fails current DeepSeek preflight after fixed allocations,
workspace and reserve.

The bundle passes current direct chat and minimal Pi wiring gates. Those gates
do not qualify maximum context, automatic cancellation recovery or the settled
10-15 tok/s target.
