# Expert Pack compiler

The compiler converts a local SafeTensors checkpoint into independently
addressable, compute-ready Expert Pack v1 records. It reads through read-only
memory mappings and never instantiates the Transformers model or materializes
the complete checkpoint.

Supported strict adapters:

- `olmoe` — `allenai/OLMoE-1B-7B-0125-Instruct` geometry;
- `qwen3_next` — `Qwen/Qwen3-Next-80B-A3B-Instruct`, including explicit MTP
  tensor preservation.

Unknown, missing, or geometrically inconsistent tensors fail conversion.

## Inspect an unfamiliar checkpoint

Before adding an architecture adapter or defining a target ABI, validate every
SafeTensors header and obtain an architecture-neutral byte inventory:

```powershell
.\.venv\Scripts\python.exe -m compiler inspect-source `
  --source C:\path\to\snapshot
```

The command is read-only. It does not map tensor payloads, convert weights, or
create output files. Unsupported dtypes, unsafe shard paths, overlapping tensor
ranges, index/header disagreements, and truncated shards fail closed.

Use `--tensor-groups` during adapter development to group numeric layer/expert
IDs while retaining each observed dtype and shape variant. This is still a
metadata-only operation; it can produce a long JSON report.

For the pinned DeepSeek-V4-Flash geometry, add `--contract deepseek_v4`. The
contract exhaustively validates names, dtypes, shapes, byte lengths, routing,
CSA/HCA, MTP, and quantization metadata. It remains source-only and cannot
start an Expert Pack conversion.

Add `--estimate-representations` with that contract to compare compact source,
eager FP8, eager INT8-per-row, and one-expert hot-cache payload sizes. The
estimate reads metadata only and explicitly excludes container/runtime
overhead.

`qualify-deepseek-expert` is the bounded payload gate. It decodes one complete
real routed expert, checks every result bit against PyTorch's UE8M0 semantics,
and reports deterministic candidate INT8 hashes/error metrics without retaining
converted weights. NumPy and a PyTorch build with `float8_e8m0fnu` are required
on the qualification host.

`export-deepseek-expert` hashes one expert and writes a small descriptor for its
six `(shard, source offset, destination offset, bytes)` ranges. It copies no
weights. The CUDA admission gate gathers those ranges directly from the
immutable SafeTensors checkpoint into one bounded staging buffer.

`qualify-deepseek-shared` and `export-deepseek-shared` apply the same boundary
to the always-active FP8 shared expert. E4M3FN weights and UE8M0 128×128 block
scales are independently compared with PyTorch before the SM86 candidate hash
is accepted.

`export-deepseek-shared-set` validates the checkpoint once and atomically emits
the 43 small extent descriptors used for startup residency. It hashes about
1.08 GB of authoritative source payload but copies none of it.

## Install

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .\compiler

# Optional NumPy row-conversion acceleration
.\.venv\Scripts\python.exe -m pip install -e ".\compiler[fast]"
```

## Compile

```powershell
.\.venv\Scripts\python.exe -m compiler compile `
  --source C:\path\to\snapshot `
  --output C:\path\to\model.expert-pack `
  --source-id Qwen/Qwen3-Next-80B-A3B-Instruct `
  --source-revision <immutable-revision> `
  --adapter qwen3_next
```

## Validate

```powershell
.\.venv\Scripts\python.exe -m compiler validate C:\path\to\model.expert-pack
```

Output is written to `<output>.partial` and becomes the final directory only
after packs, manifest, hashes, and `COMPLETED` validate. Interrupted conversion
can resume only with identical source bytes and options:

```powershell
.\.venv\Scripts\python.exe -m compiler compile <same arguments> --resume
```

`--reclaim-source-shards` is destructive and deliberately not part of the
normal example. Use the P6 preflight/wrapper only after reviewing its exact
reclaim schedule and preserving a recoverable source checkpoint.

The dependency-free and optional NumPy paths produce the same quantization ABI.
See [Expert Pack v1](../docs/expert-pack-v1.md).
