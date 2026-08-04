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
