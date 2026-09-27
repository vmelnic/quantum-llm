# Expert Pack v1

Status: implemented QPack container and record ABI, 2026-09-27.

## Artifact layout

The compiler converts a local SafeTensors checkpoint into an immutable
execution artifact. A completed artifact contains:

- `manifest.json` with source identity, file hashes, formats and program hash;
- `runtime-model.tsv` with the schema-3 executable program;
- dense and routed-expert QPack files plus indexes;
- tokenizer, generation and optional processor assets;
- `COMPLETED`, written only after validation succeeds.

Compilation writes a partial directory. Resume is accepted only when source
identity and every relevant option match. Publication uses a sibling candidate,
validates it, promotes by rename and validates the promoted identity again.

## Record layout

Dense and expert records use a 256-byte little-endian header, 4 KiB record
alignment and 256-byte payload-section alignment. Headers declare kind,
version, flags, quantization ABI, dimensions, byte ranges, source-name identity
and payload SHA-256. Readers do not infer an encoding from byte count.

Implemented quantization ABIs are:

| ABI | Encoding |
|---:|---|
| 0 | raw BF16, F32 or I64 tensor |
| 1 | symmetric INT8 per output row |
| 3 | FP4 E2M1 with one UE8M0 scale per 32 input values |
| 4 | ABI-3 payload with two-matrix ReLU2 expert layout |
| 5 | source-native NVFP4 E2M1 with block-16 E4M3FN scales and global factors |
| 6 | row-major OCP MXFP6 E3M2 with block-32 UE8M0 scales |

Raw dense records currently store BF16, F32 or I64 when the adapter assigns
those roles. Runtime state and intermediates have separate ABIs and are not
described by the stored-weight label.

## Program binding

The manifest authenticates `runtime-model.tsv` by path, size and SHA-256. The
program declares inputs, outputs, operations, tensor roles, capability ABIs,
routed-component geometry and optional exact-decode metadata. Provider binding
uses this program rather than the source adapter name.

## Validation

```powershell
.\.venv\Scripts\python.exe -m compiler validate `
  "$env:MODEL_ROOT\qwen3.8-27b-fp4"
```

Validation rejects unknown schemas or ABIs, unsafe relative paths, missing or
overlapping regions, invalid alignment, dimensions, hashes, tokenizer/program
identity and incomplete publication state.

A stored FP4 or NVFP4 payload is only the first qualification stage. A usable
artifact also requires independent source reconstruction, a provider that
executes the declared ABI and a real service-path gate.
