# Expert Pack v1

Status: current QPack container and record ABI, 2026-09-02.

## Container

Expert Pack converts SafeTensors sources into immutable records laid out for
native execution. A completed artifact contains:

- `manifest.json` with source identity, file sizes/hashes and format metadata;
- `runtime-model.tsv` with the executable schema-3 program;
- one or more dense/expert QPack files and indexes;
- tokenizer, generation and optional processor assets;
- `COMPLETED`, written only after validation succeeds.

The compiler writes a partial directory and resumes only when source identity
and options match exactly. Supported publication creates a sibling candidate
below `MODEL_ROOT`, validates it, then promotes by directory rename. The
service never points at partial or absent output.

## Record ABI

Version-1 dense and expert records have a fixed 256-byte little-endian header,
4 KiB record alignment and 256-byte payload-section alignment. Headers declare
record kind/version, flags, quantization ABI, dimensions, byte ranges,
source-name identity and payload SHA-256. Readers never infer an encoding from
record size.

Active encodings include:

| ABI | Encoding | Use |
|---:|---|---|
| 0 | declared raw tensor dtype | norms, metadata and unquantized organs |
| 1 | symmetric INT8 per output row | organs that explicitly declare it, including parts of DeepSeek's separate bundle |
| 3 | FP4 E2M1 plus UE8M0 scale per 32 input values | Qwen, Muse, Ornith, Qwen Flash and compatible SwiGLU records |
| 4 | ABI-3 payload with two-matrix ReLU2 expert layout | artifacts that declare routed ReLU2 rather than SwiGLU |
| 5 | source-native E2M1 plus E4M3FN block-16 scales and global factors | Mistral Small 4 NVFP4 W4A4 records |

Raw/F32/BF16 tensors and runtime intermediates retain their own declared
formats; an artifact must not be described simply as “4-bit”.

## Executable program

The artifact manifest binds `runtime-model.tsv` by identity and SHA-256. The
program declares inputs/outputs, components, layers, ordered operations,
capability versions, record references, geometry, encodings and optional
response/media/session contracts. Provider selection uses those declarations,
not the source adapter or advertised model name.

## Validation and qualification

```powershell
.\.venv\Scripts\python.exe -m compiler validate `
  "$env:MODEL_ROOT\qwen3.8-27b-fp4"
```

Validation fails closed for unknown fields/ABIs, unsafe paths, missing or
overlapping ranges, dimensions, alignment, hashes, tokenizer/program identity
and completion state. Python and native readers are the authoritative schema.

FP4/NVFP4 completion additionally requires:

1. payload exists in the advertised encoding;
2. source reconstruction/numerical quality passes;
3. the selected provider executes that ABI;
4. the real service path passes.

A metadata parser, an INT8 fallback or a self-referential kernel smoke does not
qualify an FP4 artifact.
