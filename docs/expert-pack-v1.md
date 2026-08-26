# Expert Pack v1

Status: current QPack storage contract, 2026-08-26.

## Purpose

Expert Pack converts a SafeTensors checkpoint into immutable, independently
validated records that are already laid out for native execution. The physical
container is not the VM: `runtime-model.tsv` declares geometry, operations and
provider ABIs, while QPack supplies the referenced bytes.

## Container

A completed artifact contains at least:

- `manifest.json`, source identity and hashes;
- `runtime-model.tsv`, bound by length and SHA-256 in the manifest;
- one or more dense/expert QPack files and indexes;
- tokenizer, generation and optional processor assets copied from the source;
- `COMPLETED`, written only after independent validation succeeds.

The compiler writes `<output>.partial`, supports resume only with identical
source bytes/options, then atomically publishes the final directory. The
supported publication workflow creates a candidate below `MODEL_ROOT` and
uses `Promote-ModelArtifact.ps1` for the final rename/rollback boundary.

## Record ABI

Records use a fixed 256-byte little-endian header. File records are 4 KiB
aligned and payload sections are 256-byte aligned. Headers declare kind,
version, flags, quantization ABI, dimensions, offsets, lengths, source-name
identity and payload SHA-256. Readers do not infer an ABI from record size.

The deployed Qwen profile is:

| Field | Value |
|---|---|
| profile | `fp4-e2m1-ue8m0-block32-v1` |
| ABI | 3 |
| matrix values | FP4 E2M1, two nibbles per byte |
| scale | one UE8M0 code per 32 input values of each output row |
| scale range | codes 1..254; zero and NaN code 255 are never emitted |
| accumulation | provider-owned FP32/native packed CUDA path |

Raw tensors, router/norm state and runtime intermediates retain their declared
formats; describing the artifact as simply “4-bit” would be incomplete. The
current Qwen3.8 artifact has a 14,775,390,208-byte `dense.qpack` with 1,199
records. It is not an INT8 artifact.

## Executable program

`runtime-model.tsv` schema 3 declares:

- program inputs/outputs and tensor ABIs;
- components, layers and ordered operations;
- required kernels/capabilities and their versions;
- record references, geometry and encoding names;
- optional exact-decode, response and multimodal contracts.

The common runner binds declarations to providers by capability and encoding,
not by model name or architecture ID. Source adapters may still be
family-aware because upstream tensor names/configuration are not universal.

## Validation

```powershell
.\.venv\Scripts\python.exe -m compiler validate `
  "$env:MODEL_ROOT\qwen3.8-27b-fp4"
```

Validation is fail-closed for unknown fields/ABIs, missing records, unsafe
paths, dimensions, alignment, byte ranges, hashes, tokenizer assets, program
identity and completion state. The Python validator and native readers are the
authoritative schema; the obsolete static JSON schema was removed because it
accepted old INT8/family layouts but rejected the deployed schema-3 FP4
artifact.

FP4 completion additionally requires the source quality gate and the real
SM86 provider path. A successful parser or INT8 execution does not qualify the
FP4 artifact.
