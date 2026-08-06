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

`export-deepseek-routed-catalog` applies that contract to all 43 × 256 routed
experts in the main model. It emits one compact lookup table plus one extent
table; it does not duplicate the 147,169,738,752 source bytes. Every source
shard handle is opened once and all hashing passes through one fixed 8 MiB
buffer, so RAM does not grow with checkpoint size. Each complete expert is
SHA-256 hashed in its six-extent ABI order, and the directory is atomically
published only after all 11,008 records are present. The auxiliary MTP head is
intentionally outside this main-decode catalog.

`export-deepseek-mtp-set` authenticates that auxiliary layer as a separate,
atomic resource. It hashes the source extents for 19 typed tensors, 7 FP8
matrices, one shared expert, and 256 compact routed experts without copying the
3.59 GB payload. Publishing this descriptor does not enable speculation or
change base greedy decoding; it establishes the input contract for the future
MTP runtime.

```powershell
.\ops\windows\Export-DeepSeekMtpSet.ps1 `
  -Snapshot C:\path\to\deepseek-v4-flash-snapshot `
  -Output C:\path\to\deepseek-mtp-set-v1
```

```powershell
.\.venv\Scripts\python.exe -m compiler export-deepseek-routed-catalog `
  --source C:\path\to\deepseek-v4-flash-snapshot `
  --output C:\path\to\deepseek-routed-catalog
```

`qualify-deepseek-shared` and `export-deepseek-shared` apply the same boundary
to the always-active FP8 shared expert. E4M3FN weights and UE8M0 128×128 block
scales are independently compared with PyTorch before the SM86 candidate hash
is accepted.

`export-deepseek-shared-set` validates the checkpoint once and atomically emits
the 43 small extent descriptors used for startup residency. It hashes about
1.08 GB of authoritative source payload but copies none of it.

`qualify-deepseek-fp8-matrix` and `export-deepseek-fp8-matrix` expose the same
no-copy path for any validated dense FP8 matrix. The candidate ABI is generic
INT8-per-row and feeds the normal dense GEMV kernels rather than the expert
directory.

`qualify-deepseek-compact-matrix` is a read-only pre-kernel screen for smaller
dense representations. It compares symmetric 4/5/6-bit and FP4 E2M1 candidates
with FP16 block scales, reporting both reconstruction error and four
deterministic projection errors. A screened candidate is not a published ABI;
it must still pass the independent attention/full-model oracles.

`export-deepseek-dense-set` atomically emits descriptors for all 236
main-model FP8 matrices. The descriptor set is metadata only; the authoritative
4.78 GB of weights remains in the original shards.

`export-deepseek-hca` describes one attention or FFN hyper-connection site and
emits a small deterministic FP32 oracle. It does not copy the model tensors;
the runtime still gathers `fn`, `base`, and `scale` from their original
SafeTensors extents. The oracle covers stream collapse, the doubly-stochastic
Sinkhorn matrix, and stream expansion after a deterministic sublayer output.

`export-deepseek-typed-set` emits metadata-only descriptors for all 834
main-model BF16/F32/I64 tensors. The runtime preserves their source dtype and
streams tensors larger than its fixed staging slot while calculating one
incremental SHA-256 over the original byte order.

`export-deepseek-csa` describes one compressed-attention layer and emits a
decode oracle for its real BF16 compressor weights. Both checkpoint schedules
are supported: overlap pooling at ratio 4 and ordinary gated pooling at ratio
128. The bundle is a qualification fixture, not a replacement weight format.

`export-deepseek-attention-oracle` independently composes the SM86 dense ABI,
real typed weights, HCA, Q/KV transforms, sparse attention, and grouped output
projection for four sequential tokens. Layer 0 covers the checkpoint's pure
sliding-window path; layer 2 additionally closes and consumes the first
compressed group on its fourth token. The same oracle then applies the real FFN
HCA/norm and hash router, emitting expected top-6 IDs and weights for the four
attention outputs. For the fourth token it also describes the six routed plus
shared expert source extents and composes an independent full-block output
through the admitted SM86 ABI. The runtime still loads the authoritative full
checkpoint through its normal model-state/cache paths.

`export-deepseek-io-oracle` closes the main-model boundary around the untied
BF16 embedding and output head. It expands one real embedding row, applies the
four-stream HC head and final RMSNorm with the checkpoint's BF16 rounding
boundaries, then scans the 129,280-row head through a bounded row buffer. The
small output contains expected streams, logits, and greedy argmax; it does not
copy either roughly 1 GiB source tensor.

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
