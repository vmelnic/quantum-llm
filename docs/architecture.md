# Architecture

Status: implemented architecture, 2026-09-27.

## Host boundary

The serving system is self-contained on one Windows host. One RTX 3090 owns
the active model program. Two local P100 cards can execute compatible routed
expert pages, but no active alias depends on them and measured paths did not
improve throughput. The POSIX control machine synchronizes source and invokes
operations; it owns no model state.

The service has one hot provider slot. Requests may queue and exact sessions
may be parked, but model execution is serialized.

## Artifact-driven execution

```text
checkpoint
  -> source adapter
  -> candidate artifact
       manifest.json
       COMPLETED
       runtime-model.tsv
       QPack records and indexes
       tokenizer/generation/processor assets
  -> Python service
  -> native VM runner
  -> capability-matched provider
```

`runtime-model.tsv` uses `expert-runtime-model-v1`, schema 3. It declares the
model geometry, ordered operations, tensor bindings, routed components,
encodings and required kernel capabilities. The native runner loads the
authenticated program, finds exactly one compatible provider module and
compiles the operation sequence. Missing or ambiguous bindings fail startup.

Common service and VM code do not select an architecture from a model name,
fixed layer count or upstream tensor path. Source adapters translate upstream
layouts. A new mathematical operation or encoding requires a declared
capability, an ABI and an independent numerical gate.

## State and placement

| State | Access pattern | Implemented placement rule |
|---|---|---|
| embeddings | selected rows | gather selected rows; avoid full scans per token |
| dense projections and MLP | every target call | remain beside the primary compute path |
| recurrent state | every next token | remain in the active GPU slot |
| full-attention KV | every retained position | use device-resident packed state or exact host pages with measured transfer cost |
| sparse-attention index | every sparse query | retain the compact exact index on device when declared |
| router and shared experts | every routed layer | resident and exact |
| routed experts | selected top-k only | immutable pages eligible for RAM, VRAM and NVMe caching |
| vocabulary head | every generated token | resident; scratch rows grow only on demand |
| MTP state | speculative requests only | allocate and account separately |

The common fixed routed-VRAM ceiling is 12 GiB. It is a cache ceiling, not an
up-front allocation. The implemented `fit` policy may derive an
artifact-specific expert cache only before any expert page is admitted, after
reserving declared future state and 1 GiB emergency headroom. Current aliases
select `fixed`.

## Qwen attention state

The exact path stores authoritative F16 KV in 256-token pinned-host pages. A
bounded disposable device mirror serves prefixes that fit. When it cannot fit,
the provider releases the mirror and stages exact host KV. Values remain F16;
the transfer path cannot meet the maximum-context throughput target.

The operational Qwen aliases select `q4-f16-per-head` (Q4H). Each 256-value K
or V head record stores 128 signed-Q4 bytes and one F16 scale. A K/V pair uses
260 bytes, so target KV occupies 4.0625 GiB at 262,144 positions. Q8 MTP pages
raise the combined target-plus-draft state to 4.566406 GiB. The batch-five
attention kernel consumes the packed representation directly and expands only
transient integer operands inside a tile. Q4H is lossy and does not satisfy the
exact-F16 target.

The source also implements other diagnostic KV formats. They are selectable
only through an explicit alias or worker policy and are not active defaults.

## Program providers

The runner currently links three provider modules selected by capability:

- a dense/MoE provider for declared dense-attention and routed programs;
- a dense FP4/NVFP4 provider for compatible declared operation sets;
- a hybrid DeltaNet/MoE provider for compatible recurrent programs.

Provider names are diagnostic. Selection uses the artifact's required kernel
set and ABI ranges. The service validates the returned context, KV, placement,
MTP, sampling and session contract before reporting readiness.

## Routed experts

Expert identity includes artifact namespace, component, layer, expert ID and
encoding ABI. Acquisition follows one state machine:

```text
ABSENT -> SSD_LOADING -> RAM_READY -> GPU_UPLOADING -> VRAM_READY
   +------------------------- failure ----------------------> FAILED
```

Demand outranks warm or prefetch work. Leases prevent eviction during CPU or
CUDA execution. A selected expert is never replaced with zero, and aggregation
preserves router order and weights. Standard-FP4 prefill may group rows by
expert ID to reuse weights while preserving the exact route.

An optional secondary-device executor can own compatible routed pages on the
P100s. Activations and results cross pinned host memory because peer access is
unavailable. It is a capacity path, not an active performance dependency.

## Sessions

Session behavior is a provider capability. A retaining provider can park
populated KV and continuation state in host RAM, release the hot GPU slot and
resume from an exact prompt/media prefix. Only the suffix is prefilled. A
failed suffix feed rolls back to the last client-visible committed prefix;
partial assistant output is not retained after cancellation.

Qwen also implements persistent restart/resume. It stores immutable 4 MiB
SHA-256 chunks and a transactional manifest below
`${MODEL_ROOT}/.session-cache/<artifact>`. The manifest binds artifact,
tokenizer, template, codec, media identity and token prefix. Startup loads only
metadata. Matching a prefix loads the state lazily. NVMe is used for restore,
never as an active per-token KV tier.

## Service boundary

The Python service owns authentication, bounded request parsing, official chat
templates, media normalization, streaming, queue and context admission. One
artifact-declared parser instance converts native output into reasoning,
visible text and structured tool calls from the first delta through final
output.

The native worker owns artifact state, execution, sampling and tier telemetry.
The line protocol supports begin, feed, step, cancel, status and shutdown;
session commands are sent only when the provider advertises them.

`/health`, `/ready`, `/model-info` and `/metrics` use bounded snapshots.
`ready=true` means the worker is alive and accepting admission; it does not
mean a cache is warm or a latency objective is met.

## Failure boundary

Unknown manifest fields, unsafe paths, record versions, operations, encodings,
capabilities, dimensions or hashes fail closed. Startup also fails if the
provider contract disagrees with launch policy. Stop must terminate Python and
native descendants and return model allocations on every CUDA device.

The server provides bearer authentication. TLS and public rate limiting require
a trusted external edge.
