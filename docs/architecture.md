# Architecture

Status: implemented architecture, 2026-10-21.

## Host boundary

The serving system is self-contained on one Windows host. One RTX 3090 owns
the active model program. The A520M/RTX 3090 and X99E/three 16 GiB Tesla P100
machines are independent rigs. Active serving does not depend on the P100
rig. The POSIX control machine synchronizes source and invokes operations;
it owns no model state.

The service has one hot provider slot. Requests may queue and exact sessions
may be parked, but model execution is serialized.

The user-authorized Swift/KAT GGUF service runs separately through
`ops/llama.sh` and llama.cpp, on the same 3090 with one active model. Its
registry, artifacts, task and API port are separate from the artifact VM;
see [llama.cpp CLI](llama-cli.md). It is not a universal VM qualification.
The user-authorized latest-stable update installs a clean Windows Release/CUDA
v0.6.0 build through `artifacts/llama-install.json`. Future launches and the
prepared launch resolve that executable; the active process was not restarted.

Swift's prepared policy restores one slot with 262,144 context positions,
MTP-10, full GPU layers and automatic fitting disabled. The user requested
this rollback after reporting severe slowdown with the two-slot fitted
policy. Activation remains with the user; no rollback inference was run.
KAT retains its separate one-slot policy.

Swift Heretic's prepared vision policy adds a pinned Q8 projector through an
optional artifact-declared registry object. Publication authenticates both
GGUF files; `--mmproj` and `--no-mmproj-offload` place the visual encoder on
CPU/RAM. The backbone, target KV and MTP remain on the GPU with context
262,144, one slot and MTP-10. Pi image input is enabled for the existing
`swift-heretic` ID. Deployment preparation passed file/argument gates without
restarting; live image quality and residency remain unmeasured.

The same separate GGUF registry also declares GPT-OSS-20B MXFP4 with one
slot, 131,072 native context positions, F16 target KV, native high reasoning
and no draft/projector. Generic optional reasoning policy avoids model-name
branches and preserves existing xhigh policies. The model passed publication
and launch-preparation gates; Pi is enabled in home settings without
changing defaults. No model start or live inference qualification was run.

## Independent Strata service

The separately authorized Strata service uses its own signed upstream engine,
registry, model pack, `QuantumLLM-Strata` task and API port 8082. It does not
change artifact-VM execution or llama.cpp. GPU-resident dense/state tensors
and cached experts cooperate with CPU computation of nonresident routed
experts. See [Strata CLI](strata-cli.md) for operations and serving policy.

The home Pi registry exposes it as `strata-3090/swift-flash-next-iq3-s`.
Pi configuration is in `~/.pi/agent/`; the removed project `.pi` directory
remains absent. Its actual Pi request timings and qualification limits are
recorded in [Benchmarks](benchmarks.md#actual-pi-request-with-embedded-code-2026-10-21).

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

An optional startup MTP proposal limit flows from `MODEL_MTP_DRAFT_DEPTH`
through the service and the generic worker launch policy. The runner and the
selected dense-MTP provider apply the same override to their in-memory
artifact descriptors before binding. The stored manifest/program/tensor
bytes stay unchanged. Missing policy keeps the artifact default, currently
four proposals for the Qwen QPack. Verifier rows, recurrent checkpoint strides,
attention scratch and acceptance counters follow the effective count rather
than a fixed five-row allocation. Provider workspace/context limits remain
fail-closed. Worker protocol 13 advertises the effective draft depth; an
explicit request requires protocol 13 and an exact echoed value at startup.
The `.env` proposal default applies only when the authenticated artifact
declares an adjustable sampled-MTP program. Other artifacts omit the worker
override; explicitly requested unsupported overrides still fail. Both start
and restart validate the artifact/policy before stopping the prior service.

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


### MLA/NVFP4 optimization candidate, 2026-10-07

The compressed BF16 MLA organ retains its page ABI and complete causal support.
For longer prefixes it scores eight positions per tile and reuses decoded
latent values in shared memory; short prefixes retain the serial kernel.
MLA-only operation sets omit standard attention staging and split arenas.
Mixed declared attention sets retain those workspaces. Selection depends on
operation capabilities, not model names or fixed layer counts.

Native NVFP4 routed prefill now supports the existing four-selection grouped
work layout. Each expert's packed weights/scales are decoded once per group,
while gate/up/down activation quantization remains selection-specific. BF16
projection/intermediate boundaries and ordered route aggregation are preserved.
Scalar decode and standard-FP4 grouping retain their existing paths.

This candidate passed offline native numerical/regression gates and subsequent
direct Mistral and Pi xhigh greetings. Qwen direct chat also passed on the
same clean-built common runner. Long-context coding qualification remains
pending; see [Benchmarks](benchmarks.md). It adds no MTP predictor, weight quantization
ABI, MLA quantized-cache ABI, external executor or additional serving host.

### Retained-RAM expert uploads, 2026-10-07

Pageable retained expert records reuse the existing fixed pinned staging pool
for H2D. A staging lease remains owned by the cache entry through asynchronous
upload completion, including cancellation. Already pinned host banks and
non-pinned pools preserve their existing upload paths. Pool pressure queues
uploads without overwriting an in-flight record; host copies, routing and
weight values remain unchanged. This adds no reserved memory or protocol ABI.
The exact clean-built source passed byte/ownership/cancellation tests and
Mistral/Pi/Qwen greetings. A trial that prioritized dropping device-backed
RAM duplicates slowed decode and the complete request sequence; it was
rejected and reverted. The existing victim ranking remains in use.


### Resident MLA context splits, 2026-10-07

Compressed MLA decode with at most eight rows and at least 4096 visible
positions uses 1024-position context splits. Each GPU block writes its local
softmax maximum, mass and unnormalized latent sum; a second GPU kernel merges
these in stable global softmax coordinates and applies the original value
projection once. Complete causal support, BF16 latent pages, expert routing
and the weight ABI remain unchanged. F32 summation order differs and is
qualified against an independent materialized-K/V FP64 oracle.

Dedicated optional scratch is bounded by eight rows, declared head count,
latent rank and configured context. At the current 32-head/256-rank/262K
geometry it reserves 64.5 MiB, reused between layers. Larger prefill batches,
short prefixes and callers without scratch retain the tiled/serial paths.
The launch structure layout and artifact/wire versions do not change. The
kernel gate passes at actual geometry through 262K. The same clean-built
runner also passes direct Mistral/Pi greetings, a Pi request with 4408 prompt
tokens exercising split decode, Qwen direct chat and process/GPU cleanup.
Mistral is left running on the qualified candidate. Whole-model populated
context, capacity and coding throughput remain separate qualification gates;
real-client latency is still poor. See [Benchmarks](benchmarks.md).
