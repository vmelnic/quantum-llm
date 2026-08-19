# Architecture

Status: current runtime architecture as of 2026-08-19. The active target is
Qwen3.8-27B FP4; DeepSeek-V4-Flash remains the sparse demand-paged
compatibility target. Remaining work is tracked in
[Current state and next work](moe-vm-next.md).

## Objective

Run artifact-declared transformer programs through one lifecycle, API and
native runner, whether the artifact is dense or sparse. The common path must
not select a model family, fixed layer count or tensor naming convention.

For sparse models, selected experts are immutable logical pages resolved from
VRAM, RAM or storage without requiring the whole model to fit either memory
tier. Exact top-k and routing weights are preserved. For dense models such as
Qwen3.8-27B, the same program/provider boundary executes a resident FP4 pack;
expert paging is not involved.

## System view

```text
official checkpoint
       │ strict source adapter: names/metadata → tensor roles and capabilities
       ▼
transactional artifact below MODEL_ROOT
  ├── manifest, hashes, tokenizer and generation config
  ├── one or more typed/FP4 packs
  └── runtime-model.tsv: geometry, ordered operations and required ABIs
       │
       ▼
OpenAI HTTP → admission → official chat template → retained request state
       │                                             │
       │                                             ▼
       │                                  compiled operation program
       │                                             │
       │                          capability/geometry provider binding
       │                                             │
       └──── streaming text/reasoning/tools ◄─ token selection ◄─ logits
                                                     │
                           dense FP4 CUDA ───────────┤
                           sparse exact router ─► expert-page store
                                                    ├─ VRAM
                                                    ├─ RAM/H2D
                                                    └─ SSD/IOCP
```

## Artifact boundary

An adapter may understand upstream family-specific tensor names and config
semantics. Its output is provider-neutral:

- tensor roles and shapes;
- model geometry and ordered operations;
- numeric/quantization ABI for every record class;
- tokenizer and response grammar assets;
- immutable source, pack, index and program hashes.

Common compiler validation, lifecycle, API, program execution and placement do
not branch on Qwen or DeepSeek. A new model using supported operations and
encodings needs an adapter only. Genuinely new math, encoding or geometry needs
a capability provider and an independent numerical gate.

Publication is transactional. Compiler output is completed and validated in a
candidate directory before it replaces `${MODEL_ROOT}/<stable-name>`. `out/`
contains builds and experiments, never service-ready model artifacts.

## Compiled program and providers

`runtime-model.tsv` schema 3 is the current ordered execution contract. Startup
parses it, binds each required capability to a provider and validates its
geometry/ABI once. Token execution uses compiled handles rather than model
names or tensor-path lookups.

Current provider sets include:

- Qwen3.8 dense SM86 operations: embedding, RMSNorm, gated GQA full attention,
  gated linear/recurrent attention, dense gated MLP, MTP boundary, vocabulary
  head and sampling;
- DeepSeek sparse SM86 operations: its declared attention/CSA/HCA schedule,
  router programs, shared/routed experts, MTP draft/verify and exact
  demand-paged expert placement.

Provider constraints such as supported head dimension or encoding are explicit
startup failures. They are not disguised model-family checks.

## Qwen3.8 execution

The published Qwen3.8 artifact declares 64 text layers: 16 full-attention and
48 linear-attention layers. Its dense pack has 1,199 records and
14,775,390,208 bytes. Matrix payloads use FP4 E2M1 with UE8M0 block-32 scales
(quant ABI 3); raw ABI-0 records and float32 norms/metadata remain separately
declared. The model has no routed experts.

The provider keeps immutable weights in the device-resident execution tier and
allocates mutable request state under explicit credits. Full-attention K/V is
paged; short requests commit only the pages they touch. Linear-attention
recurrent/conv state is position ordered. Retained append-only sessions reuse
both state classes and prefill only the new suffix.

The reference service reserves 5,120 MiB of aggregate KV capacity and one
worker slot. A real 262,016-token prompt plus 128 generated tokens completed.
That proves allocation and execution through all 262,144 positions, not the
speed SLO: TTFT was 5,423.422 seconds and decode after the first token was
approximately 3.24 tok/s.

## DeepSeek expert-page execution

An expert is addressed by artifact namespace, routed component, layer and
expert ID. Its bounded state machine is:

```text
ABSENT → SSD_LOADING → RAM_READY → GPU_UPLOADING → VRAM_READY
   └──────────────────────── failure ───────────────────────► FAILED
```

One in-flight acquisition exists per key. Typed host/device leases and CUDA
events prevent eviction while an exact route uses a record. Demand is strictly
prioritized over bounded prefetch/warm work. Protected/probationary RAM,
transient/protected VRAM, census warm-up and request-attributed telemetry are
cache policies below the same logical-page contract.

The scheduler publishes the complete selected union. A partial top-k never
executes: missing pages suspend the FFN until the exact union is ready or fail
the request explicitly. Model size may exceed RAM+VRAM because only dense state
and bounded active expert pages must fit those tiers.

## HTTP and response protocol

The Python service owns tokenization, bounded admission, streaming, session
matching and OpenAI wire objects. The native worker owns model state, compiled
program execution and token selection. Their line protocol is versioned; the
current worker advertises protocol 8 and actual provider capabilities.

Qwen3.8 uses the official tokenizer chat template and Transformers response
parser. Default reasoning effort is `xhigh`. Sampling defaults come from the
artifact generation config and are executed by the provider. Official XML
function calls are converted into normal Chat Completions/Responses tool-call
objects; tool results re-enter the official conversation template.

DeepSeek has no Transformers chat template and therefore uses only its pinned
official encoder adapter. Artifacts without a declared response grammar reject
tools. This is capability selection, not a common-path model-name branch.

## Admission and failure model

Budgets for request slots, queue depth, body bytes, KV pages, RAM cache, VRAM
cache and staging are independent. Admission reserves prompt plus requested
output credits before generation. Exhaustion returns an explicit bounded error
instead of depending on pagefile pressure or CUDA OOM.

- malformed/mismatched model data prevents readiness;
- unsupported capabilities or geometry fail at startup/request parsing;
- a client disconnect cancels before another decode step is admitted;
- worker failure fails dependent requests explicitly;
- session state is published only after complete successful allocation;
- the stop path kills the Python/native descendant tree and must release GPU
  ownership;
- source checkpoints are not deleted by normal compile/publish/deploy flows.

## Observability

`/model-info` publishes artifact identity, program/provider capabilities,
limits, placement, KV geometry and worker protocol. JSONL request telemetry
contains prompt/generated token counts, server TTFT, wall time and provider
phase counters without prompt content.

For reasoning models, first visible content is not first generated token.
Performance claims use server TTFT and generated-token counts consistently;
the terminal client's visible-content timing is not authoritative when hidden
reasoning is enabled.

## Remaining architectural boundary

The common program/provider execution path is implemented for the current
dense FP4 operation set, while DeepSeek retains genuinely distinct sparse
math/storage providers. Physical pack formats are not yet one universal
container: dense Expert Pack and DeepSeek compact/bundle representations enter
through adapters. That divergence must remain explicit.

There is no external owner or remote execution deployment. A future remote
provider, if ever justified, would move activations to expert compute and obey
the same all-or-nothing route lease semantics; it is not part of the current
goal.
