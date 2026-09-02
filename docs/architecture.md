# Architecture

Status: implemented architecture and boundaries, 2026-09-02.

## Scope

Quantum LLM serves validated executable artifacts from one self-contained
Windows host with one RTX 3090. The control host may synchronize source and
invoke operations, but it is not a model-state owner. There is no remote expert
executor, distributed fallback or external coordinator in the serving path.

The runtime solves two different placement problems:

- dense/hybrid models have hot matrices and growing attention state; moving
  either through PCIe or NVMe every decode call is a capacity fallback, not a
  throughput solution;
- sparse MoE models touch only routed experts; unselected expert pages are cold
  and may be retained or demand-paged without changing exact top-k semantics.

## Artifact-driven system

```text
official checkpoint
       |
       | strict source adapter
       v
candidate below ${MODEL_ROOT}
  manifest + hashes + COMPLETED
  tokenizer/generation/processor assets
  QPack or compact payloads
  runtime-model.tsv
       |
       | validate and promote
       v
common Python HTTP service
  authentication, limits, templates, streaming, admission
       |
       v
common native VM runner
  program validation and capability binding
       |
       +------------------------------+
       |                              |
       v                              v
SM86 FP4/NVFP4 provider       compressed sparse provider
Qwen/Flash/Mistral/Muse/      DeepSeek dense/shared state
Ornith operations             plus exact expert paging
```

`runtime-model.tsv` declares ordered operations, tensor roles, geometry,
encodings and required capabilities. Providers advertise the declarations they
can execute. Common service/runtime code does not select Qwen, Mistral, Muse,
Ornith, DeepSeek, fixed layer counts or family tensor paths. Upstream names are
handled only by strict source adapters; genuinely new mathematics or encoding
requires a provider capability and numerical gate.

The physical containers are not falsely unified. Expert Pack v1 stores QPack
records used by the FP4/NVFP4 provider. DeepSeek uses a separately authenticated
compact layout suited to its source representation and paging geometry. Both
converge at the executable-program, logical-page and provider boundaries.

## Model organs and placement

| Organ | Access pattern | Placement rule |
|---|---|---|
| embeddings | selected rows | gather/stage selected rows; never scan per token |
| dense projections/MLP | every target call | remain beside fastest compute |
| recurrent state | fixed-size, every next token | remain in the active GPU slot |
| full-attention K/V | every populated retained position | keep complete exact state beside attention or accept the measured transfer cost |
| sparse-attention index | every sparse-attention query | keep compact exact index resident when possible |
| MoE router/shared experts | every routed layer | resident and exact |
| routed experts | selected top-k only | immutable logical pages eligible for RAM/VRAM/NVMe placement |
| vocabulary head | every generated token | resident or explicitly accounted hot traffic |
| draft/MTP state | only when verification policy uses it | allocate conditionally and account separately |

## Active artifacts

| Alias | Program and representation | Request state | Callable scope |
|---|---|---|---|
| `qwen` | 64-layer hybrid model; 14,775,390,208-byte QPack; FP4 E2M1/UE8M0 block-32 matrices | progressive exact F16 KV for 16 full-attention layers plus recurrent state; bounded disposable VRAM mirror over authoritative host pages | text, reasoning, tools and image understanding |
| `qwen-flash` | 48 layers; 36 Gated DeltaNet, 12 QSA, Hyper, PLE, 512 experts/layer, exact top-10 FP4 routing | recurrent state plus progressive exact F16 QSA K/V; compact index retained on device where declared | text/reasoning/tools; quality and long-context performance remain unqualified |
| `mistral` | source-native block-16 E2M1/E4M3FN NVFP4 W4A4 MoE plus BF16 organs | artifact-declared BF16 MLA latent KV pages | text/reasoning/tools; auxiliary multimodal records are not callable support |
| `muse` | 52-layer dense FP4 text model with global and exact 2,048-token sliding attention | exact F16 global pages plus cyclic exact F16 sliding windows; artifact maximum 131,072 | text only; preserved vision records are auxiliary |
| `ornith` | 40-layer hybrid model; 30 recurrent, 10 full attention, 256 experts/layer, exact top-8 standard FP4 routing | progressive exact F16 KV; complete 17,154,703,360-byte expert pool can live in the host bank | text/reasoning/tools |
| `deepseek` | resident dense/shared organs; 43 routed layers x 256 experts; exact top-6 compact FP4 pages | provider-owned artifact KV and request state; no exact checkpoint/rewind/session-retention capability | text/reasoning/tools through exact demand paging |

An artifact limit is an admission ceiling, not allocated context. Muse is
clamped to 131,072; the other active artifacts currently advertise 262,144.
Neither value proves that the context was populated or interactive.

## Dense and attention state

Qwen's target F16 KV is allocated in 256-token pinned-host pages. Host pages
are authoritative. While the populated prefix fits a bounded exact F16 device
mirror, attention consumes the mirror and avoids a per-token PCIe scan. Parking
releases the disposable mirror; restore may rebuild it. On overflow the mirror
is released and the provider falls back atomically to bounded exact host-KV
staging. This preserves values but cannot meet the saturated-context throughput
target because all full-attention KV remains hot.

Muse and Mistral derive KV bytes and dtype from their artifact programs rather
than a Qwen-shaped estimate. Muse's global and sliding windows remain distinct;
Mistral's latent cache remains BF16. Qwen Flash stages only model-selected exact
QSA payload while keeping its exact index device-side under the balanced policy.
These are model-declared semantics, not runtime sparsification.

## Sparse expert state

Every expert is identified by artifact namespace, component, layer, expert ID
and encoding ABI. Acquisition has one state machine:

```text
ABSENT -> SSD_LOADING -> RAM_READY -> GPU_UPLOADING -> VRAM_READY
   +------------------------- failure ----------------------> FAILED
```

Demand outranks bounded warm/prefetch work. Leases prevent eviction while CPU
or CUDA work consumes a page. RAM uses probationary/protected retention; VRAM
uses transient/protected classes. Missing selected experts are never treated as
zero, and aggregation preserves router order and weights.

Ornith's routed pool fits the configured 48 GiB host budget, so a warmed request
can execute without NVMe reads. DeepSeek's 147,169,738,752-byte routed pool does
not fit RAM, so cold routes still require storage. RAM-ready DeepSeek misses may
be split between exact CPU execution and GPU fills using measured costs; CPU
and GPU partial outputs are merged exactly. This makes the model executable
beyond RAM+VRAM but does not remove novel-route bandwidth cost.

The validated common routed-VRAM budget is currently 12 GiB. It is a ceiling,
not an upfront allocation. A 13 GiB DeepSeek profile fails preflight after
immutable allocations, workspace and the 1 GiB emergency reserve, so it is not
a supported common configuration.

## Sessions and concurrency

The FP4/NVFP4 provider advertises exact retained sessions. A retained request
owns populated KV pages and a continuation blob in host RAM while releasing the
single hot GPU slot. Resume feeds only the suffix when the client prefix
matches; unchanged prefixes are valid zero-delta resumes. Failed suffix feed or
cancellation rolls back to the last client-echoable committed prompt.

This is continuity, not parallel decode. The Python command channel and hot
provider slot are serialized. Multiple Pi processes may queue and retain short
histories, but several actually populated 262K F16 contexts compete for real
RAM/KV bytes. `MODEL_WORKER_CAPACITY` does not create concurrent GPU execution.

DeepSeek does not advertise exact checkpoint/rewind/retain/drop. The service
must not send those commands. Cancelling a long DeepSeek prefill may leave the
worker unhealthy; readiness must be checked and the service restarted before
another gate.

## Response and API boundary

The Python service owns authentication, request limits, official chat
templates, media normalization, streaming and admission. One artifact-declared
response parser instance owns reasoning, visible text and structured tool calls
from the first token through finalization. Pi receives standard API events and
does not parse model-native XML or bracket syntax.

The native worker owns artifact state, execution, sampling and tier telemetry.
Unsupported session, vision, tool or operation behavior fails explicitly.
`/health`, `/ready`, `/model-info` and cached `/metrics` must remain bounded
during long inference; `ready=true` does not mean caches are warm or an SLO is
met.

## Hardware and failure boundary

The supported build targets CUDA SM86 on one RTX 3090. There is no active
multi-GPU allocator, P2P transport or Pascal provider. The server provides
bearer authentication but not TLS, public rate limiting or a hardened edge.
Stop must terminate Python and native descendants and release model VRAM.
