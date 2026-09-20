# Architecture

Status: implemented architecture and boundaries, 2026-09-20.

## Scope

Quantum LLM serves validated executable artifacts from one self-contained
Windows host. One RTX 3090 owns the primary program. Two local P100s can own
compatible routed-expert pages, but current end-to-end measurements do not
justify enabling that path for speed. The control host only synchronizes and
invokes operations; there is no remote model-state owner or fallback.

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
       |                              |
       +--------------+---------------+
                      v
          optional in-process SM60/61
          routed-expert executor
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
| `qwen` / `qwen-f16` / `qwen-abliterated` | 64-layer hybrid model; 14,775,390,208-byte QPack; FP4 E2M1/UE8M0 block-32 matrices; the abliterated alias selects separately published third-party weights with identical geometry | `qwen` and `qwen-abliterated` select resident `q4-f16-per-head` target KV; `qwen-f16` selects progressive exact F16 KV with a bounded disposable VRAM mirror | text, reasoning, tools and image understanding; the abliterated checkpoint has direct-chat and minimal-Pi wiring qualification only; per-head Q4 quality remains a user-run real-project Pi gate |
| `qwen-flash` | 48 layers; 36 Gated DeltaNet, 12 QSA, Hyper, PLE, 512 experts/layer, exact top-10 FP4 routing; scalar decode may execute the selected standard-FP4 experts on local P100s | recurrent state plus progressive exact F16 QSA K/V; compact index retained on device where declared | text/reasoning/tools; P100 execution is exact but slower than the primary path in the measured short chat; quality and long-context performance remain unqualified |
| `mistral` | source-native block-16 E2M1/E4M3FN NVFP4 W4A4 MoE plus BF16 organs | artifact-declared BF16 MLA latent KV pages | text/reasoning/tools; auxiliary multimodal records are not callable support |
| `muse` | 52-layer dense FP4 text model with global and exact 2,048-token sliding attention | exact F16 global pages plus cyclic exact F16 sliding windows; artifact maximum 131,072 | text only; preserved vision records are auxiliary |
| `ornith` | 40-layer hybrid model; 30 recurrent, 10 full attention, 256 experts/layer, exact top-8 standard FP4 routing | resident `q4-f16-per-head` target KV and the fixed common expert cache; the complete 17,154,703,360-byte expert pool can live in the host bank | text/reasoning/tools; per-head Q4 service wiring is qualified separately from model quality |
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

K1 (`fp4-e2m1-ue8m0-block32-key-outlier1`) remains a separate implemented
lossy KV policy:
values use block-32 FP4; keys add one FP16 largest-magnitude correction per
block. At Qwen's 262,144-position ceiling it reduces target KV from 16 GiB to
4.75 GiB, or 5.015625 GiB including MTP pages. It is no longer selected by an
active alias. K1 is never reported as exact F16.

`q4-bfp16-block32-key-outlier1` is a second, explicitly lossy target-KV
policy. Each 256-value record stores signed Q4 payload, one FP16 base, and
four-bit block-32 exponents; K also stores one aligned FP16 outlier per block.
The K+V record pair is 300 bytes, so Qwen target KV occupies 4.6875 GiB at
262,144 populated positions and 5.191406 GiB with its Q8 MTP pages. The
batch-five decode kernel reads the head-major packed planes once per K/V tile,
expands only transient INT8 Tensor Core operands in shared memory, and evaluates
all five causal queries there. It does not materialize a decoded F16 cache.
Large cold-prefill batches retain bounded transient BF16 staging for the
vendored FlashAttention path; that staging is not the decode path or retained
KV representation. It remains an implemented research format but is not
selected by an active alias.

`q4-f16-per-head`, shortened to **Q4H**, is the lower-overhead packed policy selected for the
40--50 useful-token/s experiment. K and V each store 128 signed-Q4 code bytes
and one FP16 scale for the complete 256-value head, for a 260-byte K+V record
pair. Qwen target KV therefore occupies 4.0625 GiB at 262,144 positions, or
4.566406 GiB including the unchanged Q8 MTP pages. The same batch-five kernel
loads each packed K/V tile once for all speculative queries and expands it only
to transient signed-INT8 Tensor Core operands. It never builds an F16 cache.
This policy is more lossy than block-floating Q4. It is the default only for
compatible 256-dimensional GQA artifacts: Qwen, Qwen Abliterated and Ornith.
Passing throughput does not qualify fidelity or exact-F16 semantics.

Sampling defaults and the thinking runaway ceiling are artifact policy,
not model-name branches. The single generic sampling-policy contract carries
the two explicit thinking/non-thinking profiles and an optional
`maximum_thinking_tokens`; its schema label is metadata, not a runtime behavior
switch. The server fails closed on invalid fields and clamps only thinking
requests. The
all three operational Q4H artifacts use `presence_penalty=0.5` in both
sampling profiles. Qwen and Qwen Abliterated also declare a 32,768-token
thinking-output ceiling after a real long-context Pi session entered unbounded
lexical repetition. The initial 1.5 penalty on those two artifacts caused
visible Romanian lexical and grammatical degradation and was rejected. Their
service context remains 262,144 positions; non-thinking output remains bounded
only by the populated prompt and that context. Other artifacts retain their
own declared or tokenizer defaults.

For compact non-F16 Qwen attention, prefill uses the vendored official
FlashAttention implementation over artifact-sized paged segments. Gated
DeltaNet recurrent prefill uses a numerically gated value-major warp update,
and the complete compact path uses 1,024 workspace rows. F16 host-authoritative
prefill retains the 512-row bounded staging path. These choices are capability-
and geometry-driven; common service code does not branch on the Qwen family.

Muse and Mistral derive KV bytes and dtype from their artifact programs rather
than a Qwen-shaped estimate. Muse's global and sliding windows remain distinct;
Mistral's latent cache remains BF16. Qwen Flash stages only model-selected exact
QSA payload while keeping its exact index device-side under the balanced policy.
These are model-declared semantics, not runtime sparsification.

Dense-provider scratch capacity is not synonymous with resident allocation.
The vocabulary head keeps one logits row for normal service sampling and grows
only when an invocation actually requests a multi-row head. Decoded FP4 weight
staging is allocated only for eligible batched single-tile operations, reused
within that sequence, and released on success, cancellation or failure. A
multi-tile maximum-context prefill never allocates that staging arena. Runtime
telemetry reports capacity and currently resident bytes separately.

Exact-decode ABI 2 is an artifact-declared experimental capability, not an
exact-F16 qualification. It rolls one source MTP layer recurrently for a declared
depth of three or four, keeps the target vocabulary head unchanged, and may
limit only the draft head to an artifact-declared prefix. Its proposer K/V is
signed Q8 with one FP16 scale per token/head/vector; target K/V retains the
selected target codec. Before a speculative cycle it checkpoints recurrent
and proposer state, verifies the guaranteed token plus all draft positions in
one target batch, applies exact sampled `p/q` rejection and residual
correction, then commits or restores/replays the accepted prefix. The compact
attention kernel loads each K/V tile once and evaluates all speculative queries
inside that tile. This capability remains available for research artifacts,
and the operational K1 experiment uses it. The target pass exports a recurrent
checkpoint after every verified row, so exact `p/q` rejection restores the
selected row without a second target execution. The measured 32K/128K gates
improved to 26.19/23.89 useful tok/s, but the 262K two-point extrapolation is
21.39 tok/s; it is therefore implemented and deployable, not a 40-50 tok/s
completion claim.

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

The routed-VRAM policy is alias-declared and provider-neutral. `fixed` keeps the
configured common ceiling. `fit` is a startup-only policy: after immutable
dense tensors and fixed workspaces have been allocated, and before any expert
has entered VRAM, the provider computes

```text
available routed bytes = CUDA free after fixed allocations
                       - maximum declared KV pages
                       - maximum disposable KV mirror
                       - possible logits-workspace growth
                       - emergency reserve
effective cache       = min(expert-pool bytes, available routed bytes)
```

The cache is reconfigured only while it has zero admitted device bytes; later
resize attempts fail closed. Future request-owned allocations remain reserved
even though context is allocated progressively. The effective byte ceiling is
reported in `/model-info`, so a configured context limit is never mistaken for
free cache capacity.

Ornith's routed pool fits the configured 48 GiB host budget, so a warmed request
can execute without NVMe reads. DeepSeek's 147,169,738,752-byte routed pool does
not fit RAM, so cold routes still require storage. RAM-ready DeepSeek misses may
be split between exact CPU execution and GPU fills using measured costs; CPU
and GPU partial outputs are merged exactly. This makes the model executable
beyond RAM+VRAM but does not remove novel-route bandwidth cost.

An explicitly configured compatible routed component can execute selected
experts on the two P100s. Routing, stable aggregation, attention, recurrent
state and all other organs stay on the RTX 3090. Exact F32 activations/results
cross pinned host memory because CUDA peer access is unavailable; expert
weights remain in fitted, slot-aligned P100 arenas. Every selection validates
artifact identity and its source, execution and encoding ABIs. This is an
optional expert placement, not a dense-weight sharder or general multi-GPU
allocator.

The validated common `fixed` primary-RTX routed-VRAM budget is currently 12
GiB. It is a ceiling, not an upfront allocation. A 13 GiB primary DeepSeek
profile fails preflight after immutable allocations, workspace and the 1 GiB
emergency reserve, so it is not a supported common configuration. Fitting is
therefore an explicit per-alias policy rather than a global budget increase.
The former `ornith-k1` profile fitted 15.9375 GiB after accounting for its
smaller maximum K1 state. The optional P100 arena is separate and
reserves its own fitted capacity at startup without loading expert payloads
into those slots.

## Sessions and concurrency

The FP4/NVFP4 provider advertises exact retained sessions. A retained request
owns populated KV pages and a continuation blob in host RAM while releasing the
single hot GPU slot. Resume feeds only the suffix when the client prefix
matches; unchanged prefixes are valid zero-delta resumes. Failed suffix feed or
cancellation rolls back to the last client-echoable committed prompt.

For providers that advertise `session_persistence`, the same complete parked
state can survive a service restart. Qwen writes immutable 4 MiB SHA-256 chunks
and a transactional manifest below
`${MODEL_ROOT}/.session-cache/<artifact>`. The manifest binds the artifact,
tokenizer/template, selected KV codec, media identity and exact token prefix.
At boot only the metadata index is loaded; no session RAM or GPU pages are
reserved until an exact client prefix selects an entry. TTL and byte-budget LRU
remove inactive entries; `./ops/model.sh clear-cache <model>` removes only the
selected artifact cache. NVMe is an on-resume load tier, never an active decode
KV tier.

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

The primary provider targets SM86; the optional expert executor targets
SM60/61. P100 attention, dense-Qwen sharding and Flash acceleration failed
their measured gates; details belong in [Benchmarks](benchmarks.md) and
[Research decisions](research-decisions.md). The server provides bearer
authentication but not TLS, public rate limiting or a hardened edge. Stop must
terminate Python/native descendants and release model memory on every device.
