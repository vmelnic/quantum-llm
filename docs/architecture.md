# Architecture

Status: implemented architecture and explicit boundaries, 2026-08-23.

## Purpose

Quantum LLM is a self-contained native inference runtime for models whose
useful execution state does not map cleanly to a single GPU allocation. It is
built around two distinct cases:

1. a dense/hybrid model such as Qwen3.8-27B, where every target-call matrix is
   hot and the difficult state is long-context KV;
2. a sparse MoE model such as DeepSeek-V4-Flash, where dense organs are hot but
   only the routed experts selected by the exact router are needed for a token.

The project does not claim that SSD paging makes arbitrary dense models fast.
It pages only state that is absent from the current exact execution path, and
reports the resulting VRAM, RAM, PCIe and storage traffic.

## End-to-end system

```text
official checkpoint
       |
       | strict source adapter: names/config -> tensor roles/capabilities
       v
transactionally published artifact under ${MODEL_ROOT}
  manifest + hashes + tokenizer/generation assets
  qpack/compact payloads
  runtime-model.tsv: geometry + ordered operations + numeric ABIs
       |
       v
common HTTP service -> admission -> chat template -> retained session
       |                                      |
       v                                      v
common native VM runner -> capability binding -> request state
       |                                      |
       +------------------+-------------------+
                          |
              +-----------+-----------+
              |                       |
       dense FP4 provider       sparse DeepSeek provider
       resident weights         exact router + expert pages
              |                       |
       exact/paged KV           VRAM <- RAM <- NVMe
              +-----------+-----------+
                          |
                  sampled/greedy tokens
```

The common service and VM do not select Qwen, DeepSeek, a fixed layer count or
a tensor path. The artifact declares operations and ABIs; providers advertise
which declarations they can execute. A source adapter may understand an
upstream family because checkpoint tensor names are not universal. New
mathematics or a new encoding requires a provider, not another HTTP server or
launcher.

## The model as organs

| Organ | Access pattern | Correct placement rule |
|---|---|---|
| embedding | selected rows | keep/stage selected rows; do not scan it per token |
| dense projections and MLP | every target call | fastest compute memory; streaming only solves capacity |
| recurrent/linear-attention state | every next token, fixed size | device resident while the request owns a hot slot |
| full-attention K/V | every retained position is active at decode | compute must stay near the complete shard; per-token NVMe/PCIe streaming is a throughput failure |
| MoE router | every routed layer | resident and exact |
| routed experts | only selected experts | immutable logical pages, eligible for demand paging |
| shared experts | every routed layer | resident model state |
| vocabulary head | every generated target token | resident or explicitly accounted hot traffic |
| MTP/draft state | only when the generation policy can use exact verification | allocate conditionally; never pay it for incompatible sampling |

## Qwen3.8 dense execution

The current artifact is `${MODEL_ROOT}/qwen3.8-27b-fp4` and declares:

- 64 text layers: 16 full-attention and 48 recurrent/linear-attention layers;
- 262,144 maximum positions;
- one 14,775,390,208-byte `dense.qpack` with 1,199 records;
- FP4 E2M1 matrix payloads with UE8M0 block-32 scales, quant ABI 3;
- separately declared raw/FP32 records, norms and metadata;
- text, MTP and vision operations in a schema-3 `runtime-model.tsv` program;
- no router and no routed experts.

The FP4 weights are loaded into the RTX 3090 execution tier. Prompt execution
uses bounded device-resident activation tiles, so hidden states no longer make
a full GPU-to-host-to-GPU round trip after every operation.

Exact target F16 KV is request-owned and allocated progressively in pinned RAM
in 256-token pages. Declaring a 262K context does not allocate 262K KV at
`BEGIN`. At attention time, the current provider stages bounded K/V spans and
executes one layer at a time on the GPU. This preserves exact F16 values but is
not a fast saturated-context design: at 262K, all 16 GiB of target K/V is hot
for every scalar decode call.

The 48 recurrent layers keep their causal convolution/matrix state in the hot
provider slot. Checkpoint, rewind and retention state preserve exact append-only
session semantics.

## Qwen sessions and multi-agent behavior

The current RTX 3090 provider has one hot execution slot. Multiple Pi sessions
may declare a 262K maximum because the maximum is not reserved upfront. An
inactive request is parked in bounded host RAM:

```text
active session
  exact F16 KV pages already owned in host RAM
  recurrent + hidden + optional compact device state in hot slot
        |
        | END RETAIN
        v
parked session
  populated F16 pages + one compact continuation blob in RAM
  zero hot provider slots
        |
        | BEGIN RESUME + suffix only
        v
active session
```

Resume is transactional. A failed suffix prefill rewinds to the committed
prompt checkpoint and reparks it. A client cancellation retains only the
client-echoable prompt, never partial assistant output. An unchanged prefix is
a valid zero-delta resume.

This is capacity multiplexing, not simultaneous GPU execution. Several agents
can retain short/medium histories, but if several sessions actually populate
hundreds of thousands of F16 tokens, pinned RAM and park/restore bandwidth
become the admission limits. `MODEL_WORKER_CAPACITY` does not remove the
serialized Python command channel and provider mutex.

## DeepSeek sparse execution

DeepSeek's bundle keeps dense, shared, attention and MTP resources as declared
model state. Its 11,008 routed experts are immutable logical pages identified
by artifact namespace, component, layer, expert ID and encoding ABI. The
current compact record is 13,369,344 bytes of authenticated FP4/UE8M0 payload.

```text
ABSENT -> SSD_LOADING -> RAM_READY -> GPU_UPLOADING -> VRAM_READY
   +------------------------- failure ----------------------> FAILED
```

One acquisition exists per key. Demand outranks bounded prefetch and warm work.
Protected/probationary RAM and transient/protected VRAM prevent cold first
touches from immediately displacing known-hot pages. Leases and CUDA events
prevent eviction while a route uses a record.

The scheduler publishes the complete exact top-6 union for each routed layer.
Execution waits until every selected expert is available; no missing expert is
treated as zero and top-k is never reduced. Stable aggregation preserves the
router's selection order and weights.

This is where NVMe/RAM/VRAM paging is valid: unselected experts are cold for
the current token. It makes a model larger than RAM+VRAM executable, but novel
routes still pay first-touch storage traffic and cannot be advertised at the
settled-cache rate.

## Artifact and provider boundary

`runtime-model.tsv` is the executable program contract. It binds:

- model geometry and ordered operations;
- tensor roles and input/output ABIs;
- numeric/storage encodings;
- provider capabilities and parameters;
- optional response, vision and exact-decode capabilities.

Startup validates manifests, completion markers, sizes, hashes, record layouts
and operation compatibility before readiness. Large pack SHA-256 is mandatory,
but uses the Windows native cryptographic provider on the supported host.

Expert Pack and the DeepSeek compact bundle are still different physical
containers. They converge at the model descriptor, operation-provider and
logical-page contracts. The repository must not claim one universal physical
format until the DeepSeek payload can be represented without losing its
authenticated source ABI or paging geometry.

## API and failure boundaries

The Python front end owns HTTP, authentication, body/media limits,
tokenization, official chat templates, streaming and admission. The native
worker owns model state, the compiled program, cache placement and token
selection. Their protocol advertises actual capabilities; unsupported session,
vision, tool or KV behavior fails explicitly.

- `/health` is process liveness; `/ready` also requires an admitting worker.
- `/model-info` reports the artifact, limits, provider and KV geometry.
- `/metrics` uses a cached bounded worker snapshot and must remain responsive
  during a long prefill.
- queue, body, output, context, image, RAM, VRAM, KV and worker-slot budgets are
  independent.
- non-loopback serving requires bearer authentication; TLS and rate limiting
  belong at the deployment edge.
- stop must terminate both Python and native descendants and release VRAM.

## Hardware boundary

The implemented CUDA build targets SM86 and one RTX 3090. There is currently no
multi-device allocator, P2P transport or SM60/SM61 provider. RTX 3090 + Tesla
P100/P40 placement is a quantitatively defined roadmap item, not implemented
functionality. See [Research decisions](research-decisions.md) and
[Roadmap](roadmap.md).
