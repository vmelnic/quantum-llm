# DeepSeek performance tasks

This is the implementation backlog for moving the DeepSeek-V4-Flash backend
from a correct functional slice to a genuinely asynchronous, GPU-driven
runtime. Tasks are ordered by dependency, not by apparent ease. A task is
complete only when its acceptance criteria pass in the production worker; an
isolated microbenchmark is supporting evidence, not completion.

## Target and physical constraints

The aspirational target is 30 output tokens/s. Results must distinguish:

- cold and warm execution;
- single-stream and aggregate throughput;
- prefill and autoregressive decode;
- model steps/s and accepted output tokens/s when speculation is enabled.

The current direct packed-FP4 routed kernel takes about 4.39 ms for one layer.
Across 43 layers that is about 189 ms/model step before attention, shared FFN,
and the output head, so the current kernel cannot reach 30 model steps/s even
with a perfect cache. A cold token selects 3,449,290,752 routed bytes; storage
cannot remain on the critical path at the target rate. The implementation must
therefore improve both compute and residency instead of attributing the gap to
one subsystem.

## P0 — truthful attribution and contracts

- [ ] Add cumulative nanosecond counters for directory planning/release,
  storage wait, RAM retention copy, H2D admission, routed CUDA, CPU experts,
  shared FFN, dense attention/state, output head, and whole model steps.
- [ ] Publish cache hits/misses, bytes, high-water marks, uploader counters,
  scheduler counters, and stage times through worker `STATS` and the HTTP
  metrics/model-info surface.
- [ ] Add a reset or delta-snapshot boundary so one request can be attributed
  without restarting the worker.
- [ ] Stop advertising placement prefetch merely because a placement profile
  enables it. Report `observing`, `warming`, `ready`, or `disabled` from actual
  runtime state.
- [ ] Make protocol timing name the token being computed and the token being
  emitted. A future protocol revision should return the first prediction from
  `BEGIN` rather than hiding preparation of the next token in TTFT.

Acceptance: one cold and one immediately repeated production request explain
at least 95% of wall time with non-overlapping critical-path categories. Metrics
must not imply that a policy is active when it is only collecting evidence.

## P1 — learned residency and warm start

- [ ] Consume `RouteCensus::stable_warm_set()` at worker startup.
- [ ] Convert the byte budgets into deterministic per-layer RAM and VRAM warm
  sets while reserving dense, shared, KV, request, staging, and OS headroom.
- [ ] Bulk-load the selected set with bounded concurrency before declaring the
  warm tier ready; allow readiness to distinguish service-ready from warm-ready.
- [ ] Seed a model-family hot list when no census exists, and replace it with
  measured evidence after a completed workload.
- [ ] Implement bounded one-layer-ahead expert prefetch. Predicted records must
  be lower priority than exact current-layer demand and trivially cancellable.
- [ ] Record useful, late, incorrect, cancelled, and evicted-before-use prefetch
  outcomes so lookahead remains an evidence-driven policy.

Acceptance: startup state and cache contents match the authenticated census and
configured budgets; warm requests show the declared hit rate and never alter
router output or generated tokens.

## P1 — persistent asynchronous execution

- [ ] Give every request a persistent non-default CUDA stream, reusable hybrid
  workspace, RoPE storage, and event set.
- [ ] Precompute bounded RoPE tables or generate RoPE on device; remove the
  synchronous per-step host upload.
- [ ] Keep directory hit planning on device. Return only compact miss metadata
  to the host and replace per-layer `cudaStreamSynchronize` calls with event
  dependencies.
- [ ] Release directory pins asynchronously after the last consuming kernel.
- [ ] Split I/O completion, RAM admission, H2D, and CUDA publication into
  separate bounded queues. IOCP completion threads must never wait for CUDA.
- [ ] Use persistent pack handles instead of `CreateFile`/`CloseHandle` for
  every expert read.
- [ ] Replace per-record pageable `vector` allocation and retention copies with
  a bounded slab/arena owned by the RAM cache.
- [ ] Allow multiple H2D transfers in flight, coalesce compatible records, and
  publish them only after their completion event succeeds.

Acceptance: the steady hot path contains no device-wide or stream-wide host
barrier per layer; queue depth, outstanding bytes, cancellation, and memory are
bounded. Failure remains fail-closed and never exposes a partially uploaded
expert.

## P2 — real batching and weight reuse

- [ ] Replace scheduler-only interleaving with a layer-major batch of active
  requests.
- [ ] Build a stable expert union per layer and execute each unique expert once
  for all routed rows that selected it.
- [ ] Decouple request capacity from prefill chunk size and use adaptive causal
  chunks with correct attention visibility.
- [ ] Preserve per-request routing order during weighted aggregation.
- [ ] Apply backpressure from actual row, staging-byte, KV-page, and compute
  credits rather than only request count.

Acceptance: batched and isolated execution produce the accepted numerical
result, and repeated weights are measurably read fewer times. Report both
single-request latency and aggregate throughput.

## P2 — SM86 kernel work

- [ ] Replace scalar INT8-weight/FP32-activation dense GEMV with an optimized
  decode path using quantized or lower-precision activations and SM86-friendly
  vector/DP4A kernels.
- [ ] Add grouped INT8-MMA/Tensor Core kernels for prefill and multi-row MoE,
  while retaining the faster measured decode path for batch one.
- [ ] Tile and tune the direct FP4/UE8M0 routed kernel; unpack reusable weight
  blocks once per tile and reduce redundant scale decoding.
- [ ] Fuse compatible RMSNorm, quantization, bias/gating, RoPE, top-k, and
  aggregation operations to reduce launches and global-memory traffic.
- [ ] Capture stable resident decode segments in CUDA graphs after dynamic
  placement decisions are resolved.
- [ ] Add a hardware-profiled kernel dispatcher; never select Tensor Cores or
  CPU execution solely from a static assumption.

Acceptance: a fully resident 43-layer model step has a measured path toward a
33 ms output-token budget. Each kernel change must pass the existing independent
oracles and an end-to-end token equality/tolerance gate appropriate to its ABI.

## P3 — representation and multi-device placement

- [ ] Quantize dense/shared weights to a smaller compute-ready representation
  when quality gates allow, freeing VRAM for routed experts.
- [ ] Evaluate a smaller authenticated routed representation instead of assuming
  the 147 GB checkpoint layout is the final serving layout.
- [ ] Generalize expert ownership so compute executes on the GPU or worker that
  owns the weights; transfer activations and weighted partial outputs, not
  weights per token.
- [ ] Add measured multi-GPU placement based on capacity, device bandwidth,
  peer bandwidth, kernel rate, cache heat, and current queue debt.
- [ ] Keep remote expert workers behind the same placement interface with
  explicit deadlines, credits, retries, and exact fallback semantics.

Acceptance: placement is reproducible from a published hardware/profile input,
all budgets remain bounded, and adding a device improves either single-stream
latency or aggregate throughput without changing model semantics.

## P3 — useful-token acceleration

- [ ] Implement the checkpoint's native MTP state and verification path only
  after base model steps are fast and cache-resident enough to make rejection
  affordable.
- [ ] Batch the union of experts needed by target verification.
- [ ] Publish acceptance rate, rollback/replay cost, target model steps, and
  useful output tokens separately.
- [ ] Disable speculation automatically when its moving critical-path cost is
  higher than ordinary decode.

Acceptance: useful output tokens/s improves on representative multi-turn
workloads, not only on a synthetic prompt, with identical target-model greedy
semantics.

## Hardware decision gate

After P0–P2, use the measured residency and kernel floors to select the next
host. SATA must leave the target critical path. Candidate improvements include
NVMe, more VRAM, multiple GPUs with owner-compute, and high-channel system
memory. CPU cores remain a concurrent lane for RAM-resident misses; the current
all-core result does not justify moving hot routed compute away from CUDA.

If 30 tok/s refers to aggregate service throughput, qualify it separately with
real concurrent requests. Never present aggregate throughput as single-stream
decode.
