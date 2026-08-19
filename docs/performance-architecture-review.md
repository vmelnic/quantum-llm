# Performance architecture review

This document records the independent Kimi review and the subsequent direct
code verification performed on 2026-08-06. It explains why both backends then
delivered roughly one output token per second for changing chat despite much
faster exact-route hot measurements.

Status 2026-08-10: the implementation constraints listed below were addressed
by the W0–W5 serving campaign (docs/inference-30toks-plan.md §8) — retained
sessions, decoupled prefill, async directory planning, frozen placement,
vectorized GEMV dispatch, event-driven uploads, the widened DeepSeek pipeline,
and immediate dispatch of a lone decode request. This document remains the
design record; measured results are in [Performance evidence](benchmarks.md).
It is not the current implementation plan; the next session starts from
[MoE VM current state and remaining work](moe-vm-next.md).

## Reference system and objective

The reviewed deployment has:

- one RTX 3090 with 24 GB VRAM;
- a Ryzen 5 5600 with 6 cores and 12 hardware threads;
- approximately 64 GB system RAM;
- a local SATA SSD;
- one useful single-stream output target of approximately 30 tok/s.

The target excludes artificial repeated-route benchmarks. It refers to a
declared real-chat workload with changing prompts and retained conversation
history.

## Shared architectural diagnosis

The primary boundary is synchronous expert demand paging at every model layer:

```text
attention/router
      |
      v
device route plan + host synchronization
      |
      v
missing experts? -- yes --> SSD/RAM read --> synchronous H2D upload
      |                                      |
      +--------------- wait -----------------+
      |
      v
expert compute --> next layer
```

Routing is data-dependent, so a layer's exact experts are unknown until that
layer's router finishes. The current runtime frequently waits for demand reads
and uploads before it lets the layer continue. CPU execution receives useful
work only after an expert is already RAM-resident and cannot remove the
first-touch SSD bottleneck.

The visible single busy CPU thread is mostly a symptom of this control flow:
the coordinator spin-polls or waits while the CPU executor workers and I/O
threads have little executable work. Using every CPU core does not remove the
weight-bandwidth lower bound.

## DeepSeek-V4-Flash

### Weight traffic

The compact routed record is 13,369,344 bytes. Each token selects six routed
experts in each of 43 layers:

```text
43 layers × 6 experts × 13,369,344 bytes
    = 3,449,290,752 bytes/token
    ≈ 3.21 GiB/token
```

The complete routed pack is approximately 147 GB. The reference RAM and VRAM
caches hold only a fraction of its 11,008 expert records.

Approximate supply-only bounds are:

| Expert source | Approximate ceiling when all active expert bytes use it |
|---|---:|
| SATA SSD at 0.5 GB/s | 0.14 tok/s |
| RAM to GPU over PCIe | 3–4 tok/s |
| CPU directly from dual-channel DDR4 | below 10–13 tok/s before real compute overhead |
| VRAM | sufficient only for the resident working set |

At 30 tok/s, PCIe can carry only about 400–450 MB of expert data per token.
Therefore roughly 88% of the 258 selected expert-layer pairs must already be
VRAM-resident. The current routed VRAM tier cannot provide that hit rate for
arbitrary routes.

### Verified implementation constraints

- The direct compact upload calls `cudaStreamSynchronize` while holding the
  shared uploader lock
  ([`expert_uploader.cu`](../runtime/src/cuda/expert_uploader.cu)).
- The scheduler allows one speculative prefetch and suspends prefetch while a
  demand acquire is active
  ([`deepseek_scheduler.cpp`](../runtime/src/cuda/deepseek_scheduler.cpp)).
- The worker config explicitly selects one in-flight prefetch and one transition
  prediction per layer
  ([`deepseek_worker.cpp`](../runtime/tools/deepseek_worker.cpp)).
- RAM retention copies each 13.4 MB record before its first upload
  ([`expert_cache.cpp`](../runtime/src/expert_cache.cpp)).
- Two IOCP workers and four staging buffers bound the supply pipeline.
- The hybrid CPU/GPU path synchronizes GPU to CPU and CPU back to GPU; the lanes
  do not form a fully overlapped critical path.
- GPU phase profiling adds blocking event synchronization per layer only when
  explicitly enabled. It is diagnostic overhead, not the normal deployment's
  principal bottleneck.
- The MTP resource reserve is charged when the bundle contains MTP resources,
  even when speculative generation is not enabled.

### DeepSeek verdict

Thirty arbitrary-chat tok/s is physically infeasible on the reference system.
Software work can reduce the gap between the observed 0.3–0.6 tok/s and the
hardware supply ceiling, but it cannot turn SATA, dual-channel DDR4, or PCIe
into the bandwidth required to stream 3.45 GB per token at 30 tok/s.

Thirty tok/s may only become plausible for an unusually stable workload whose
selected experts remain almost entirely in VRAM. That condition has not been
demonstrated for DeepSeek and must not be inferred from Qwen's hot result.

General-purpose DeepSeek at this target requires a material hardware or format
change, such as:

- enough high-bandwidth multi-channel RAM to hold the compact model and feed
  expert compute;
- enough aggregate accelerator memory for the active working set;
- distributed expert compute close to the memory that owns each expert;
- a substantially smaller direct-compute representation and matching kernels.

## Qwen3-Next 80B

### Weight traffic

The aligned INT8 record is 3,158,272 bytes. Each token selects ten routed
experts in each of 48 layers:

```text
48 layers × 10 experts × 3,158,272 bytes
    = 1,515,970,560 bytes/token
    ≈ 1.41 GiB/token
```

The 24,576 routed records occupy approximately 72.3 GiB. The tested 18 GiB VRAM
and 48 GiB RAM expert budgets do not hold the entire routed pack, and every
RAM-resident but non-VRAM expert still crosses PCIe before GPU execution.

The physical distinction from DeepSeek is important:

- Qwen's exact-route native hot path has measured 47.6703 tok/s;
- its repeated-route API path has measured about 30 tok/s;
- therefore its resident GPU compute path can express the target;
- changing chat falls to roughly one tok/s when its working set causes demand
  misses and serialized movement.

Thirty tok/s remains infeasible for uniformly changing routes. It is plausible
for a conversation whose recent expert working set fits in the routed VRAM tier
and achieves roughly 80% or better VRAM hits. The repository does not yet
contain a real-conversation route-diversity measurement proving this condition.

### Verified implementation constraints

1. **The API worker never freezes placement.**

   Native hot gates call `settle_placement()`. The worker protocol does not, so
   placement feedback keeps copying routing indices and scores to the host in
   every layer
   ([`qwen3_next_runner.cpp`](../runtime/tools/qwen3_next_runner.cpp)).

2. **Directory planning synchronizes every layer.**

   `pin_or_collect_misses()` performs device-to-host copies followed by
   `cudaStreamSynchronize`, including on an all-resident route
   ([`expert_directory.cu`](../runtime/src/cuda/expert_directory.cu)).

3. **The batched INT8 GEMV misses the existing vector path.**

   `gemv_batch`, including batch one, selects the byte-at-a-time kernel. A
   `char4` vectorized kernel already exists but is only selected by the separate
   single-row entry point
   ([`transformer_kernels.cu`](../runtime/src/cuda/transformer_kernels.cu)).

4. **Expert upload is synchronous and globally serialized.**

   The uploader performs its H2D copies, synchronizes, and only then invokes the
   completion. RAM-retained records are pageable after their pinned staging
   buffer is released, making later re-uploads slower.

5. **Cache eviction performs repeated linear scans.**

   Admission and eviction scan a large `std::map` under one global mutex. A cold
   token can cause many complete scans on its critical path.

6. **Conversation state is not retained between HTTP requests.**

   Each turn creates a new worker slot and re-prefills the entire formatted chat
   history. KV, Conv, and DeltaNet state are discarded. This raises TTFT,
   replays expert traffic, and biases cache temperature with repeated history.

7. **Prefill is only a four-token microbatch in the deployed Qwen profile.**

   The generic server default is one, but the lifecycle wrapper explicitly uses
   worker capacity four. This is better than scalar prefill but remains far from
   a dedicated large-token prefill path.

8. **The decode batcher waits up to two milliseconds for peers.**

   A lone single-stream request pays the configured batching window at each
   step. This is insignificant near one tok/s but material at a 30 tok/s target.

### Qwen verdict

Qwen is the correct backend for the next attempt at the 30 tok/s target. The
target is credible only for a declared conversation distribution with a bounded
expert working set, not as a promise for arbitrary routes. Its existing hot
measurements prove GPU compute viability and expose several concrete sources of
avoidable API overhead.

## Implementation order

### Qwen: establish a credible 30 tok/s workload

1. Measure unique routed experts per turn, cumulative per-conversation working
   set, consecutive-token overlap, and bytes served by SSD/RAM/VRAM.
2. Dispatch batch-one GEMV to the existing vectorized kernel and verify exact
   output plus hot throughput.
3. Freeze placement after an explicit warmup boundary or make feedback
   double-buffered and asynchronous.
4. Remove the two-millisecond batching wait when only one request is active.
5. Add session-owned KV/Conv/DeltaNet state and append-only conversation prefill.
6. Make expert uploads event-driven, preserve a bounded pinned re-upload tier,
   and decouple upload completion from IOCP workers.
7. Replace linear eviction scans with an indexed bounded policy.
8. Implement a larger weight-reusing prefill path.
9. Pin a measured per-conversation working set only if route-diversity evidence
   proves it fits.

The hot API path should first exceed 45–60 tok/s to leave enough headroom for a
partially warm real conversation. If a conversation's expert union materially
exceeds routed VRAM capacity, the 30 tok/s branch should be rejected before
placement-policy work begins.

### DeepSeek: approach the hardware ceiling honestly

1. Measure per-token SSD/RAM/VRAM bytes and scheduler expert-wait time.
2. Convert compact uploads to event-driven completion.
3. Permit multiple bounded prefetches and demand reads concurrently.
4. Use route-census predictions for the same layer of the next token while the
   current token continues through later layers.
5. Remove the hybrid lane's full-stream synchronization or disable that lane for
   single-stream latency.
6. Move RAM retention off the initial miss-to-upload path.
7. Replace spin polling with event/condition-driven progress.
8. Evaluate MTP only by useful output tok/s after the supply path is pipelined.

The expected result of this sequence is a truthful approach to the current
host's few-tok/s ceiling for mixed routes. It is not a credible path to 30
arbitrary-chat tok/s.

## Work that should not be repeated unchanged

- forcing a fixed CPU/GPU expert split;
- the removed activation-INT8 DP4A path;
- increasing history-only residency beyond safe VRAM capacity;
- disabling prefetch globally;
- treating RAM caching alone as a solution to first-touch traffic;
- using aggregate batching as evidence for single-stream latency;
- optimizing attention before expert supply telemetry shows it dominates;
- presenting an identical repeated prompt as general chat throughput.

## Final decision

The two approximately one-token-per-second results have a common architectural
symptom but not the same feasibility:

> Qwen has a credible path to approximately 30 tok/s for conversations with a
> measured, VRAM-resident working set. DeepSeek exceeds the reference system's
> memory-supply capability before software optimization can guarantee 30 tok/s
> for arbitrary chat.

Future performance claims must identify model, route distribution, cache state,
TTFT, post-first-token rate, end-to-end rate, and SSD/RAM/VRAM bytes per token.
