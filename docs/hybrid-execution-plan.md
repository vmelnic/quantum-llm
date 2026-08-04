# Hybrid CPU/GPU execution plan

This plan converts the current working heterogeneous path into a measured,
dynamic scheduler. Each stage is implemented, documented, validated, and
committed independently. A later stage may depend only on a committed and
passing earlier stage.

## Baseline and constraints

The qualified Qwen3-Next 80B profile uses an RTX 3090, a Ryzen 5 5600
(6C/12T), 64 GiB dual-channel DDR4-3200, 48 GiB RAM expert cache, and an 18 GiB
VRAM expert cache. Current evidence is:

- 47.6703 tok/s hot single-stream runner;
- 41.6653 tok/s hot four-request aggregate runner;
- approximately 19.12 tok/s after first token for a repeated 25-token API
  prompt;
- 51,263 VRAM expert acquisitions, 8,157 CPU expert groups, 9,281 CPU expert
  selections, and no SSD miss in the qualified mixed batch;
- 521,830,400 CPU-result H2D bytes in that batch, versus approximately 76 MiB
  of selection vectors containing useful CPU results.

The aggregate result is not a 30 tok/s single-request SLO. The implementation
must continue to report runner, API, single, aggregate, cold, and warm results
separately.

The following invariants do not change:

- exact router top-k and routing weights;
- no omitted expert and no partial top-k;
- stable aggregation independent of CPU/GPU completion order;
- bounded RAM, VRAM, staging, KV, queues, and worker slots;
- model/container hashes and quantization ABI remain fail-closed;
- source checkpoint and the validated 80B Expert Pack are retained until an
  operator explicitly approves deletion.

## Stage 1 — compact CPU result transport — completed

Replace the dense `rows × top_k × hidden` CPU output transfer with:

```text
compact_outputs[cpu_selection_count, hidden]
cpu_slot_by_selection[rows × top_k]
```

The CPU executor keeps global selection IDs for input-row lookup and receives
compact output slots. CUDA aggregation resolves each non-GPU selection through
`cpu_slot_by_selection`. GPU selections keep the existing stable order.

Acceptance:

- CPU and split CUDA contract tests compare compact aggregation with the dense
  reference;
- full-model isolated/interleaved/chunked token equality remains true;
- `cpu_result_h2d_bytes` equals compact payload bytes, excluding the small
  mapping;
- pagefile growth remains zero and throughput gates do not regress below their
  existing required thresholds.

Measured result on the qualified mixed batch:

- 9,521 CPU selections;
- 77,996,032 result bytes, exactly `selections × 2048 × sizeof(float)`;
- 85.05% fewer CPU-result H2D bytes than the previous 521,830,400-byte dense
  masked transfer;
- 42.5976 tok/s aggregate versus the previous 41.6653 tok/s;
- exact isolated/interleaved/chunked output and zero pagefile growth.

## Stage 2 — critical-path telemetry and dynamic scheduling

### Stage 2a — measured CPU/GPU expert lanes — completed

The runner records CUDA events around the resident GPU expert lane for every
layer and combines them with the existing host-timed CPU lane and full expert
phase. Telemetry now publishes CPU/GPU selections, nanoseconds per selection,
compact result/map H2D bytes, and observed overlap. `cpu_gpu_overlap_seconds`
is a conservative lower bound:

```text
max(0, CPU expert time + GPU expert time - whole expert-phase wall time)
```

The whole expert phase also contains compact result transfer and stable
aggregation, so the overlap value intentionally does not attribute that
overhead to either compute lane.

On the qualified mixed batch, instrumentation measured:

- 9,523 CPU selections at 112,652 ns/selection;
- 58,157 GPU selections at 6,953.87 ns/selection;
- 1.07279 s CPU lane, 0.404417 s GPU lane, and at least 0.136184 s overlap;
- 78,012,416 result bytes plus 320,950 mapping bytes transferred CPU→GPU;
- 42.5398 tok/s aggregate, exact interleaved/chunked outputs, zero SSD misses,
  and zero pagefile use.

This evidence shows that a CPU selection is roughly 16.2 times as expensive as
a resident GPU selection on this workload. It does not imply that every
RAM-resident expert should be uploaded: upload, eviction, queue debt, expert
record size, and reuse horizon still decide whether a miss amortizes.

### Stage 2b — dynamic lane scheduler — completed

`HybridDispatchPlanner` is a platform-neutral, bounded planner. For each
routed expert it receives selection count, record bytes, CPU availability,
GPU availability, and existing GPU residency. It applies these rules:

1. an already-resident expert stays on GPU;
2. an expert with only one feasible executor uses it or the layer fails;
3. flexible RAM misses are ordered by descending CPU work;
4. each is assigned to the choice minimizing the projected layer critical
   path, with CPU as the deterministic no-upload tie break;
5. a GPU upload is eligible only when the cache can admit it without evicting
   an equally hot/in-use resident;
6. current-route GPU entries receive cache references before admission or
   upload can select a victim.

The cost model reflects the current implementation: H2D uploads are serialized
before routed GPU compute, while CPU expert compute can overlap the GPU lane.
CPU, resident-GPU, and RAM→GPU observations update bounded EWMAs. Initial CPU
and GPU values come from Stage 2a; H2D begins at a conservative 8 GiB/s until a
clean RAM-resident upload observation exists.

Every plan returns per-expert reason plus both alternative critical-path costs.
A 256-decision circular trace retains the newest snapshots; aggregate reason
counters and current EWMAs appear in runner JSON. The planner rejects duplicate,
empty, unavailable, or over-bound candidate sets rather than dropping work.

Measured frozen-placement gate:

- 42.2259 tok/s aggregate;
- exact interleaved and chunked-prefill token equality;
- 1,693 split-layer plans and 56,198 candidates;
- 47,861 resident-GPU and 8,337 CPU-only decisions;
- zero rejected plans, zero SSD misses, zero expert H2D, and zero pagefile use.

Cost wins are intentionally zero in this measured window: after warmup the
gate freezes placement, so RAM misses are CPU-only by policy. Unfrozen service
epochs may make cost-based upload decisions; benchmark output separates their
pre-measurement counters from the frozen window. The warmup recorded 3,896
CPU critical-path wins and zero GPU-upload wins: with this short microbatch and
synchronous uploader, no eligible RAM miss amortized its transfer.

The implemented cost basis covers:

- CPU queue and compute time;
- resident GPU queue and compute time;
- H2D queue, bytes, and completion time;
- selected rows per expert;
- overlap and the final layer critical path.

For a RAM-resident GPU miss, the planner compares measured estimates:

```text
CPU queue + CPU(expert, rows)
H2D queue + upload(expert bytes) + GPU queue + GPU(expert, rows)
```

Cached GPU experts retain GPU priority. Small-load misses prefer CPU; a
high-load miss may use the bounded transient GPU path when transfer amortizes.
The scheduler balances lanes rather than applying a fixed CPU percentage.

Acceptance evidence:

- deterministic scheduler tests cover resident/forced CPU/forced GPU,
  transfer-vs-compute choices, EWMA updates, bounds, duplicate rejection,
  circular trace order, and stable ties;
- every decision publishes a reason/cost snapshot in bounded telemetry;
- exact full-model tokens and all memory/failure invariants remain unchanged;
- runner measurement separates pre-measurement and frozen-window decisions.

Cold/warm API qualification remains part of the later service/profile stage;
it must not be represented as equivalent to this frozen runner result.

## Stage 3 — score-aware cache and bounded prefetch

Add routing-score evidence to the existing frequency/reuse cache. Retain a
bounded score history and combine reuse, selection load, measured CPU debt,
upload cost, and victim temperature. Prefetch is admitted only if it has
credits, cannot evict an in-use/hotter expert, and cannot starve current work.

Potential predictors are sequence-local reuse, near-top-k scores, and
workload/prefix warm sets. Prediction failure and cache pollution are first-
class metrics.

Acceptance:

- no unbounded score or trace history;
- prefetch cancellation and stale-epoch behavior are deterministic;
- useful/wasted prefetch bytes and hit-rate delta are observable;
- cold/warm service evidence improves or the feature remains disabled by
  default.

## Stage 4 — CPU executor efficiency and autotuning

Keep FP32 activations and the current numerical ABI first. Improve the AVX2
executor through cache blocking, packed traversal, multi-row reuse, software
prefetch, and allocation reuse. At startup, a bounded calibration selects the
physical-core/SMT thread count and kernel tile from real hardware measurements.

Activation INT8 or another arithmetic ABI is a separate future stage because
it requires numerical and model-quality qualification.

Acceptance:

- the CPU reference tolerance does not change;
- calibration has a strict time/memory bound and a deterministic fallback;
- measured effective weight bandwidth and selections/s are published;
- the scheduler consumes calibrated costs instead of a hard-coded CPU rate.

## Stage 5 — operator placement profiles

Expose policy goals, not fixed expert percentages:

- `latency`: favor single-stream GPU residency;
- `balanced`: minimize measured CPU/GPU/PCIe critical path;
- `capacity`: conserve VRAM and admit larger expert working sets.

GPU power limiting remains an external operator policy. It is qualified only
after telemetry shows a CPU/cache-bound workload; the runtime must not change
system power limits implicitly.

Acceptance:

- profile and effective budgets appear in `/model-info` and artifacts;
- overload and insufficient-memory failures remain explicit;
- each profile has a documented intended workload and evidence boundary.

## Stage 6 — larger-model decision

Disk cleanup and model downloads are separate operator decisions. Research
must report exact reclaimable paths, retained assets, free-space requirements,
checkpoint size, active parameter mass, architecture gaps, license, and a
repacking estimate. Nothing is deleted and no model is downloaded without
explicit approval.

The next checkpoint should exercise a real architectural requirement: a larger
expert directory, greater active mass, a new routing structure, or storage
sharding. Size alone is not sufficient justification.

## Longer-term boundary

For hundreds-of-billions to trillion-parameter MoEs, a dual-channel desktop CPU
cannot be the only expert tier. The same scheduler must be able to choose a
local CPU executor, local GPU, local storage, or a remote expert worker. The
durable rule remains: move activations to the compute that owns weights; do not
move whole experts between machines per token.
