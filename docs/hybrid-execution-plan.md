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

Measure per layer and execution epoch:

- CPU queue and compute time;
- resident GPU queue and compute time;
- H2D queue, bytes, and completion time;
- selected rows per expert;
- overlap and the final layer critical path.

For a RAM-resident GPU miss, compare measured estimates:

```text
CPU queue + CPU(expert, rows)
H2D queue + upload(expert bytes) + GPU queue + GPU(expert, rows)
```

Cached GPU experts retain GPU priority. Small-load misses prefer CPU; a
high-load miss may use the bounded transient GPU path when transfer amortizes.
The scheduler balances lanes rather than applying a fixed CPU percentage.

Acceptance:

- deterministic scheduler tests cover CPU, GPU, transfer, saturation, and tie
  decisions;
- every decision publishes a reason/cost snapshot in bounded telemetry;
- exact full-model tokens and all memory/failure invariants remain unchanged;
- real API cold/warm and runner single/aggregate evidence is recorded without
  comparing different placement states as equivalent.

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
