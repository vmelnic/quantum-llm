# Measured placement profile

Placement is derived from measured lane costs and explicit memory envelopes.
Model size alone never selects RAM, VRAM, CPU, or GPU ownership.

## Cost calibration

The DeepSeek qualification measures three independent quantities:

- all-core CPU time for six native packed-FP4 experts;
- CUDA event time around only the six routed packed-FP4 selections, excluding
  the invariant shared expert and HCA post-processing;
- eight pinned-host-to-device copies of one authenticated compact expert.

The resulting `PlacementCostProfile` carries both values and nonzero sample
counts/bytes. The profile solver rejects absent, zero, negative, NaN, or
infinite measurements. `HybridDispatchPlanner` receives these values as its
initial CPU, GPU, and H2D EWMAs; later observations may adapt them within the
same bounded planner.

On the current qualification host, one real run measured:

| quantity | result |
| --- | ---: |
| all-core CPU per packed expert | 9.342 ms |
| routed CUDA per packed expert | 0.543 ms |
| pinned H2D | 3.393 GB/s |
| H2D sample | 106,954,752 bytes |

For a synthetic six-cold-expert RAM route using those real measurements, the
critical-path planner selected two CPU and four upload/GPU experts. Its
projected layer critical path was 18.685 ms. This is not a model throughput
claim; it demonstrates that measured serialized upload can make a mixed cold
route preferable even though every hot expert is much faster on CUDA.

## Memory solver

Each memory domain is supplied explicitly with:

- runtime-usable bytes after external OS/driver/co-tenant policy;
- known fixed bytes for dense state, KV/request state, workspaces, and staging;
- emergency reserve;
- exact compact record bytes;
- required minimum and optional operator maximum expert slots.

The solver computes only:

```text
available cache bytes = usable - fixed - emergency reserve
expert slots          = floor(available cache bytes / record bytes)
```

An operator cap may lower the resulting slot count. A required minimum cannot:
the profile fails with backpressure instead of silently starting an unsafe or
nonfunctional configuration. Returned cache bytes are exact whole-record
multiples. The same result seeds `MemoryResourceGovernor`, which remains the
runtime admission authority once execution starts.

The qualification reports a conservative instantaneous plan after subtracting
current non-cache host/device commitments and retaining 4 GiB host plus 1 GiB
device emergency headroom. These policy reserves are inputs, not discovered
hardware facts. A production deployment must expose them as operator config
and record them with the hardware/profile artifact.

## Evidence boundary

Calibration runs at model qualification or controlled startup, not per token.
Normal FFN execution does not create CUDA timing events when no timing output
is requested. The production worker must publish the measured samples, memory
inputs, solved slot counts, and current EWMA values so an operator can explain
every placement decision.
