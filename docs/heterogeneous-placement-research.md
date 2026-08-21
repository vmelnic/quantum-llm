# Heterogeneous organ placement

Status: architecture research note, 2026-08-19. Nothing in this document is
implemented or performance-qualified unless explicitly identified as existing
evidence. The numerical Qwen3.8 derivation is in
[Heterogeneous placement benchmark](heterogeneous-placement-benchmark.md).

## Objective and invariants

Treat a model as a set of immutable compute organs plus mutable request state,
and place each organ together with the compute that consumes its data. The goal
is to use VRAM/GPU, RAM/CPU and NVMe without copying every non-VRAM tensor into
VRAM before it can execute.

The invariants are:

- model math, selected experts, routing weights, context and sampling are not
  reduced;
- a RAM-resident shard is executed by the CPU from RAM; it is not counted as a
  RAM placement if its weights are uploaded to the GPU for every invocation;
- active NVMe traffic is admitted only when its byte rate fits the target
  throughput; NVMe capacity does not imply decode bandwidth;
- placement is declared by operator capabilities, tensor roles, split axes,
  reductions and state ownership, never by a Qwen, DeepSeek or Kimi branch;
- algebraically exact partitions still require a numerical gate because
  floating-point reduction order can change rounding.

## The execution object

A decoder is a depth-ordered dataflow. Immutable weights do not flow between
layers. A comparatively small activation frontier and layer-owned state do:

```text
activation frontier
        |
        v
depth mix / norm -> attention(layer state) -> FFN or routed experts
        |                                           |
        +---------------- next frontier <-----------+
```

The frontier need not be one tensor. A conventional residual block has one
hidden stream; mHC can have several streams; AttnRes can retain bounded block
summaries. The artifact must describe that frontier instead of the runtime
assuming one family-specific hidden tensor.

The useful organ classes are:

1. always-used dense compute: norms, attention projections, dense/shared MLPs,
   routers and the vocabulary head;
2. request state: full-attention KV, compressed/sparse KV, recurrent state and
   rollback/checkpoint state;
3. conditional compute: routed experts;
4. lookup and modality organs: embeddings and vision/audio encoders;
5. draft organs: MTP or another artifact-declared speculative component.

## Critical-path model

For a dependent sequence of stages, latencies add. Splitting whole consecutive
layers between CPU and GPU therefore does not make a single token concurrent.
The useful split is inside an algebraically parallel stage.

For shard `i` of one stage, a lower-bound model is:

```text
T_i >= max(weight_bytes_i / local_weight_bandwidth_i,
           state_bytes_i  / local_state_bandwidth_i,
           operations_i   / compute_rate_i)
       + activation_transfer_i / link_bandwidth_i

T_stage >= max_i(T_i)        # shards execute concurrently
T_token >= sum(T_stage)      # dependent stages remain ordered
```

A placement is feasible at output rate `R` only if both its capacity constraints
and `T_token <= 1/R` hold. This model is deliberately a lower bound: launches,
synchronization, imperfect overlap and reductions can only increase time.

## Exact decomposition boundaries

### Embedding lookup

Only one vocabulary row is read per decoded token. The complete table can stay
in RAM; the CPU decodes the selected row and transfers one hidden vector. This
is different from the vocabulary head, which scans every output row.

### Dense gated MLP

For intermediate index set `I = I_gpu union I_cpu`:

```text
g_gpu = SiLU(Wgate[I_gpu] x) * Wup[I_gpu] x
g_cpu = SiLU(Wgate[I_cpu] x) * Wup[I_cpu] x

y = Wdown[:, I_gpu] g_gpu + Wdown[:, I_cpu] g_cpu
```

Each side retains its own gate, up and matching down-projection weights. The CPU
receives `x` and returns one partial hidden vector; expert or dense weights do
not cross PCIe.

### Attention state

Partition the key/value sequence into GPU and CPU ranges. Each side computes an
online-softmax summary:

```text
m_j = max(scores_j)
l_j = sum(exp(scores_j - m_j))
o_j = sum(exp(scores_j - m_j) * values_j)
```

The summaries merge algebraically:

```text
M = max(m_gpu, m_cpu)
L = exp(m_gpu - M) l_gpu + exp(m_cpu - M) l_cpu
O = exp(m_gpu - M) o_gpu + exp(m_cpu - M) o_cpu
attention = O / L
```

Thus RAM-resident KV can be scanned by the CPU while the GPU scans VRAM KV. No
KV copy to the GPU is required. Partitioning complete layers instead would
serialize CPU and GPU on the depth dependency and is not the intended design.

Attention can also split by independent head groups. Each side owns the
corresponding output-projection columns and returns a partial hidden vector.

### Recurrent attention

DeltaNet/KDA-style state is naturally partitioned by recurrent heads. The CPU
owns the weights and state for its head group, the GPU owns the remainder, and
their output-projection partials are reduced exactly as above.

### Mixture of experts

Experts are independent after exact routing and before weighted aggregation.
VRAM experts execute on the GPU and RAM experts execute directly on the CPU.
Only expert inputs, outputs, IDs and routing weights cross the link.

An expert that must be read from NVMe still has to enter a compute-capable tier.
For a target rate `R`, its hard admission condition is:

```text
active_nvme_bytes_per_token <= sustained_nvme_bytes_per_second / R
```

Prefetch can hide latency only while other useful work is long enough; it does
not change this byte inequality.

### Vocabulary head

The head can be row-sharded and partial logits can be reduced for exact
normalization/sampling. It is usually a poor CPU placement when the GPU can
scan the whole head much faster; the artifact capability allows it, while the
placement solver decides from measured rates.

## Tier roles

### VRAM/GPU

- high-byte organs invoked for every token;
- the GPU share of active KV/recurrent state;
- MTP when it reduces target calls sufficiently to repay its own traffic;
- hot routed experts;
- small norms, routers, reduction state and execution workspaces.

### RAM/CPU

- embedding lookup tables;
- an attention-state partition executed directly by CPU kernels;
- balanced intermediate/head shards of dense and recurrent operations;
- warm routed experts executed directly from their packed representation;
- inactive sessions and prefix state awaiting restoration.

### NVMe

- the immutable published artifact;
- modality organs not used by the current request;
- inactive session/prefix snapshots;
- cold routed experts only within the active-byte inequality;
- sparse-selected state blocks only when the selected byte rate is feasible.

Full-attention KV is read for every target call. Moving it to NVMe does not
reduce work and is therefore not a valid active Qwen placement at the current
throughput target.

## Model-independent runtime contract

The artifact/provider boundary needs to declare for each operation:

- legal split axes: rows, intermediate channels, heads, KV ranges or experts;
- the tensors and mutable state co-owned by each shard;
- activation input/output ABI for every execution tier;
- the exact reduction operation and accumulation precision;
- estimated bytes and operations per invocation;
- whether access is unconditional, routed, modality-gated or session-inactive.

The planner then assigns shards under memory and critical-path constraints. A
new model using supported mathematics supplies shapes and roles; it does not
receive a new launcher or family-specific placement policy.

## Qwen3.8 conclusion

The current evidence supports this initial candidate, not an implementation
claim:

- keep scan-heavy target weights, MTP, recurrent state and most active KV in
  VRAM;
- keep the embedding table in RAM;
- keep the unused vision organ outside VRAM;
- evaluate a CPU-computed KV range before weight sharding because the CPU is a
  larger fraction of measured attention bandwidth than of measured GPU weight
  bandwidth;
- keep active full-attention KV off NVMe;
- preserve inactive session state on RAM/NVMe independently of the active
  request.

The executable-page experiment now establishes a sharper boundary. A dense
FP4 SwiGLU can be partitioned exactly by intermediate channels and executed
concurrently across resident GPU, resident CPU and a direct-I/O GPU lane. On
the real layer-0 Qwen3.8 FFN the output error relative to the current resident
CUDA operation was `1.1920929e-7` with cosine `1.0`, while resident weight VRAM
fell by 33.27%.

That result does not make NVMe an active dense tier. One 47.5 MB direct page
took 18.1491 ms to read, compared with 0.236384 ms for the entire resident GPU
SwiGLU. Even excluding the prototype's scalar integrity checks, the derived
paged critical path was at least 23.239 ms. Therefore:

- executable pages are a valid universal storage/execution primitive;
- CPU execution can own an independently reducible shard when its measured
  completion time balances the GPU shard;
- NVMe pages are admissible for conditional/cold data, not unconditional dense
  Qwen decode weights;
- a smaller page size improves scheduling granularity but does not change the
  dense active-byte lower bound;
- page authentication must not perform a scalar full-page SHA twice on the
  request critical path.

The complete evidence is recorded in
[Executable SwiGLU page experiment](heterogeneous-placement-benchmark.md#executable-swiglu-page-experiment).

For DeepSeek/Kimi-class MoE, the same contract assigns complete experts to
GPU/VRAM or CPU/RAM and applies the NVMe active-byte gate to the remaining cold
route. No performance claim follows from total parameter sparsity alone.

## Required qualification before implementation acceptance

1. Measure a packed CPU kernel for the actual FP4 KV ABI; theoretical DDR
   bandwidth is not a result.
2. Validate split online-softmax against the existing GPU/reference oracle over
   short and maximum context.
3. Measure one complete full-attention layer with concurrent GPU and CPU KV
   ranges, including synchronization and reduction.
4. Recompute the placement from measured CPU bandwidth and memory capacity.
5. Integrate the complete generic split only if the layer result improves the
   real critical path.
6. Re-run the populated-context service gate and bounded reference-quality
   comparison. A microbenchmark alone is not completion.

## External architecture references

- [Qwen3.8-27B artifact source](https://huggingface.co/Qwen/Qwen3.8-27B)
- [DeepSeek-V4 official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/README.md)
- [DeepSeek-V4 Flash configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/blob/main/config.json)
- [Kimi K3 official report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
