# Benchmarks and evidence

## DeepSeek-V4-Flash vertical slice

The pinned 284B-class checkpoint has a qualified real layer-2 decode block on
RTX 3090. The fourth token follows three prior attention updates and emits/consumes
the first ratio-four compressed slot. It then routes and executes six real
routed experts plus the always-active shared expert.

| Component | Result |
|---|---:|
| attention | 6.80 ms/token |
| FFN HCA + norm + hash route | 0.78 ms/token |
| routed top-6 + shared + FFN HCA post | 7.19 ms/token |
| composed hot compute | 14.77 ms/layer-token |
| cold gather/admission/publication for seven experts | 410 ms |
| full-block RMSE / maximum error | `1.04e-4` / `4.76e-4` |

The production cold-path gate keeps the shared expert resident and lets the
outer scheduler acquire the six routed experts from the complete catalog. With
the global acquire credit set to two, the observed peak was exactly two; all
six completed and the resumed block retained the same `4.76e-4` maximum error.
This is a bounded orchestration result, not a claim that parallel cache waiters
make one physical staging slot perform parallel I/O.

The main-model edge gate reuses resident BF16 embedding/head weights. One real
embedding row matched bit-for-bit; the complete 129,280-logit projection
matched its independent row-streamed oracle with `6.89e-7` RMSE and `5.72e-6`
maximum error, including the same greedy token (65,270). This is correctness
evidence; head latency is not yet reported as a throughput optimization.

The cold 410 ms is outside the hot compute interval. Conversely, 14.77 ms is
one layer, not one generated token; extrapolating it across 43 layers would be
well below the throughput target. This result establishes the complete
attention/router/cache/MoE/HCA execution contract and locates the next compute
boundary. It is not a full-model tokens/s claim.

### First complete chat prompt

The official DeepSeek encoder turns `Hi` into five input tokens. A single
request retained its KV/compressed-attention state while executing all 43
layers at every position, then projected 129,280 logits. The result was:

| Measurement | Result |
|---|---:|
| prompt tokens | 5 |
| layer advances | 420 |
| routed acquisitions | 859 |
| prompt decode | 3.74 s |
| average prompt decode | 0.75 s/token |
| greedy token | 19,923 (`Hello`) |

This proves end-to-end checkpoint execution and tokenizer compatibility; it
does not yet meet the throughput SLO. Layer-major traversal is 5.32x faster
than the original 19.91-second token-major baseline. The 64-entry global routed
cache used about 10.7 GB total VRAM during execution and released back to the
normal driver baseline after exit. A larger seven-routed-slots-per-layer
policy was rejected:
route turnover forced costly allocation and FP8-to-INT8 admission churn and did
not complete the five-token prompt within five minutes.

The compact layer-shard pack preserved the same token and exposed the cold
boundary explicitly. Its first run took 29.05 seconds while reading 12.57 GB
of routed source and publishing 22.73 GB of expanded expert data over 859
acquisitions. Repeating the request from the Windows file cache took 3.61
seconds, about 3.5% below the prior 3.74-second warm result. The pack therefore
improves layout and recovery, but does not turn SATA into a viable per-token
weight tier. Cold and warm numbers must not be compared as kernel speed.

A bounded persistent CUDA slot pool then replaced allocation on every expert
turnover. The same request performed 902 logical expert publications using 107
physical device allocations, 795 exact-size slot reuses, and one persistent
compact staging allocation. It completed in 3.63 seconds versus the 3.61-second
warm run, so the change is throughput-neutral at this cache size. Its value is
bounded allocator behavior and enabling a larger active set without repeating
the allocation storm; it is not reported as a tokens/s gain.

### First autoregressive decode token

The controller was extended past TTFT while retaining all 43 layers of
attention, HCA, and KV state. Feeding the sampled `Hello` token back into the
model produced token ID 3 next. That complete decode step took 3.83 seconds
(`0.261 tok/s`) and added exactly 258 routed acquisitions: six misses in every
layer. This is the real single-stream inter-token baseline, not prompt
throughput. It proves that the 64-slot global cache loses the entire previous
token's routed working set before traversal returns to the same layer.

The routed down projections were then changed from six serial expert loops per
output block to six parallel selection outputs followed by deterministic
routing-order aggregation. A three-token gate retained the exact token IDs
`[19923, 3, 1730]` and the full-block maximum error remained `4.77e-4`.
Isolated FFN observations moved from roughly 6.92–7.22 ms to 6.65–6.98 ms.
The observed two-token decode average improved from 2.53 to 1.51 seconds
(`0.395` to `0.662 tok/s`), but this is not attributed wholly to the kernel:
the run still read 19.36 GB and file-cache state affects end-to-end latency.
Only eight routed misses disappeared. The qualified claim is a modest grouped
FFN compute improvement with unchanged output, not a solved storage boundary.

The separate uncompressed layer-0 gate exercises the checkpoint's
`compress_ratio=0` sliding-window mode. Attention measured 5.42 ms/token and
the full block matched its independent oracle with RMSE `3.55e-5` and maximum
error `2.20e-4`. Its attention request state is 685,568 bytes because it owns no
compressed/index cache. This closes a functional prerequisite for owning all
43 layers; it is not presented as an optimization comparison with layer 2.

## Tested configuration

- model: `Qwen/Qwen3-Next-80B-A3B-Instruct`;
- source revision: `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`;
- source tensors: 162,659,161,528 bytes;
- Expert Pack v1 INT8: 81,903,198,208 bytes;
- dense pack: 4,191,133,696 bytes;
- hardware: RTX 3090 24 GB, approximately 64 GB RAM, Windows, local SSD;
- runtime profile: 48 GiB RAM cache, 18 GiB VRAM expert cache;
- worker: four request slots, 4096-token maximum context.

## Qualified hot-path gates

The final full-model gate produced:

| Gate | Result | Required |
|---|---:|---:|
| single-stream hot runtime | 47.6703 tok/s | 10 tok/s |
| four-request aggregate hot runtime | 42.5976 tok/s | 30 tok/s |

The gate used a warmup in the same process, retained expert residency, then
reset request state and measurement counters. It observed no expert-weight H2D
in the measured hot window, no pagefile growth, identical batched versus
isolated token sequences, and exact chunked-prefill versus scalar-prefill
sequences on the full model.

The mixed batch contains prompt lengths 3, 3, 4, and 7. With a four-token
prefill chunk, the seven-token case crosses a `4 + 3` boundary and proves that
KV and recurrent DeltaNet state survive more than one chunk.

### Compact CPU-result path

The mixed batch previously copied a dense masked CPU output buffer whenever any
selection used the CPU. Compact selection mapping changed only transport and
aggregation storage:

| Measurement | Dense masked | Compact |
|---|---:|---:|
| CPU selections | 9,281 | 9,521 |
| CPU-result H2D | 521,830,400 B | 77,996,032 B |
| aggregate throughput | 41.6653 tok/s | 42.5976 tok/s |

The compact byte count is exactly `9,521 × 2,048 × 4`; no unused top-k slots
are transferred. Despite minor route/cache-count variation between runs, H2D
fell 85.05%, output equality remained exact, and pagefile growth remained zero.

### CPU/GPU expert-lane telemetry

A subsequent run added one CUDA event pair per layer around resident expert
execution. Throughput was 42.5398 tok/s, a 0.14% difference from the compact
path's 42.5976 tok/s. This is within run variation and leaves the 30 tok/s gate
with substantial headroom.

| Measurement | Result |
|---|---:|
| CPU expert selections | 9,523 |
| resident GPU expert selections | 58,157 |
| CPU expert time | 1.07279 s |
| resident GPU expert time | 0.404417 s |
| CPU time per selection | 112,652 ns |
| GPU time per selection | 6,953.87 ns |
| observed CPU/GPU overlap | at least 0.136184 s (33.67% of the shorter lane) |
| compact result H2D | 78,012,416 B |
| compact map H2D | 320,950 B |

GPU time is measured by CUDA events; CPU time and the enclosing expert phase
use a steady host clock. The overlap is therefore reported as a conservative
lower bound, not a full CUDA timeline. Transfer and aggregation remain in the
enclosing phase. These measurements are inputs to the dynamic scheduler, not a
fixed CPU/GPU split policy.

### Dynamic dispatch gate

After connecting the bounded critical-path planner, the same frozen-placement
batch gate produced 42.2259 tok/s with exact interleaved and chunked-prefill
outputs. The measured window contained 1,693 split-layer plans and 56,198
unique-expert candidates: 47,861 were already resident on GPU and 8,337 were
CPU-only RAM misses. No plan was rejected; there was no expert SSD read, expert
H2D, or pagefile use.

The measured window correctly contains no CPU-vs-upload cost wins because
placement is frozen after warmup. This is a policy boundary, not evidence that
the cost branch is dead: deterministic runtime tests exercise CPU, GPU,
transfer, saturation, and tie outcomes, while runner JSON separately reports
pre-measurement decisions. Cold/warm service results will be qualified in a
service stage and must not be compared directly with this frozen result.

The unfrozen warmup recorded 3,896 CPU critical-path wins, zero GPU-upload
wins, 10,963 forced cold GPU paths, and two CPU-only paths. This is the expected
decision for the measured short microbatch: a synchronous expert upload does
not amortize before CPU compute completes. Larger row groups or a future
overlapped uploader can cross that measured boundary without changing policy.

### Score-aware cache temperature

Adding exact selected-routing-score evidence to cache admission produced
42.893 tok/s on the same frozen mixed batch. Exact interleaved/chunked output,
zero pagefile use, zero measured SSD/H2D, and zero rejected dispatch plans were
preserved. Score collection occurs only during unfrozen placement; the
qualified hot window performs no extra score transfer or policy mutation.

The run's placement distribution changed within normal warm-cache variation
(51,284 VRAM hits and 8,136 RAM hits). This single run does not claim a cache
hit-rate improvement. Its acceptance evidence is semantic and structural: the
full model remains correct, while a deterministic pressure test proves that an
equal-frequency, higher-score expert is retained over its lower-score peer.

### Bounded prefetch policy

The first attributed predictor admitted after one observation and reported 9
useful versus 107 wasted predictions. Requiring two recent observations cut
wasted traffic from 338,345,984 B to 243,482,624 B. Its qualified run passed at
42.1741 tok/s, with a 86.08% VRAM hit ratio, exact outputs, zero measured-window
expert H2D, and zero pagefile use.

A direct run with prefetch disabled produced 38.3828 tok/s and an 83.94% VRAM
hit ratio. It had zero predicted/wasted bytes, but moved 10,658 selections to
CPU versus 9,353 with bounded prefetch. This A/B is why the one-credit predictor
remains enabled in the balanced baseline despite low strict next-use precision.
Its pollution telemetry remains a production tuning signal, not a success
metric to optimize in isolation.

### CPU executor autotuning

The bounded first-batch tuner evaluated four configurations in 1.5543 ms and
selected all 12 logical workers with 64-output gate/up and 128-output down
jobs. Persistent scratch, row-pointer reuse, and AVX2 software prefetch then
produced:

| Measurement | Result |
|---|---:|
| aggregate throughput | 42.3659 tok/s |
| CPU expert selections | 9,361 |
| CPU executor calls | 1,675 |
| CPU time per selection | 109,714 ns |
| effective weight bytes | 26,148,372,480 B |
| CPU compute time | 1.0259 s |
| effective weight traversal | 25.4882 GB/s |

The immediately preceding bounded-prefetch gate measured 113,501 ns per CPU
selection. The new result is 3.34% lower while retaining exact output, zero
measured expert SSD/H2D, and zero pagefile use. Effective traversal counts each
expert record once per grouped execution and may include cache reuse; it must
not be presented as measured socket DRAM bandwidth.

### Operator placement contract

The first profile-aware `balanced` batch gate passed at 42.2958 aggregate
tok/s. Its artifact reports the effective 48 GiB RAM and 18 GiB VRAM budgets,
prefetch enabled, and two-observation threshold. Isolated/interleaved/chunked
outputs matched; the measured window had zero expert SSD reads, zero expert
H2D, zero pagefile use, and at least 25.85 GiB free physical memory.

This is the qualified `balanced` result. The earlier one-observation predictor
and prefetch-off A/B establish policy direction for `latency` and `capacity`,
respectively, but are not independent SLO qualifications for those profiles.

This gate uses paged FP16 KV and online-softmax attention. One 6 MiB page was
physically sufficient for each short gate request. The batch run retained exact
batched-versus-isolated token equality, used no pagefile growth, and kept at
least 24.3 GiB physical RAM free.

These numbers are runner throughput under a qualified hot placement. They are
not a promise of 31 user-visible tokens/s for a cold chat.

## Service smoke

The API deployment smoke covered four concurrent completions, legacy streaming,
Responses JSON and typed streaming, Chat Completions, usage streaming,
capability errors, identity, and disconnect cancellation. A representative
cold operational window reported:

- 21 decode batches / 32 rows;
- effective decode batch: 1.5238;
- TTFT p95: 16.203 seconds;
- inter-token p95: 2.375 seconds.

Smoke latencies include cold expert placement and short, heterogeneous requests.
They demonstrate lifecycle behavior, not the hot throughput SLO. The large gap
between hot gate and cold API latency is an open production problem.

## Real text probe

A streaming Chat Completions probe used a 25-token templated English prompt and
requested 32 tokens. Two identical sequential requests on the already-running
service produced:

| Placement state | TTFT | Total | Approx. post-first-token rate |
|---|---:|---:|---:|
| first request after heterogeneous smoke | 25.664 s | 56.216 s | 1.01 tok/s |
| immediately repeated request | 0.807 s | 2.429 s | 19.12 tok/s |

The output text and token usage were identical. The warm prefill processed the
25-token prompt at roughly 31 prompt tok/s, but this is not yet a formal prefill
throughput gate. The result exposes two separate remaining costs: cold expert
placement dominates the first request, while a longer prompt/output working set
still trails the short synthetic runner's fully hot 47.67 tok/s decode rate.

## Memory observation

At startup/cold idle, total GPU usage was approximately 5.7 GiB: dense weights,
FP32 request state, CUDA context/workspace, and Windows display. After the smoke
heated the expert cache, total usage reached approximately 17.4 GiB. Stopping
the full process tree returned the GPU to approximately 421 MiB used by the
Windows desktop.

## DeepSeek-V4-Flash route trace

### Census-driven warm start

The production worker now consumes its authenticated route census before
readiness. After one qualified five-token `Hi` path populated 215 routed-layer
observations, restart selected 864 unique expert-layer records. It loaded
11,551,113,216 routed bytes in 6.27 seconds and reported the placement state as
`ready` rather than merely `observing`.

Repeating the same five-token path after a second warm restart produced the
same token `19923` and completed without a scheduler suspension, routed acquire,
RAM placement, or CPU placement. The five model steps took 340.886 ms in total,
or 68.177 ms/model step (14.67 model steps/s). Controller advancement accounted
for 323.820 ms, scheduler polling for 329.084 ms, and the complete output heads
for 11.149 ms. These nested counters overlap their parent interval and are not
summed.

This is a warm, route-repeated model-step result, not a 14.67 tok/s arbitrary
chat claim. It proves that learned startup residency removes SSD/H2D from a
known route and also shows that the remaining GPU/controller path still needs
about a 2.04x reduction to reach a 33.3 ms/model-step target.

Moving the hybrid workspace to request ownership, assigning a non-blocking
stream per request, and replacing per-step RoPE generation/upload with a
resident context table preserved token `19923`. The same five-step warm route
took 341.509 ms versus 340.886 ms before the change, effectively unchanged for
single-stream decode. This is expected: it removes allocation and default-stream
coupling needed for concurrency, but the existing per-layer directory barriers
still serialize one request. No single-stream speedup is claimed for this
slice.

A subsequent barrier-attribution run again emitted token `19923` and executed
215 layers over five model steps. Wall time inside the model step was 406.708 ms
and controller advancement was 377.690 ms. The controller partition was:

| synchronized controller interval | total | per layer |
| --- | ---: | ---: |
| attention/router host submission | 38.855 ms | 0.181 ms |
| directory plan, including attention/router drain | 201.041 ms | 0.935 ms |
| FFN host submission | 11.269 ms | 0.052 ms |
| directory release, including FFN drain | 126.442 ms | 0.588 ms |

The two drain intervals account for 86.7% of controller time. This measurement
does not imply that synchronization alone costs 327 ms: it includes the GPU
work that precedes each barrier. It does prove that the host serializes every
layer twice and that request streams cannot overlap while the directory uses a
shared scratch allocation. The next runtime slice is therefore a per-request,
event-driven directory transaction; GPU-event timings remain necessary before
choosing the first kernel to replace.

The first asynchronous slice gave every active pin a fixed device metadata
slot and completion event, then enqueued reference release after FFN instead of
draining the request stream. The same route emitted token `19923`; release calls
fell from 126.442 ms to 1.212 ms over 215 layers. As expected for a dependent
single stream, most work moved to the next planning drain: plan time rose from
201.041 ms to 310.988 ms. Whole-model time was 369.761 ms (13.52 model steps/s),
which is about 9.1% lower than that immediately preceding 406.708 ms sample but
slower than the earlier 340.886 ms best sample. No stable single-stream speedup
is claimed. The change is retained because it removes one host barrier per
layer and makes inter-request overlap possible; private asynchronous planning
is still required to realize that concurrency.

Request-private hash/pin/miss scratch and a persistent plan event then replaced
the remaining in-controller stream synchronization. An initial implementation
was rejected because it busy-polled the event 835,494 times for 215 layers. The
service loop now submits every runnable request stream, parks on one pending
event only when no other runnable work remains, and resumes through an explicit
`pending_cuda` scheduler state. The qualified single route emitted token
`19923` with exactly 215 pending polls and 215 waits. Five model steps took
342.421 ms, or 68.484 ms/model step (14.60 model steps/s), effectively matching
the previous 14.67 best sample while freeing the control core from spin.

A two-request lifecycle gate then exercised independent request streams and
scratch through warm planning, concurrent execution, cold miss suspension,
bounded acquisition, and resume. Both requests emitted the same two tokens
(`19923`, then `3`); 76 suspensions and 188 acquisitions completed without a
duplicate release, corrupted route, or leaked active request. This qualifies
concurrent lifecycle correctness, not aggregate throughput: the setup still
prefilled the two prompts sequentially and the extra route exceeded the learned
warm set.

An opt-in CUDA-event boundary profile then measured the accepted five-step,
215-layer zero-miss route without Nsight kernel-trace inflation. It emitted
token `19923` and reported:

| device interval | 5 steps | per model step | share of measured GPU phases |
| --- | ---: | ---: | ---: |
| attention + route + directory plan | 200.996 ms | 40.199 ms | 62.6% |
| FFN + asynchronous release | 119.975 ms | 23.995 ms | 37.4% |
| total measured GPU phases | 320.971 ms | 64.194 ms | 100% |

Worker model-step wall time was 345.389 ms, or 69.078 ms/step, under the
profiling events. The remaining ~4.88 ms/step covers embedding, output head,
host scheduling, and event instrumentation. This proves the warm route is now
GPU-compute dominated; storage, H2D, CPU experts, and directory misses were
zero on the measured path. Reaching 33.3 ms needs material reductions in both
major device phases, not another host-directory micro-optimization.

Intermediate events, recorded without another synchronization, refined the
same zero-miss boundary on a subsequent run:

| detailed device interval | 5 steps | per model step | share |
| --- | ---: | ---: | ---: |
| attention | 151.658 ms | 30.332 ms | 46.1% |
| router | 36.264 ms | 7.253 ms | 11.0% |
| directory plan | 20.384 ms | 4.077 ms | 6.2% |
| FFN compute | 119.387 ms | 23.877 ms | 36.3% |
| directory release | 1.304 ms | 0.261 ms | 0.4% |

The detailed intervals total about 65.80 ms/step; the worker took 69.83
ms/step under profiling. Directory plan plus release is now only 6.6% of GPU
time. The optimization order is therefore attention, FFN compute, router, then
directory only if its share grows after the compute reductions.

A block-per-row F32 GEMV was also tested for HCA's underfilled 24×16384
projection. The repeated real HCA slice improved from 0.077568 to 0.062925
ms/site and stayed far inside its `2e-4` local oracle tolerance. Nevertheless,
the full-model route incurred 15 acquisitions against the accepted census:
the changed FP32 reduction order altered later expert IDs even though the final
token remained `19923`. The kernel was removed. As with dense GEMV, local
numerical tolerance and token equality are insufficient routing gates.

An attempted all-device continuation placed a `missing_count` guard in every
routed, aggregate, shared, and HCA CUDA kernel so a hot plan could execute FFN
without returning to the host. Token `19923` and the zero-miss contract were
preserved, but two repeated measurements took 411.763 ms and 396.999 ms for the
same five model steps. The latter is 15.9% slower than the 342.421 ms
request-private-event result. CUDA 12.1 on the target host exposes no conditional
graph node API, and the per-block global predicate is too expensive for this
batch-one path. The implementation was removed before commit. A future guarded
transaction requires a cheaper conditional launch mechanism; it must not
reintroduce this rejected branch in every FFN block.

### Warm-route CUDA profile

Nsight Systems traced one five-step, 215-layer zero-miss route on the accepted
request-private-event runtime. Startup was included, so the 365-instance
`fp8_to_int8` admission kernel and startup H2D traffic are excluded from decode
conclusions. Kernels whose instance counts match the 215/430-layer execution
show the following approximate split under profiler overhead:

| kernel family | traced GPU time | share of correlated decode kernels |
| --- | ---: | ---: |
| dense INT8/BF16/F32 GEMV | 194.5 ms | ~55% |
| routed packed-FP4 gate/up + down | 96.3 ms | ~27% |
| shared-expert gate/up + down | 35.4 ms | ~10% |
| HCA, routing, attention and small transforms | remainder | ~8% |

The profile changes the optimization order: routed FP4 is not the majority of
the warm critical path, and directory hash/pin/release kernels total well under
2 ms in the traced five steps. Dense GEMV representation and execution must be
addressed before another directory micro-optimization can plausibly approach a
2x single-stream improvement.

Nsight Compute then inspected one 128-block `int8_gemv_vector` launch. It
measured 10.43 us, 404.05 GB/s, 48.36% DRAM throughput, 25.72% achieved
occupancy, and only 0.26 waves/SM. Scoreboard waits on memory dependencies
accounted for about 60.2% of cycles between issued instructions. This proves
the launch has unused machine capacity, but not that a split-K reduction is
automatically faster.

A two-warp-per-row variant doubled exposed warps for matrices with at most
2,048 rows and at least 2,048 columns. On a clean census it preserved token
`19923` and produced a zero-miss route, but took 346.860 ms for five steps,
slower than the accepted 340.886–342.421 ms range. Its different FP32 reduction
tree also changed later router selections, so its census was not warm for the
original kernel. The variant was removed before commit and the pre-profile
census was restored. Future dense work must gate routing stability as well as
final-token equality and speed.

A second dense experiment kept the original one-warp reduction and per-lane
FP32 accumulation order, but loaded two coalesced `char4`/`float4` segments
before their FMAs to expose more independent memory work. The optimized run
took 347.028 ms for five steps. Rebuilding the original kernel and running it
against exactly the same accumulated census produced zero misses, the same
token `19923`, and 334.189 ms. Thus routing stayed stable, but the extra live
values made the end-to-end path about 3.8% slower. The variant was removed and
the pre-experiment census restored. The measured scoreboard stalls therefore
cannot be converted into a win by local unrolling alone; the next dense change
must alter the representation or use an architecture-suited dot-product path.

Q8 activations plus an SM86 DP4A INT8-weight GEMV were then measured on every
distinct layer-0 INT8 attention geometry. Each number is the mean of 100 warm
iterations; the Q8 column includes activation quantization:

| projection geometry | FP32 activation | Q8 + DP4A | result |
| --- | ---: | ---: | --- |
| 512 × 4096 | 0.00940 ms | 0.01907 ms | 2.03x slower |
| 1024 × 4096 | 0.01079 ms | 0.02017 ms | 1.87x slower |
| 8192 × 4096 | 0.04440 ms | 0.05309 ms | 1.20x slower |
| 4096 × 8192 | 0.04867 ms | 0.06203 ms | 1.27x slower |
| 32768 × 1024 | 0.04273 ms | 0.04747 ms | 1.11x slower |

The candidate's Q8-versus-accepted RMSE ranged from `3.63e-4` to `1.00e-3`,
but accuracy is irrelevant because no geometry won. The accepted kernel already
streams one-byte weights efficiently; shrinking the activation does not shrink
the dominant matrix read. The Q8/DP4A experiment was removed before commit.
Future dense work must first reduce authenticated weight bytes or otherwise
reuse them across real rows/requests, then qualify the changed representation
against the independent model oracle.

The subsequent read-only compact-weight screen evaluated symmetric 4/5/6-bit
integers and FP4 E2M1, each with FP16 scales at block sizes 32, 64, and 128.
Four deterministic activation probes were projected through every distinct
layer-0 INT8 attention geometry. Block 32 was the most accurate candidate in
every family:

| candidate | storage ratio | projection-relative RMSE across 5 shapes |
| --- | ---: | ---: |
| symmetric 4-bit | 1.78x | 9.55–10.45% |
| FP4 E2M1 | 1.78x | 9.99–10.64% |
| symmetric 5-bit | 1.45x | 4.67–4.94% |
| symmetric 6-bit | 1.23x | 2.25–2.40% |

These are screening errors, not model-quality scores, but they are already far
outside the existing independent block-oracle budgets. The naive formats are
therefore rejected before kernel implementation. The screening command and
Windows runner remain as a fail-closed gate for future calibrated schemes; the
next compact candidate must use representative activation statistics and pass
the attention/full-model oracle before it can become a runtime ABI.

An eight-token single-stream diagnostic measured the placement problem rather
than claiming a throughput gate. With the 64-slot global routed cache it ran at
0.311 tok/s, performed 2,597 cold acquisitions, and read 35.80 GB. Exact route
IDs showed 46.5% consecutive reuse, rising to only 75.2% when the previous
seven routes were considered. That seven-route INT8 working set would consume
22.80 GB before dense, shared, request, or CUDA memory.

This rejects a larger history-only INT8 cache on a 24 GB GPU. The trace justifies
the compact RAM/VRAM hierarchy described in the DeepSeek backend document; it
does not justify comparing the diagnostic rate with the qualified Qwen hot
batch results above.

Enabling the first bounded tier with a 32 GiB pageable RAM budget preserved the
same eight generated token IDs. Total expert reads fell from 35.80 GB to 20.40
GB; 7.94 GB occurred after prefill, with 1,152 RAM-cache hits. Decode improved
from 0.311 to 0.516 tok/s, while the cache high watermark remained 20.40 GB.
This validates the RAM tier and also shows that repeated H2D plus FP4→INT8
admission remains far from the latency target.

Adding an independent 8 GiB routed compact-VRAM L1 preserved the same output
and left about 4.75 GB VRAM free. It recorded 1,003 hits, 1,594 misses, 952 LRU
evictions, and 21.31 GB compact H2D. Decode moved only to 0.553 tok/s. The small
gain rejects H2D-only optimization as the next focus: expanded-slot admission
and route-safe compute residency must be fixed before kernel throughput claims.

The route-safe 258-slot compute working set then completed the same eight-token
run without the stalls seen in earlier large-cache experiments. It eliminated
840 admissions through exact consecutive reuse, but 966 remained; 594 were
first-touch records and caused the same 7.94 GB of post-prefill SATA reads.
Decode measured 0.557 tok/s with identical output. This establishes the current
hardware lower bound: first-touch storage, not retained-route eviction or GEMV,
dominates the single stream.

## DeepSeek direct packed-FP4 CUDA gate

The replacement compact kernel keeps routed weights in their 13,369,344-byte
FP4/UE8M0 representation, quantizes activations to Q8 once, and performs
coalesced SM86 DP4A dots without a global INT8 expert allocation. On the real
layer oracle:

| measurement | expanded INT8 | direct packed FP4 |
| --- | ---: | ---: |
| seven-expert resident bytes | 176,390,144 | 105,414,656 |
| FFN execution | ~6.98 ms | 4.39 ms |
| block maximum error | `4.77e-4` | `6.96e-3` |

The compact result passed the declared `1e-2` block tolerance and retained the
real `Hi` → `Hello` full-model output. The full prompt used the original
six-extent checkpoint and rebuilt its working set, so its wall time is not
compared with the earlier compact-pack steady-state number.

## DeepSeek all-core packed-FP4 CPU gate

The CPU lane consumes the same retained 13,369,344-byte records directly. It
uses a persistent pool pinned across the 12 logical processors on the target
Windows host, Q8-quantizes each activation once, and does not allocate a full
INT8 copy. A varied-weight deterministic fixture first caught and rejected an
incorrect SIMD nibble permutation that a uniform fixture had hidden.

The real layer gate then executed all six routed experts on CPU and compared
every one of the 24,576 output values with the direct packed-FP4 CUDA path:

| measurement | result |
| --- | ---: |
| logical CPU workers engaged | 12 / 12 |
| six-expert CPU execution | 48.162 ms |
| six-expert CUDA execution | 4.387 ms |
| CPU versus CUDA RMSE | 0 |
| CPU versus CUDA maximum error | 0 |

This qualifies correctness and resource use, not a CPU replacement for the hot
GPU path. The placement policy should assign a compact RAM expert to CPU only
when that shortens the measured critical path versus storage plus H2D, and
should overlap that work with resident CUDA selections.

The subsequent hybrid vertical slice moved one real selection to CPU and left
five on CUDA. A reusable per-request workspace used 98,816 device bytes and
114,688 pinned host bytes. Stable aggregation reproduced the all-CUDA block
bit-for-bit (`RMSE=0`, maximum error `0`). The forced split took 10.54 ms versus
4.40 ms all-CUDA, which rejects CPU use for this single-stream hot route while
qualifying the mechanism for cold or GPU-saturated placement decisions.

The same forced split now passes through the production decode scheduler. The
real-layer gate observed five cold device acquisitions, one typed host resolve,
one peak host lease, and one completed hybrid layer. CPU/GPU output again
matched exactly. This proves the scheduler lifecycle and placement boundary;
the synthetic planner costs used to force the branch are intentionally excluded
from performance claims.

## DeepSeek measured placement profile

The real-layer qualification now separately measures all-core CPU, routed-only
CUDA, and pinned compact-record H2D costs. One run produced 9.342 ms per CPU
expert, 0.543 ms per CUDA expert, and 3.393 GB/s over a 106,954,752-byte H2D
sample. Seeded with those values, the bounded critical-path solver assigned a
six-cold-expert RAM fixture as two CPU plus four upload/GPU decisions, projecting
18.685 ms. This is a placement calibration, not tokens/s, and the fixture does
not replace an end-to-end cold/warm service gate.

## DeepSeek durable-pack service gate

The final 43-shard compact pack contains exactly 147,169,738,752 routed bytes.
The production worker bundle passed one real HTTP completion with 4096 maximum
context, two worker slots, 40 GiB RAM cache, 12 GiB VRAM cache, and balanced
placement:

| quantity | source extents | compact pack |
| --- | ---: | ---: |
| prompt tokens | 1 | 1 |
| output tokens | 2 | 2 |
| request wall time | 30.52 s | 17.24 s |
| output | `, I` | `, I` |

The compact layout reduced wall time by about 43.5% for this cold functional
slice. The production run reported 17.20 seconds TTFT and 0.016 seconds between
the two API emissions because protocol 4 synchronously prepares the following
token before returning the current one. Neither value is a 30 tok/s result.
The gate's purpose is model identity, complete-pack use, real decoding, service
accounting, and resource cleanup.

## Benchmark rules

Any published result must include:

- exact Git commit and model content hash;
- hardware, driver, CUDA version, RAM/VRAM/cache budgets;
- prompt/context/output lengths and concurrency;
- cold, warm, or frozen placement state;
- whether tokens/s means one request, aggregate requests, or model forward rows;
- TTFT and inter-token latency alongside throughput;
- correctness comparison and memory/pagefile evidence.

Do not compare results that use different cache states as though they measured
the same workload.
