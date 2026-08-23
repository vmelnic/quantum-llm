# AGENTS.md

## User preferences (durable)

- No effort estimates. The user is not interested in "engineering days",
  story points, or stage durations. Do not estimate; plan work as a flat
  list of items that all need to be done, ordered by dependency, and then
  implement and test them.
- Do not split work into endless small patches ("peticiri"); implement the
  real change, then verify it.
- For each change, make the complete plan, implement it, and test it. If the
  result fails, allow at most one local corrective patch. If it still fails,
  stop patching, reassess the system and the intended solution end to end,
  then start a new plan/implementation/test cycle from evidence rather than
  guesses. Repeat that loop until the real solution is found or a concrete
  external blocker is established.
- Communicate in Romanian; repository artifacts (code comments, commit
  messages, docs) stay in English.
- Hugging Face downloads only via xet through the repo scripts
  (`ops/windows/Start-HuggingFaceModelDownload.ps1` /
  `Get-HuggingFaceModelDownload.ps1`). Never curl/parallel ad-hoc
  downloads, never invented scheduled tasks, and do not set
  `HF_HUB_DISABLE_XET`.

## Goal fidelity (mandatory)

- Every implementation, instrumentation change, test, and benchmark must be
  tied explicitly to the active goal, an acceptance criterion, or a measured
  bottleneck that blocks it. Do not build infrastructure, run experiments, or
  optimize components merely because they are interesting or available. State
  what result will advance the goal and discard the direction when the result
  does not materially improve that path.
- The serving target is self-contained on the 3090box. There is no external
  owner, remote expert executor, distributed fallback, or outside coordinator
  in the target architecture. Do not treat one as a present or future
  dependency, escape hatch, blocker resolution, or required user decision.
- Treat the user's requested format, execution mode, and infrastructure shape
  as acceptance criteria, not implementation suggestions. Before editing,
  state the relevant invariants in the plan (for example FP4 versus INT8,
  demand paging versus full residency, and universal runtime versus a
  model-specific integration). Do not silently replace one with an easier
  baseline.
- A shortcut that changes an acceptance criterion requires explicit user
  approval before implementation. A useful experiment is not completion of a
  different requested goal.
- Report storage and compute formats exactly. Never label an INT8 artifact as
  INT4/FP4. For mixed artifacts, report the format of routed experts, dense
  matrices, embeddings, routers, and runtime intermediates separately.
- An FP4 goal is complete only when the produced artifact contains the
  requested FP4 payload, the selected execution provider actually executes
  that FP4 ABI, numerical quality is checked against the reference model, and
  the real service path passes. Compiling or running an INT8 baseline does not
  satisfy it.
- A demand-paging goal is complete only when the runtime loads selected
  experts on demand and telemetry proves the actual RAM/VRAM/storage traffic.
  Loading the full model at startup is a resident-mode test and must never be
  presented as validation of paging.

## Inference feasibility and failure discipline (mandatory)

- The active Qwen3.8 serving acceptance target is one self-contained RTX 3090,
  262,144 actually populated tokens, exact F16 KV semantics, and approximately
  15 generated tokens/s on the real coding harness. Cold prefill, reused-prefix
  prefill, and decode are separate gates; meeting one must not be reported as
  meeting another.
- Before proposing or implementing an inference optimization, write the
  capacity and steady-state traffic equations for the exact requested model,
  active context, numerical formats, hardware, and harness. At minimum account
  for weights, recurrent state, KV, draft/MTP state, execution buffers, bytes
  touched per accepted token, device bandwidth, interconnect bandwidth, and
  speculative acceptance. A runtime feature cannot override a failed capacity
  or bandwidth inequality.
- Keep `maximum context`, `active populated context`, and `harness prompt
  overhead` separate in every measurement and claim. A server accepting a
  context limit does not prove that the limit was populated, that its KV was
  resident, or that decode at that length is fast.
- Classify the working set before designing paging. Dense weights are hot on
  every token. Every retained K/V entry of a full-softmax-attention layer is
  hot on every decode token. Sparse MoE experts that were not routed are cold.
  Demand paging is valid only for data that is demonstrably not consumed on
  the current critical path; paging all active full-attention KV or dense
  weights is capacity recovery, not a throughput solution.
- Do not replace the real harness gate with a tiny direct-API prompt. Report
  cold prefill, reused-prefix prefill, decode, tool-loop wall time, populated
  context, and residency separately. Compare against the last real harness
  result and reject a direction that produces no material end-to-end gain.
- Numerical fidelity is an acceptance criterion. KV quantization, attention
  sparsification, token eviction, reduced top-k, low-rank approximation,
  summarization, or a different model are not lossless optimizations. Do not
  propose or implement them for an exact/fidelity-preserving goal without
  explicit user approval and an explicit quality gate.
- Research mechanisms must have a fail-fast quantitative gate before runtime
  infrastructure is built. Examples: required compression ratio, accepted
  speculative tokens per verification, exact-pruning fraction, transfer bytes
  per accepted token, and end-to-end harness throughput. Stop the direction
  when the measured prerequisite fails.
- Record negative results and corrected claims in the active research document
  so that a later session cannot repeat a disproven approach.
- Session retention is a provider capability, not a universal runner
  assumption. The common service must advertise it only when every selected
  callable provider implements exact checkpoint and rewind. Dense FP4 does;
  the current callable DeepSeek provider does not. Non-retaining requests must
  not receive `CHECKPOINT`, `RESUME`, `RETAIN`, or `DROP` commands.
- On the 24 GiB RTX 3090, the shared 13 GiB routed-VRAM profile fails the
  DeepSeek preflight after fixed allocations plus the 1 GiB reserve. The
  validated common profile is 12 GiB; do not raise it without re-running both
  real `model.sh chat` gates and cleanup.

### Qwen3.8-27B / RTX 3090 failures that must not be repeated

- The repository's completed 262,016-token prompt plus 128-token generation
  proved capacity only. It used the experimental FP4 KV representation, took
  5,423.422 seconds to first token, and decoded at about 3.24 tok/s afterward.
  It is neither the exact-F16 fidelity baseline nor a production-speed result.
  The old client-side `135.91 tok/s` value combined timing phases and is
  invalid.
- The `UD-Q4_K_XL` payload is about 16.35 GiB, while F16 KV for the 16
  full-attention layers at 262,144 populated tokens is 16.00 GiB. MTP adds
  about 1.28 GiB plus draft KV, before execution buffers. Configuring
  `--fit-ctx 262144` on a 24 GiB RTX 3090 therefore forced CPU placement; it
  did not validate full-GPU 262K serving.
- The Unsloth/llama.cpp path did not materially improve the real harness:
  the prior approximately 90-second first turn became 88.4 seconds. The real
  26,193-token Claude Code request decoded 29 tokens at 2.16 tok/s; the 6.28
  tok/s number came from a 57-token direct request and must not be reported as
  harness performance.
- Claude Code contributed about 26K input tokens before relevant project code,
  while a locally measured minimal Pi `hi` session used about 2.7K including
  cached input. Harness choice is part of the memory and prefill equations.
- The 2026-08-21 Claude Code qualification failed both latency and quality.
  A cold `hi` request carried 29,487 prompt tokens, reached its first token in
  82.875 seconds, and completed 143 generated tokens in 88.281 seconds. A
  request asking for a four-stanza poem about Moldovan president Maia Sandu
  carried 29,660 prompt tokens, reached its first token in 83.454 seconds, and
  returned a fabricated mathematician biography after 94.157 seconds. Do not
  call the current Qwen3.8 FP4 service Claude-compatible or production-ready.
- The same Maia Sandu prompt also failed without Claude Code. With only 100
  prompt tokens, FP8 KV consumed its 320-token output allowance entirely in
  hidden reasoning, emitted no visible answer, and invented biographical
  claims. The attempted F16 comparison used a direct diagnostic worker with
  greedy fixed-length generation, no service sampling and no EOS handling; it
  is not a `model.sh chat` result and cannot attribute the FP8 failure to the
  weights or model execution. Never cite that diagnostic as F16 chat evidence.
- The corrected real F16 service gate used `MODEL_KV_CACHE_DTYPE=fp16` and
  `./ops/model.sh chat`, with no harness prefix. `hi` used 53 prompt tokens and
  generated 30 tokens in 3.079 seconds with 2.125-second server TTFT (about
  30.40 tok/s after the first generated token). The four-stanza Maia Sandu
  prompt used 68 prompt tokens and generated 476 tokens in 19.047 seconds with
  2.812-second server TTFT (about 29.26 tok/s after the first generated token)
  and returned four stanzas without the fabricated mathematician biography.
  This is a short-context F16 service pass, not a 262K throughput or reference
  quality qualification. The terminal client's contemporaneous 85.84/60.44
  figures mixed hidden reasoning counts with first-visible-content timing and
  are invalid.
- FP8 E4M3 per-token/per-head target KV plus FP4 MTP KV physically fit all
  262,144 positions in 1,024 pages on the RTX 3090. The real 262,016-token
  prompt took 1,168.594 seconds to prefill and decoded 128 non-speculative
  tokens in 41.774 seconds (3.064 tok/s). This is capacity recovery, not the
  15 tok/s goal, and it is not eligible for deployment promotion.
- A cancelled resumed Anthropic request currently destroys its checked-out
  retained worker session. In the measured failure, a request resumed with a
  90-token delta, generated 872 tokens, disconnected, and the following
  request cold-prefilled 29,660 tokens. Preserving an exact base checkpoint
  would fix this latency bug, but must not be presented as a remedy for the
  independent factual/tool-loop quality failure.
- Moving the model from Windows DrvFS (`/mnt/d`) to native WSL ext4 fixed a
  severe load-path problem but did not solve steady-state decode. Storage load
  time and token-generation bandwidth are separate bottlenecks.
- Q4 KV was proposed after fidelity had already been required. That proposal
  was invalid: it changes cached activations and its degradation was already
  confirmed by the user.
- Lazy/paged F16 KV is useful only while the active KV still fits beside the
  weights. At fully populated 262K, all 16 GiB of full-attention KV remains hot
  per decode token; moving it through RAM/NVMe per token collapses throughput.
- The single-token bandwidth lower bound was calculated too late. At 262K,
  approximately 17.56 GB of encoded weights plus 17.18 GB of F16 KV are touched
  per token. This exceeds the RTX 3090's 936 GB/s peak at 30 tok/s even before
  kernel overhead. A valid future design must amortize work across multiple
  accepted tokens or reduce exact bytes by a demonstrated lossless mechanism;
  ordinary offload, another wrapper, or another server cannot satisfy it.
- Isolated CUDA probes established useful kernel and bandwidth ceilings, but
  the real maximum-context service remained at about 3.24 tok/s. Never promote
  a microbenchmark ceiling, a metadata/parser smoke, or post-first-token client
  arithmetic into an end-to-end service result.
- CPU/GPU organ splitting cannot be valued at theoretical DDR bandwidth.
  Whole active KV on CPU is bounded by the RAM scan and CPU attention compute;
  whole active KV on NVMe or copied through PCIe per token is slower still.
  Heterogeneous placement is valid only when compute stays with its shard and
  a measured critical-path overlap beats the current service.
- Speculative draft weights alone are not their complete memory cost. Their
  long-context state, target-feature cache, verification tree, rollback state,
  and CUDA workspaces must all be included before claiming that a DFlash/EAGLE
  or MTP configuration fits beside the target and populated KV.
- SplitZip's published 1.32x lossless ratio applies to BF16 KV, whose exponent
  has eight bits. It is not evidence for the requested IEEE FP16 KV, whose
  exponent has five bits. Never use the BF16 ratio in an FP16 feasibility
  equation unless an FP16 capture from this model independently demonstrates
  it.
- The 25%-KV oracle rank trace was only a necessary upper bound, not evidence
  that a causal proposer could construct the tree. The corrected real MTP
  self-rollout gate on the 262,016-token coding prompt retained all constructed
  prefixes (`H=32`, `N=128`) and achieved only 4.565 mean exact accepted tokens
  per cycle, with maximum 11. The proposer itself cost 0.667 s/cycle; combined
  with the 0.963 s missing-F16 transfer floor, its optimistic ceiling before
  verifier compute was 2.800 tok/s. This proposer is rejected. Do not build the
  streamed verifier behind it, repeat the invalid leaf-only accounting, or
  present beam/horizon tuning as a route from 4.565 to the required 24.
- A full-context FP4-KV block-Jacobi proposer also failed before the 262K run
  was justified. With one native-MTP seed, three 32-row target corrections,
  and at most 128 retained prefix nodes, the real RTX 3090 short-context gate
  accepted only 3.25 tokens/cycle on average (range 0--6). Its one allowed
  Tensor-Core correction reduced proposer time from about 1.11 to 0.972
  seconds/cycle. Even impossible perfect 32/32 acceptance would then be capped
  at 14.18 tok/s by proposer time plus the 1.284-second full-F16 transfer,
  before verifier compute; the observed necessary ceiling was 1.55 tok/s.
  Do not run this configuration at 262K, tune its iteration count, or build a
  verifier behind it.
- Static lossless residency also failed on the real 262,144-token F16 KV and
  the artifact-selected FP4 target weights. Favorable per-stream KV entropy
  was 13.5851715504 bits/value; favorable per-tensor FP4 raw/XOR/delta/order-1/
  order-2 coding reduced 13,622,736,896 active weight bytes to an ideal
  12,147,821,704-byte payload. Ideal weights, ideal KV, and the unavoidable
  recurrent state still total 26,893,647,848 bytes, exceeding a 24 GiB device
  by 1,123,844,072 bytes before every codec/runtime overhead. Do not implement
  this static coding family. A different reversible KV transform must first
  demonstrate at most 12.538510 bits/value plus practical allocation margin.
- Reversible structural FP16 transforms were measured on every target K/V
  stream and reached only 13.2659264431 bits/value; the zero-overhead capacity
  threshold was 12.5385101959, and the threshold with observed device use plus
  a 512 MiB runtime reserve was 11.6176117584. Do not implement or retune the
  measured exponent/channel, temporal, palette, exception, or conditioned-byte
  family as a full-residency codec.
- The current staged device-resident FP16 attention provider was measured at
  the complete 262,144-token geometry. Its one permitted split correction
  plateaued at 5.73516 ms per layer and 187.221 GB/s with a 65,536-token split.
  Sixteen attention layers therefore require at least 91.76256 ms, limiting
  attention alone to 10.8977 target calls/s before all other work. Do not tune
  this staged provider further or treat a faster attention kernel alone as a
  capacity solution.
- A favorable exact CPU/GPU split kept losslessly encoded K plus a V prefix on
  the GPU and left a 5,135,630,336-byte F16 V tail for CPU `P dot V`. The full
  AVX2/F16C working-set probe took 226.27925 ms per target call at 12 threads;
  24 threads were slower. Even granting perfect two-token MTP amplification
  and zero GPU/service time caps it at 8.8386 generated tok/s. Do not tune this
  kernel or implement this placement for the 15 tok/s target.
- Keeping every exact F16 K resident and selecting V on demand also fails the
  physical allocation gate. The measured maximum-context exact provider peak
  is about 18,754 MiB and already contains one 512 MiB K layer plus one
  512 MiB V layer. Replacing that pair with all 16 K layers adds 7,168 MiB,
  reaching about 25,922 MiB before any selective-V staging. This exceeds even
  the RTX 3090's 24,576 MiB physical capacity, and its attention-only provider
  floor is already 10.8977 calls/s before dense compute. Do not implement the
  all-K-resident/selective-V design on this host unless a preceding exact
  mechanism first frees and measures the missing capacity.
- Held-out raw-F16 Huffman coding required 13.7195388675 bits/value including
  unseen-symbol literals and codebooks. zlib, Zstandard, and LZ4 were worse.
  Basic previous-layer XOR/delta increased entropy; byte conditioning reached
  only about 13.55--13.61 bits/value versus the practical 11.6176117584-bit
  residency threshold. Do not build these codecs into the runtime.
- An unlimited, zero-cost prompt n-gram oracle over the real 262,016-token
  coding prompt achieved only 3.8485 useful tokens/cycle, maximum 17, with no
  cycle reaching the required 24. Do not implement a prompt-copy proposer for
  this trace.
- Pagewise exact-K progressive refinement with a Q4 KV surrogate failed its
  real 262,144-token lower-bound gate. Even after granting exact V, all other
  K heads, every other attention component, bounds compute, and metadata for
  free, the independent score-interval certificate required at least
  2,029.014182 MiB of one-head F16 K traffic per scalar target call. The
  absolute 15-calls/s transfer ceiling is only 272.822299 MiB, a 7.437127x
  violation. A 32-token page cannot fix it: splitting each 64-token page in
  two can improve the fractional optimum by at most 2x, leaving a proven
  1,014.507091 MiB lower bound. Do not restore this provider instrumentation,
  run the block-32 variant, or build an executable refinement cache behind it.
- AirLLM-style dense layer streaming changes capacity, not the Qwen decode
  hot-byte requirement. Streaming the measured 12.689922 GiB target-call
  weight scan over the 12.46 GiB/s link costs at least 1.018454 seconds per
  scalar call before K/V or compute. Keep layer streaming confined to fitting,
  phase-specific prefill, or genuinely conditional organs; do not propose it
  again as a 15 tok/s dense-decode mechanism.

### DeepSeek maximum-context layer-major gate

- Exact layer-major prefill must retain each routed selection output at its
  original `(row, top-k slot)` until stable top-k aggregation. For the current
  four-stream FP32 frontier, hidden 4096, top-6 route and 1,048,576 tokens, a
  whole-prompt slab is 176 GiB, not 8--16 GiB. Do not describe one hidden
  BF16 slab as the current provider's exact state.
- The bounded valid schedule is eight causal blocks of 131,072 tokens. Each
  block needs 22 GiB plus 6 MiB of exact host scratch and traverses all 43
  layers before the next block; layer attention state persists across blocks.
  This reduces the favorable routed-weight movement from 3.2124 PiB
  token-major to 1.0708 TiB (eight complete compact-pack passes), while exact
  activation traffic remains 23.5156 TiB across the host/device boundary.
  These are a rejected experiment, not a completed provider or measured
  prefill rate. The experimental `layer_major_prefill` planner was removed
  after the populated gate failed; do not reintroduce or advertise
  `causal_layer_major` for DeepSeek without a new end-to-end design and a
  passing populated-context service gate.
- The first real populated gate used 1,048,575 source-derived prompt tokens
  plus one output token. After roughly 6.5 minutes it remained inside block
  one while the RTX 3090 was 86% busy at 22,470 MiB and the worker had
  transferred 33,437,075,614 bytes. It was stopped fail-fast. This disproves
  the claim that routed-pack I/O is still the only active prefill bottleneck;
  do not repeat the scalar dense attention/router/shared-FFN schedule at 1M.
  Exact dense-row batching must pass a bounded parity gate before the next
  populated run.
- Exact two-row batching passed short-path parity and improved `hi` from 6.269
  to 5.443 seconds, but failed the one-block long prerequisite. A 131,071-token
  real repository prompt plus one output remained incomplete after more than
  14 minutes at about 93% GPU utilization and 22,474 MiB. The favorable
  eight-block lower bound therefore exceeds 112 minutes. Do not tune or rerun
  pair widths 2/4/8 as the maximum-context solution. The next schedule must
  separate large dense projection GEMMs from causal row-local state mutation,
  and must pass one complete 131,072-token block before a 1M request.

## Universal MoE infrastructure (mandatory)

- Do not add a model-family runner, scheduled task, service profile, or
  `model.sh` case when the requested goal is a universal runtime, unless the
  user explicitly approves the exception. Model families may have source
  adapters, but the published artifact must drive runtime and deployment
  through capabilities, tensor roles, geometry, quantization ABI, and
  placement policy.
- Common runtime/service code must not branch on Qwen, DeepSeek, LFM, an
  architecture ID, fixed layer counts, or family tensor paths. A new model
  should require an artifact adapter and, only for genuinely new mathematics
  or encoding, a capability/provider implementation—not another execution or
  deployment stack.
- Do not claim that the infrastructure is universal or homogenized while
  supported models still require separate runners, task types, launch
  contracts, or hardcoded model selections. State the remaining divergence
  explicitly.
- Before claiming a universal-path milestone, run the same real smoke prompt
  (`hi`) through `./ops/model.sh chat` for Qwen FP4, DeepSeek, and the newly
  added model where those models are part of the claimed support.
  A metadata/parser test is not a substitute. If any model cannot use the
  common path, the universal milestone is not complete.

## Artifact publication (mandatory)

- Host-dependent model storage paths must be configured once through the
  generic `MODEL_ROOT` environment variable. Never hardcode
  `C:/quantum-llm/work/models`, a user profile, or another machine-specific
  model root in compiler, runtime, service, or operation scripts. Per-model
  locations are resolved relative to `MODEL_ROOT` or discovered from
  manifests; do not introduce separate absolute roots per model family.
- `out/` contains build and experimental outputs. Service-ready model
  artifacts belong under `${MODEL_ROOT}/<stable-name>` and must
  be published transactionally and validated there before configuration is
  changed to reference them.
- Never leave `.env`, service tasks, or defaults pointing to an absent,
  partial, experimental, or unvalidated artifact. Verify the exact remote
  target and manifest after publication, then update configuration, start the
  service, run the real smoke, stop it, and verify GPU/process cleanup.

## Project pointers

- `docs/architecture.md` — the implemented artifact VM and memory-tier
  architecture.
- `docs/benchmarks.md` — the canonical measurement ledger; historical claims
  not present there are not current evidence.
- `docs/research-decisions.md` — rejected directions and the quantitative
  gates that prevent repeating them.
- `docs/roadmap.md` — the current dependency-ordered work plan.
- The Memory Expert (KV-attach) experiment was extracted into the standalone
  public project at `../memory-expert` (validated architecture, results, and
  runbooks live there now).
