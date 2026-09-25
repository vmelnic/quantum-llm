# AGENTS.md

Hot execution contract for coding agents. `MUST`, `MUST NOT`, and `ONLY`
are normative. Keep this file compact; durable evidence belongs in `docs/`.

## 0. Authority and context

1. Follow the current user request, then this file, then the relevant canonical
   document.
2. Do not continue stale roadmap work unless it is part of the current request.
3. Load only the document required by the task:
   - architecture or runtime change: `docs/architecture.md`;
   - performance/result claim: `docs/benchmarks.md`;
   - inference research or optimization: `docs/research-decisions.md` first;
   - backlog/status: `docs/roadmap.md`;
   - deployment readiness: `docs/production-readiness.md`.
4. Search the canonical documents before proposing a mechanism. Do not repeat
   a rejected direction unless new measured evidence changes its failed
   capacity, bandwidth, fidelity, or acceptance inequality.

## 1. Interaction and execution loop

- Communicate with the user in Romanian. Code, comments, docs, and commits are
  English.
- Never provide effort/time estimates. Use one dependency-ordered flat plan.
- Tie every edit, instrument, test, and benchmark to the active goal, an
  acceptance gate, or a measured blocker. No infrastructure or benchmarks for
  their own sake.
- Before editing, state the active invariants: model, format, context, fidelity,
  execution mode, hardware, and universal-vs-specific scope.
- Never guess when code, artifacts, telemetry, or docs can answer.
- Default to `xhigh`; token ceilings are runaway guards, not depth targets.
  Keep a compact checkpoint of verified facts, gates, blocker and next action.
  After failure, resume there and redo only invalidated dependencies. Restart
  from zero ONLY when an upstream invariant was disproved; stop reasoning paths
  that repeat without new evidence.

Mandatory change cycle:

```text
PLAN -> INSPECT -> IMPLEMENT COMPLETE CHANGE -> TEST REAL GATE
  failure #1 -> at most one local corrective patch -> RETEST
  failure #2 -> STOP EDITING -> reassess end-to-end -> NEW PLAN
```

No chains of cosmetic patches. A shortcut that changes an acceptance criterion
requires explicit user approval.

## 2. Active serving contracts

### Host

- The serving target is self-contained on one RTX 3090 host (`3090box`).
- No external owner, remote expert executor, distributed fallback, or outside
  coordinator may be introduced as a dependency or future escape hatch.
- Current validated routed-VRAM budget is 12 GiB. Do not raise it without real
  in-scope model chat gates and cleanup.

### Qwen3.8-27B

Acceptance target:

```text
GPU                 one RTX 3090, 24 GiB
active context      262,144 actually populated tokens
target KV           exact F16 KV semantics (IEEE binary16)
reasoning default   xhigh
harness             real coding harness
throughput          approximately 15 generated tokens/s, measured as useful output
```

Cold prefill, reused-prefix prefill, decode, and tool-loop wall time are
independent gates. Passing one never implies another.

### DeepSeek scope exclusion

- DeepSeek is out of scope. Agents MUST NOT research, modify, build, deploy,
  start, stop, restart, benchmark, smoke-test, or otherwise operate any
  DeepSeek model, artifact, provider, paging path, or task.
- DeepSeek MUST NOT be used as a universal-path, regression, release, cleanup,
  or acceptance gate. Existing DeepSeek code, artifacts, documentation, and
  historical evidence are read-only and may be consulted ONLY when required to
  avoid collateral damage to in-scope work.
- ONLY a later explicit user request that names DeepSeek and places it back in
  scope may revoke this exclusion.

## 3. Feasibility gate before inference work

Before proposing or implementing an inference optimization, write equations
for the exact model/context/formats/host/harness. Include:

```text
weights + recurrent state + exact KV + draft/MTP state + workspaces/reserve
bytes touched per target call and per accepted token
GPU, RAM, PCIe, and NVMe bandwidth bounds
speculative acceptance and proposer/verifier cost
```

Required distinctions:

- maximum configured context != populated context != harness prompt overhead;
- dense weights are hot per token;
- all retained K/V in full-softmax attention is hot per decode token;
- unrouted MoE experts are cold and eligible for paging;
- moving hot dense weights or active KV through RAM/NVMe recovers capacity but
  is not a throughput solution.

Research MUST have a fail-fast quantitative prerequisite before runtime code:
required compression ratio, exact-pruning fraction, accepted tokens/cycle,
transfer bytes/accepted token, or real harness throughput. Stop when it fails.

## 4. Fidelity and truthful claims

- Fidelity changes require explicit approval and a quality gate: KV
  quantization, attention sparsification, eviction, reduced top-k, low-rank
  approximation, summarization, or another model are not lossless.
- Report routed experts, dense matrices, embeddings, routers, KV, and runtime
  intermediates separately. Never call INT8 `INT4` or `FP4`.
- FP4 completion requires: FP4 payload exists; the selected provider executes
  that FP4 ABI; reference numerical quality passes; real service path passes.
- A numerical oracle MUST be independent of the runtime path it validates.
  Do not reuse the same decoder, scale direction, layout assumption, or helper
  on both sides and call agreement a fidelity result.
- Demand-paging completion requires selected experts loaded on demand and
  telemetry proving actual storage/RAM/VRAM traffic. Full startup residency is
  not a paging result.
- Never promote a microbenchmark, parser/metadata smoke, configured context
  limit, or client-side arithmetic as end-to-end service throughput.
- Real harness gates cannot be replaced by tiny direct-API prompts. Report
  populated prompt, cold/reused prefill, TTFT, decode, reasoning/visible tokens,
  tool-loop wall time, residency, and cleanup separately.

## 5. Universal artifact VM

- Common runtime/service code MUST NOT branch on Qwen, DeepSeek, Ornith, LFM,
  architecture ID, fixed layer count, or family tensor paths.
- A new model may add a strict source adapter. It may add a provider/capability
  only for genuinely new mathematics or encoding. It MUST NOT add a family
  runner, scheduled task, service profile, launch contract, or `model.sh` case
  unless the user explicitly approves the exception.
- Runtime/deployment selection is artifact-driven: capabilities, tensor roles,
  geometry, quantization ABI, and placement policy.
- Session retention is a provider capability, never a universal assumption.
- Streaming uses one artifact-declared parser instance from first delta through
  final structured tool extraction. Pi receives standard API events; do not add
  a second final reparse/reconcile path or family-specific output parser.
- A universal-path claim requires the same real `hi` smoke through
  `./ops/model.sh chat` for Qwen and every newly claimed in-scope model.

## 6. Artifact and model operations

- `MODEL_ROOT` is the ONLY host-dependent model-storage root. No hardcoded
  `C:/...`, `D:/...`, profile directory, or per-family absolute model root.
- `out/` is for builds/experiments. Service artifacts belong under
  `${MODEL_ROOT}/<stable-name>`.
- Publish transactionally: candidate -> validate manifest/completion/format ->
  promote -> update config -> start -> real smoke -> stop -> verify process/GPU
  cleanup. Never point config/tasks/defaults at partial or absent artifacts.
- Hugging Face downloads use Xet ONLY through:
  - `ops/windows/Start-HuggingFaceModelDownload.ps1`
  - `ops/windows/Get-HuggingFaceModelDownload.ps1`
- Never use curl/ad-hoc parallel downloads, invented scheduled tasks, or
  `HF_HUB_DISABLE_XET`.

## 7. Rejected-work guard

`docs/research-decisions.md` is the normative rejection ledger. In particular,
do not reintroduce without a new prerequisite that changes the failed math:

- Q4/FP8 target KV for an exact-F16 goal;
- ordinary dense layer streaming/offload as a decode-throughput solution;
- rejected static/reversible KV codecs;
- rejected MTP/tree, block-Jacobi, prompt-copy, or exact-refinement proposers;
- rejected single-GPU staged-attention, CPU V-tail, or all-K-resident layouts;
- DeepSeek `layer_major_prefill`, `causal_layer_major`, or row-width tuning as
  the maximum-context solution.

Record every new negative result and corrected claim in
`docs/research-decisions.md` or `docs/benchmarks.md`, not in this hot contract.

## 8. Verification commands and release boundary

Canonical Python contract:

```bash
python3 -m unittest -v \
  tests.compiler.test_expert_pack \
  tests.server.test_expert_server
```

Use the Windows Release/CUDA build and CTest path when native runtime code
changes. For universal service changes, run real `model.sh chat` gates in this
order: new/affected model, then Qwen. DeepSeek is excluded by section 2. Test Pi
only when harness behavior is part of the goal. Do not claim unrun gates.

- `cmake --fresh` resets CMake configuration; it does NOT remove stale object
  files or libraries. The canonical Windows Release/CUDA build MUST use
  `--clean-first` before linking deployable binaries.
- After any C++/CUDA header, launch-structure, provider ABI, quantization ABI,
  or worker-protocol change: stop the service, sync source, run the canonical
  clean build, then run gates on that exact produced binary. Never reuse a
  pre-build chat result as release evidence.
- If a native failure appears after an ABI/header change, reproduce it once on
  the canonical clean build before editing inference code. A failure that
  disappears after the clean rebuild is a build-integrity defect; fix the
  build pipeline, not model logic.
- Changing an aggregate launch/protocol structure requires auditing every
  producer and consumer, every aggregate initializer, and a native test that
  executes each affected provider path. Wire or artifact layout changes also
  require the corresponding protocol/ABI version and fail-closed validation.
- New-model qualification order is source/reference fidelity -> native
  numerical gate -> direct `model.sh chat` -> streaming/tool transport -> real
  harness. Do not skip a failed stage or call parser transport a coding pass.

## 9. Canonical pointers

- `docs/architecture.md`: implemented VM and memory tiers.
- `docs/benchmarks.md`: current measurement ledger; other historical claims are
  not evidence.
- `docs/research-decisions.md`: rejected mechanisms and quantitative gates.
- `docs/roadmap.md`: active dependency order.
- `docs/production-readiness.md`: release blockers.
- `../memory-expert`: extracted KV-attach experiment and its own runbooks.
