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

- `docs/inference-30toks-plan.md` — the serving-speed work plan for the
  3090 host (Qwen3-Next-80B / DeepSeek-V4-Flash).
- The Memory Expert (KV-attach) experiment was extracted into the standalone
  public project at `../memory-expert` (validated architecture, results, and
  runbooks live there now).
