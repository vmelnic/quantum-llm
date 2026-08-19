# MoE VM current state and remaining work

Status: canonical implementation handoff, 2026-08-17.

This is the document to read before resuming work. It separates what is proven
from what the name “MoE VM” still promises but does not yet implement. Other
plans in `docs/` are evidence or history unless this document imports an item
explicitly.

## Goal

Build one artifact-driven sparse-model runtime in which:

- the artifact declares topology, ordered operations, tensor roles, routing,
  expert encoding and numeric ABIs;
- the common launcher and runtime do not branch on Qwen, DeepSeek, LFM, layer
  count or model geometry;
- an expert is an immutable logical page that may be satisfied from VRAM, RAM,
  SSD-to-GPU, CPU execution or a future remote owner;
- providers implement operation/encoding ABIs rather than model families;
- exact routing and the checkpoint's top-k are preserved. A missing selected
  expert fails closed; it is never dropped or replaced with zero.

The goal is not to fit the whole model in RAM or VRAM. The model may be much
larger than either tier; only the dense resident state and the bounded active
expert working set must fit their declared budgets.

## Verified state

The following is implemented and was exercised on the Windows/RTX 3090 host:

- one control command: `ops/model.sh`;
- one scheduled task: `QuantumLLM-ExpertVm`;
- one serving binary: `expert-moe-vm-runner.exe`;
- one generic model root: `MODEL_ROOT=D:/quantum-llm/work/models`;
- project, build tree, published packs and mutable route state on the Windows
  host's D: NVMe; Hugging Face source snapshots remain in the independent C:
  user cache;
- data-driven aliases in `ops/model-aliases.tsv`; `qwen` resolves only to the
  FP4 artifact;
- immutable `runtime-model.tsv` schema 2 programs bound by manifest byte count
  and SHA-256;
- artifact-declared model dimensions, routed components, router programs,
  required capabilities, ordered operations, tensor bindings and operation
  parameters;
- provider negotiation by capability/ABI, with provider-owned geometry checks;
- logical expert identity and exact-route resolution through the common
  SSD/RAM/VRAM cache/store boundary;
- demand/prefetch/warm priority, host-only preload, protected/probationary RAM,
  transient/protected VRAM, census warm-up, prefill-route protection, parallel
  MTP acquisition and full priority-attributed cache telemetry;
- Qwen3-Next 80B FP4, DeepSeek-V4-Flash and LFM2-8B-A1B all produced a real
  answer to the same `hi` request through `./ops/model.sh chat` and the common
  service path;
- the final Windows/CUDA build passed all four CTest targets and 59 Python
  tests.

The final functional probes were:

| Artifact | Answer prefix | TTFT | Total wall | Post-first-token rate |
|---|---|---:|---:|---:|
| Qwen3-Next 80B FP4 | `Hello! How can I help you today?` | 11.755 s | 21.449 s | 1.13 tok/s |
| DeepSeek-V4-Flash | `Hello! How can I help you today?` | 8.901 s | 15.728 s | 1.32 tok/s |
| LFM2-8B-A1B FP4 | `Hello! How can I assist you today?` | 7.508 s | 12.141 s | 2.37 tok/s |

These are lifecycle/correctness probes, not a comparable performance suite.
They prove that the same public path serves all three artifacts. They do not
prove a speed improvement from the VM refactor. The service and GPU worker were
stopped after the probes.

The newest controlled DeepSeek performance result predates the final VM smoke
but uses the same exact paging/cache work: priority-ordered task indexes raised
a settled identical-prompt probe to 6.54–6.69 tok/s with zero SSD reads and
about 47.8 GB H2D/request. The matched novel suite still took 321.47 seconds
for eight prompts, and novel/retained decode in the preceding matched probes
remained below one tok/s. This is the current optimization baseline; the final
cold `hi` result must not be substituted for it or vice versa.

## What is universal now

| Boundary | State | Exact meaning |
|---|---|---|
| lifecycle and API | implemented | One script, task, server contract and worker protocol |
| artifact discovery | implemented | A direct child of `MODEL_ROOT` can be selected without a common-path model branch |
| model program | implemented | Schema 2 expresses layers, operations, components, tensor roles and numeric parameters |
| expert virtual address | implemented | Namespace/component-layer/expert identity is independent of a global hard-coded layer table |
| local tiering | implemented | Exact expert routes resolve through bounded SSD, RAM and VRAM state |
| capability negotiation | implemented as infrastructure | `ExecutionProviderRegistry` can bind each required operation capability independently |
| actual serving executor composition | not implemented | `expert-moe-vm-runner` still selects one complete `WorkerProviderDefinition` and enters that provider's existing model loop |
| physical container | not unified | Qwen/LFM use Expert Pack v1; DeepSeek uses worker bundle v3 plus compact pack v1 |
| numeric geometry | partial | Common descriptors are dynamic, but current CUDA providers retain validated geometry/algorithm limits |
| generic HF ingestion | partial | The compiler is strict and artifact-driven after adaptation, but new upstream tensor naming/semantics still require an adapter |
| FP4-only serving policy | partial | aliases/artifacts are FP4-only, but generic legacy quant ABI 1 reader/compiler code still exists |
| CPU/remote placement | partial/missing | Local CPU paths exist for supported records; remote leases are an interface only, with no transport or remote worker |
| representative 30 tok/s | not achieved | Historical resident-route results do not establish arbitrary-chat throughput |

The most important distinction is the two provider registries. The
per-operation `ExecutionProviderRegistry` and compiled numeric program exist,
but the serving entry point currently uses `WorkerProviderRegistry`, which
requires one provider to cover the whole artifact. Therefore the control plane
is generic while the full numeric executor is not yet a composable VM.

## Current artifact layout

Under `MODEL_ROOT` the supported artifacts are:

```text
D:/quantum-llm/work/models/
  qwen3-next-80b-expert-pack-fp4/
  deepseek-v4-flash/
    worker-bundle-v3/
  lfm2-8b-a1b-expert-pack-fp4/
```

Qwen and LFM use Expert Pack v1. DeepSeek retains its model-specific physical
bundle and compact routed shards, loaded through a storage adapter. Adding
`runtime-model.tsv` did not rewrite, merge or requantize existing pack payloads.
The Qwen adapter declares FP4 as its only accepted routed-expert profile; the
common compiler enforces adapter-declared profiles without a family branch.
Generic INT8 quant ABI 1 compatibility remains available to adapters that
explicitly declare it.

## Remaining implementation, dependency order

1. **Freeze the current cross-model acceptance contract.** Keep one fixture
   test for schema-2 descriptor parsing, capability binding, storage-adapter
   selection and exact-route validation. Keep the real three-artifact `hi`
   gate as the integration baseline. Record artifact hashes and do not use the
   functional probe as a throughput claim. Keep Qwen's adapter-declared FP4
   restriction fail-closed while allowing other adapters to publish only the
   profiles they declare.

2. **Define one next-generation physical artifact contract.** Specify a
   container that can hold multiple routed components, dense/shared tensors,
   tokenizer assets, source/compute encodings, catalogs and the model program
   without naming a model family. It must support one or many pack files and
   preserve independent record hashes and transactional publication. Migrate
   DeepSeek metadata/catalogs into it without changing or copying the 147 GB
   FP4 payload merely to rename the format. Retain old-format readers as input
   adapters during migration, not as permanent serving branches. Remove the
   current infrastructure contradiction in which the asynchronous DeepSeek
   pack wrapper rejects destinations under `work/` while `MODEL_ROOT` is
   deliberately `D:/quantum-llm/work/models`.

3. **Make the serving loop execute the compiled operation program.** Replace
   whole-worker provider selection with a common request/session interpreter
   that walks `CompiledOperationProgram`. Each instruction calls its bound
   operation provider; model-specific loops cease to own control flow. Startup
   binds numeric provider handles once, so strings/maps remain outside the
   token hot path.

4. **Extract state and tensor ownership from complete workers.** Define
   provider APIs for immutable tensor binding, request-state sizing/allocation,
   operation execution, route publication, token output and state teardown.
   Move the existing Qwen, DeepSeek and LFM attention/router/expert/head code
   behind those APIs. A provider may own genuinely different math, but it may
   not select behavior from a model-family name.

5. **Finish geometry-driven capability validation.** Every remaining numeric
   constant must be either an artifact parameter or an explicit provider
   capability constraint. Parameterize kernels where the existing algorithm
   already supports the geometry; reject unsupported geometry at startup where
   a new kernel is truly required. Do not turn family branches into disguised
   `if layer_count == ...` branches.

6. **Stabilize the compiler adapter boundary.** A new checkpoint adapter may
   map source tensor names and upstream semantic metadata into the universal
   artifact. After compilation, common validation, lifecycle, placement and
   execution must not change. Prove the boundary with a fourth public FP4 MoE
   model whose operation/encoding set is already supported; no common runtime,
   server, launcher or alias-code edit is allowed for that onboarding.

7. **Unify local execution targets behind the expert-page contract.** Make
   device-resident, host-to-device, local CPU and storage acquisition explicit
   provider capabilities with identical exact-route completion semantics.
   Keep resource credits and eviction independent of total model size. Add a
   remote owner only after local semantics and cancellation are fully tested;
   move activations to remote expert compute, never whole expert weights per
   token over an ordinary network.

8. **Resume performance work only on the generic path.** Preserve quality and
   exact top-k. Measure request-attributed SSD rereads, RAM/VRAM hits, H2D
   bytes, pinned leases, queue priority, MTP acquisition union and useful
   accepted tokens. The priority tiers, census RAM, transient VRAM, parallel
   MTP acquisition and prompt-route placement are already implemented; do not
   repeat them as new work. Execute only the measured backlog below. Any
   optimization must improve the same real changing-prompt suite and must not
   add a family-specific fast path.

9. **Complete release qualification.** Re-run portable tests, Windows/CUDA
   tests, independent pack validation, all three real chats, corrupt/unsupported
   artifact rejection, cancellation and full process-tree stop. Add the fourth
   model gate from item 6. Only then may the whole runtime be described as a
   generic MoE VM rather than a generic control plane over several numeric
   workers.

## Performance backlog after executor unification

The following measured follow-ons survive the architecture refactor. They are
not prerequisites for items 1–7 above and must be implemented through generic
cache/scheduler/provider contracts:

1. Replace generic recency with a bounded value policy that distinguishes
   prefill and verify multiplicity from likely future decode value. Evaluate it
   in the exact-route LRU/Belady oracle first, then require lower bytes and wait
   in the matched service suite. Wider prefill protection and raw-frequency
   promotion already regressed and must not be repeated unchanged.
2. Add byte-budgeted transition prefetch from SSD to protected RAM for absent
   experts. Exact demand cancels/upgrades it. Automatically disable it when
   speculative read bytes rise without a storage-wait reduction; the current
   predictor produced roughly 1,702 incorrect of 3,440 predictions.
3. Publish the next-layer prediction before current-layer FFN completion and
   overlap only already-RAM-resident uploads. The gate is lower
   `scheduler_expert_wait_ns` with no increase in uploaded bytes.
4. Proactively evict one unmatched retained worker slot before an unrelated
   `BEGIN`, removing the known fail/retry round trip without changing tokens.
5. Replace the remaining eviction map scan only with a deterministic priority
   structure that beats the measured ~0.96 s/request residual. Two attempted
   residency indexes did not and were removed; keep the validated read/upload
   task indexes.
6. Revisit MTP placement only after expert movement falls. Current traces often
   accept zero or one of three drafts, and two request-local gating variants
   increased SSD reads and were reverted.

## Explicit non-goals

- no top-k reduction, expert omission or substitution;
- no lower-bit change justified only by speed; quantization changes require a
  separate quality decision and are not part of this plan;
- no model-family switches in `ops/model.sh`, the HTTP server, common runtime
  or common cache;
- no hard-coded layer count, expert count or geometry in universal code;
- no new artifact name that implies a new payload layout when only metadata
  changed;
- no remote tier before local exact-route, failure and cancellation contracts
  are complete;
- no claim that the VM refactor itself improved tokens per second.

## Resume checklist

Begin by reading this document, then inspect the dirty worktree before editing:

```bash
git status --short
git diff --stat
./ops/model.sh config
```

Build and validate on the Windows/CUDA host through the repository scripts:

```bash
set -a
source .env
set +a
./ops/sync-to-windows-host.sh "$QUANTUM_LLM_REMOTE" D:/quantum-llm
./ops/run-on-windows-host.sh Invoke-BuildExpertRuntime.ps1 -Configuration Release
```

Run the common-path functional gate with the same input for every artifact:

```bash
./ops/model.sh start qwen
./ops/model.sh chat qwen

./ops/model.sh start deepseek
./ops/model.sh chat deepseek

./ops/model.sh start lfm2-8b-a1b-expert-pack-fp4
./ops/model.sh chat lfm2-8b-a1b-expert-pack-fp4

./ops/model.sh stop all
```

Enter `hi`, record the complete answer and `/stats`, then exit each client.
Check `/model-info` identity rather than trusting a listening port. At the end,
confirm that the scheduled task is not running and no Python/native worker
retains the GPU.

When an implementation fails, diagnose the complete boundary before changing
it. Apply one evidence-based corrective patch, rerun the same gate, and if it
still fails, stop patching and reassess the architecture before the next
implementation loop.
