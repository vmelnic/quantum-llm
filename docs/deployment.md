# Deployment

Status: supported single-host lifecycle, 2026-09-27.

## Artifact and task layout

The Windows host contains the synchronized repository, the Release/CUDA build,
the repository-owned Python environment and published artifacts below
`${MODEL_ROOT}`.

One scheduled task, `QuantumLLM-ExpertVm`, owns the active Python service and
native worker. Starting another model stops the previous task. The serving
path has no external model-state owner.

## Active aliases

`ops/model-aliases.tsv` is authoritative. Primary aliases are:

| Alias | Advertised model | Artifact below `MODEL_ROOT` | KV policy |
|---|---|---|---|
| `qwen` | `qwen3.8-27b-fp4` | `qwen3.8-27b-fp4` | lossy `q4-f16-per-head` |
| `qwen-f16` | `qwen3.8-27b-fp4` | `qwen3.8-27b-fp4` | exact `fp16` |
| `qwen-abliterated` | `qwen3.8-27b-abliterated-fp4` | `qwen3.8-27b-abliterated-fp4` | lossy `q4-f16-per-head` |
| `qwen-flash` | `qwen3.8-flash-next-fp4` | `qwen3.8-flash-next-fp4` | `fp16` |
| `mistral` | `mistral-small-4-119b-nvfp4` | `mistral-small-4-119b-nvfp4` | artifact-declared |
| `muse` | `muse-glimmer-30b-fp4` | `muse-glimmer-30b-fp4` | `fp16` |
| `ornith` | `ornith-1.5-35b-a3b-fp4` | `ornith-1.5-35b-a3b-fp4` | lossy `q4-f16-per-head` |

The registry also contains long-name synonyms and explicit Abliterated
research/reference aliases for MSE-v2 weights, K1 KV and F16 KV. These are not
additional published model families. Every current alias uses the fixed
routed-VRAM policy.

## Lifecycle

```bash
./ops/model.sh config qwen
./ops/model.sh install
./ops/model.sh start qwen
./ops/model.sh status
./ops/model.sh chat qwen
./ops/model.sh restart qwen
./ops/model.sh clear-cache qwen
./ops/model.sh stop all
```

`start` performs these steps:

1. synchronize Git-visible files when `MODEL_SYNC_ON_START=1`;
2. authenticate and inspect the selected artifact program;
3. clamp context and exact-KV resources to the artifact contract;
4. validate the Q4H/MTP contract when applicable;
5. check the Windows Python environment;
6. stop the existing task;
7. install and start the common task;
8. wait for readiness and verify model, limits and KV policy.

Synchronization does not copy or delete ignored model, build, log or cache
state.

## Resource policy

The maintained baseline is:

```text
MODEL_RAM_CACHE_GIB=48
MODEL_VRAM_CACHE_GIB=12
MODEL_WORKER_CAPACITY=1
MODEL_KV_CACHE_MIB=5136
MODEL_KV_PAGE_TOKENS=256
MODEL_PLACEMENT_PROFILE=balanced
```

The 12 GiB value is the routed-expert cache ceiling. It is separate from dense
weights, KV, workspaces and the 1 GiB emergency reserve. Do not increase it
without a new capacity calculation and real affected-model chat gates.

Optional P100 settings are accepted only as one complete configuration:

```dotenv
MODEL_ACTIVE_EXPERT_DEVICES=auto
MODEL_ACTIVE_EXPERT_DEVICE_CACHE_GIB=<per-device>
MODEL_ACTIVE_EXPERT_HOST_CACHE_GIB=<total-host-budget>
```

Compatible providers may use that tier for routed pages. Active deployment
leaves it disabled because measured paths were slower.

## Publication

`Promote-ModelArtifact.ps1` requires the candidate and destination to share a
parent directory. It validates the candidate program, renames an existing
stable artifact to a timestamped rollback, promotes the candidate, then checks
that program and manifest identities did not change.

Release sequence:

1. build Windows Release/CUDA with `--clean-first`;
2. run CTest and the canonical Python suites;
3. compile a sibling candidate;
4. validate format, hashes, source quality and provider ABI;
5. promote transactionally;
6. start the affected alias;
7. run its real chat gate, then Qwen for a universal change;
8. run Pi when harness behavior changed;
9. stop and verify process and GPU cleanup;
10. keep or remove the rollback only after the release result is known.

## Network

Loopback plus the SSH tunnel is the supported default. A non-loopback bind
requires `EXPERT_API_KEY`. The service does not terminate TLS and does not
implement public rate limiting; use a trusted edge for public exposure.
