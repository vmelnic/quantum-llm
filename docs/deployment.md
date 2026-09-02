# Deployment

Status: supported single-host lifecycle, 2026-09-02.

## Contract

The Windows target contains the synchronized project, Release/CUDA build,
repository-owned server environment and all published artifacts. The serving
path has no external model-state owner.

Stable artifacts live under `${MODEL_ROOT}`. `Promote-ModelArtifact.ps1`
validates a candidate, moves an existing stable directory to a timestamped
rollback, promotes by rename and validates again. Rollbacks are temporary
release safety, not permanent alternate models; remove them after the promoted
artifact passes real gates and recovery is no longer needed.

## Active aliases

`ops/model-aliases.tsv` is authoritative:

| Alias | Advertised model | Path below `MODEL_ROOT` | KV selection |
|---|---|---|---|
| `qwen` | `qwen3.8-27b-fp4` | `qwen3.8-27b-fp4` | `fp16` |
| `qwen-flash` | `qwen3.8-flash-next-fp4` | `qwen3.8-flash-next-fp4` | `fp16` |
| `mistral` | `mistral-small-4-119b-nvfp4` | `mistral-small-4-119b-nvfp4` | artifact-declared |
| `muse` | `muse-glimmer-30b-fp4` | `muse-glimmer-30b-fp4` | `fp16` |
| `ornith` | `ornith-1.5-35b-a3b-fp4` | `ornith-1.5-35b-a3b-fp4` | `fp16` |
| `deepseek` | `deepseek-v4-flash` | `deepseek-v4-flash/worker-bundle-v3` | artifact-declared |

Aliases select an artifact and advertised ID only. The task
`QuantumLLM-ExpertVm`, HTTP service and native VM are common.

## Lifecycle

```bash
./ops/model.sh config qwen
./ops/model.sh install
./ops/model.sh start qwen
./ops/model.sh status
./ops/model.sh chat qwen
./ops/model.sh restart qwen
./ops/model.sh stop all
```

`start` synchronizes Git-visible files by default, validates the artifact
contract, clamps configured context/KV values to artifact requirements, checks
the pinned Python environment, stops the previous service, installs one task,
waits for readiness and verifies model/limit identity. Sync never transfers or
deletes ignored model, build, log or cache state.

## Network and authentication

The safe default is loopback on 3090box plus the tunnel used by the local
client. A non-loopback `MODEL_HOST` requires `EXPERT_API_KEY`. The native edge
does not provide TLS or public rate limiting; add them at a trusted proxy before
public exposure.

The reference host uses WireGuard only for `10.10.88.0/24`; public traffic,
including Hugging Face Xet, uses Ethernet. Routing all traffic through the
WireGuard tunnel can starve SSH/RDP during downloads. The maintained
`Set-WireGuardSplitTunnel.ps1` transaction validates the private route, public
route and direct HTTPS before committing.

## Resource policy

The current common baseline is:

```text
one RTX 3090
MODEL_RAM_CACHE_GIB=48
MODEL_VRAM_CACHE_GIB=12
MODEL_WORKER_CAPACITY=1
MODEL_KV_PAGE_TOKENS=256
MODEL_PLACEMENT_PROFILE=balanced
```

The 12 GiB VRAM value is a cache ceiling, not a reservation. DeepSeek fails
preflight at 13 GiB after fixed allocations, workspace and the 1 GiB reserve;
do not raise the common value without real Qwen and DeepSeek chat gates plus
cleanup. Model context is also a ceiling: pages grow only for populated tokens.

## Release sequence

1. validate source identity, candidate completion, hashes, format and numerical
   quality;
2. promote the candidate transactionally;
3. run the clean Windows Release/CUDA build, CTest and canonical Python suite;
4. run `model.sh chat` with `hi` for the affected model, then Qwen and DeepSeek;
5. for universal releases, repeat for every claimed active model;
6. run Pi only when harness behavior is part of the release;
7. stop the service and verify worker/process/GPU cleanup;
8. remove obsolete rollback/candidate directories after the stable artifact is
   accepted.

Parser, manifest and microkernel tests never replace the real service gate.
