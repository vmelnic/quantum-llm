# Deployment

Status: supported single-host lifecycle, 2026-08-23.

## Deployment contract

The runtime is deployed from a POSIX control host to one Windows/CUDA host.
The target contains the project, build, server environment and all published
model artifacts. There is no external expert owner or distributed fallback.

Stable artifacts live at `${MODEL_ROOT}/<stable-name>`. Conversion output is a
sibling candidate. `Promote-ModelArtifact.ps1` verifies the program identity,
renames an existing stable directory to a timestamped rollback directory,
renames the validated candidate into place and validates it again. `.env` and
the service must never point to a partial candidate.

The two supported aliases are declared in `ops/model-aliases.tsv`:

| Alias | Advertised model | Artifact relative to `MODEL_ROOT` |
|---|---|---|
| `qwen` | `qwen3.8-27b-fp4` | `qwen3.8-27b-fp4` |
| `deepseek` | `deepseek-v4-flash` | `deepseek-v4-flash/worker-bundle-v3` |

Aliases select artifacts only. The server, scheduled task and VM runner remain
common and capability-driven.

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

`start` synchronizes Git-visible files by default, checks the pinned Python
environment, stops the competing model, installs one scheduled task, waits for
readiness and verifies the advertised model, context and output limits. Sync
does not copy ignored `work/`, `artifacts/`, model files, logs or builds and
does not delete remote data.

`stop all` terminates the common scheduled task and processes bound to the
configured port. Always use it after a release gate and verify that the Python
server, native worker and GPU allocation are gone.

## Network and authentication

The safe default is loopback on the Windows host plus the SSH tunnel opened by
`chat.sh` or another local client. If `MODEL_HOST` is non-loopback,
`EXPERT_API_KEY` is mandatory. The server provides bearer authentication but
not TLS, rate limiting or a public-network security boundary; place those at a
trusted edge before exposing it.

## Resource policy

The public limits are configured once in `.env`:

- `MODEL_MAX_CONTEXT` and `MODEL_MAX_OUTPUT_TOKENS`;
- request body, image pixels and image patch tokens;
- RAM/VRAM cache budgets and KV page size/dtype;
- worker slots and waiting requests;
- generation timeout and placement profile.

Qwen exact target KV uses `MODEL_KV_CACHE_DTYPE=fp16`. A 262K maximum is an
admission ceiling, not an upfront allocation and not evidence of acceptable
latency. Several retained sessions share one hot provider slot; parking gives
continuity, not parallel execution.

## Release sequence

1. validate the candidate artifact and its source/quantization quality;
2. promote it transactionally below `MODEL_ROOT`;
3. build and run the supported test suites;
4. start Qwen, run `hi` through `model.sh chat`, then stop it;
5. start DeepSeek, run the same smoke, then stop it;
6. verify service descendants and GPU allocations are absent;
7. preserve the previous stable artifact until rollback is no longer needed.

Do not promote a microbenchmark, metadata parser or resident-mode result as a
demand-paging or production release result.
