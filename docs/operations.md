# Operations

## Deployment model

The tested pre-production profile is one Windows GPU host, one Python HTTP
front-end, and one persistent C++ CUDA worker. Task Scheduler supplies restart
and logon activation; it is a pilot supervisor, not a full Windows service
manager.

## Control host

The optional POSIX wrappers sync Git-visible source without copying `.git`,
ignored models, work directories, logs, artifacts, or build output:

```bash
export QUANTUM_LLM_REMOTE=user@gpu-host
export QUANTUM_LLM_REMOTE_ROOT=C:/quantum-llm
./ops/sync-to-3090box.sh
./ops/run-on-3090box.sh Invoke-BuildExpertRuntime.ps1 -Configuration Release
```

Sync is non-destructive: obsolete remote source files are not removed. Use a
fresh deployment directory for a release or clean obsolete tracked files during
the release procedure.

## Lifecycle

```powershell
# Install or replace configuration and optionally start
.\ops\windows\Install-ExpertServerTask.ps1 -Start -BuildId <commit>

# Stop and release VRAM; keep the task registered
.\ops\windows\Stop-ExpertServer.ps1

# Start an installed task
Start-ScheduledTask QuantumLLM-P6ExpertServer

# Stop, kill descendants, and unregister
.\ops\windows\Uninstall-ExpertServerTask.ps1
```

Stopping only the PowerShell task is insufficient on some Windows versions:
the CUDA worker can survive as a grandchild. The repository stop/uninstall
scripts resolve the server command line and kill its full descendant tree.

## Health and identity

- `/health`: worker process is alive;
- `/ready`: worker is healthy and server is accepting requests;
- `/model-info`: model ID, build ID, manifest/index hashes, active requests,
  cache budgets, context, capacity, and timeouts;
- `/metrics`: Prometheus counters and bounded latency-window percentiles.

Always check build ID and model content hashes after deployment. A listening
port alone is not proof that the new worker started.

## Logs and artifacts

Generated state is intentionally outside Git:

```text
logs/expert-server.jsonl
artifacts/expert-runtime-build-latest.json
artifacts/p6-service-smoke-latest.json
artifacts/p6-gate-latest.json
```

Ship logs and metrics to external storage for a real pilot. Local JSONL has no
rotation or retention manager. Never log API keys or prompt content.

## Capacity and overload

The default profile has four active worker slots and eight queued requests.
Admission and worker-slot acquisition are bounded. Overload returns HTTP 503;
clients should use bounded exponential backoff with jitter and a request
deadline.

An 18 GiB VRAM expert-cache budget does not allocate 18 GiB at startup. Cache
use grows with expert reuse and remains hot until eviction or process stop.
Stopping the service releases model VRAM.

## Rollback

1. Stop the task with the repository script.
2. Deploy a previously validated commit into a fresh directory.
3. Build and run compiler/runtime tests.
4. Validate the existing immutable Expert Pack.
5. install with the rollback commit as `BuildId`.
6. Require `/model-info` identity and the full service smoke to pass.

Code rollback does not mutate the original Hugging Face checkpoint or the
Expert Pack. ABI incompatibility must fail startup; never bypass validation.

## Backup and recovery

Expert Pack is reproducible from the pinned source checkpoint, adapter, and
compiler commit. Keep at least one of:

- the original checkpoint plus revision and hashes;
- an independently validated Expert Pack plus manifest and pack hashes.

A `.partial` conversion is not deployable. Resume only with the same source
bytes and options. Do not treat a model directory without `COMPLETED` as valid.

## Remote exposure

Loopback is the safe default. For network access:

1. set `EXPERT_API_KEY` in the service environment;
2. bind the service only on a private interface;
3. terminate TLS and enforce rate/body limits at a reverse proxy;
4. restrict health/metrics/model-info separately;
5. use firewall allowlists and rotate the shared key.

The built-in server is not an internet-facing edge server.
