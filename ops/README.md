# Operations automation

`ops/` contains deployment automation only. Product code lives in `core/`,
`compiler/`, and `runtime/`; public contracts live in `docs/` and `schemas/`.
The complete operator walkthrough is [Install, configure, and use](../docs/deployment.md).

## POSIX control-host wrappers

Copy `.env.example` to `.env` and set the remote host and immutable model
artifact paths. The lifecycle wrapper consumes that file:

```bash
./ops/model.sh config             # resolved, non-secret configuration
./ops/model.sh install            # sync + pinned server Python environment
./ops/model.sh start              # CHAT_MODEL from .env; sync + replace + wait
./ops/model.sh status
./ops/model.sh chat
./ops/model.sh stop

./ops/model.sh start qwen         # explicit one-command model switch
./ops/model.sh start deepseek
./ops/model.sh stop all           # release all model processes and VRAM
./ops/model.sh sync               # sync without changing the running service
```

Starting a model first stops both known scheduled tasks because they share one
GPU and one API port. It checks the installed Python/tokenizer dependencies
before disrupting the running model. `MODEL_MAX_CONTEXT` and
`MODEL_MAX_OUTPUT_TOKENS` become
the advertised API limits and are checked against `/model-info` before `start`
returns. `CHAT_MAX_TOKENS` is an optional independent per-turn ceiling; the
reference configuration gives chat the full model output allowance. Chat uses
the sole model returned by `/v1/models`, so an explicit model switch does not
require rewriting `.env` before connecting.

The sync includes only Git-visible, non-ignored files. It therefore excludes
`.git`, models, work, logs, artifacts, and build output. It does not delete
remote files.

Interactive streaming chat can own its SSH tunnel and clean it up on exit:

```bash
cp .env.example .env
# Set CHAT_SSH in .env, then:
./ops/model.sh chat
```

The Memory Expert experiment (KV-attach external memory) was extracted into
the standalone public project at `../memory-expert`; its `ops/` directory
keeps the original control-host/worker runbooks as reference.

## Windows scripts

The directory is intentionally limited to supported operator workflows:

- `Invoke-Bootstrap.ps1`, `Invoke-Inventory.ps1`,
  `Invoke-BuildExpertRuntime.ps1`, and `Install-ServerEnvironment.ps1` prepare
  and verify a host;
- `Start-*Download.ps1`, `Get-*Download.ps1`, and their workers implement
  pinned, resumable checkpoint acquisition and completion checks;
- `Invoke-ExpertPack.ps1`, `Invoke-P6Preflight.ps1`, and
  `Invoke-P6Conversion.ps1` reproduce and validate the Qwen Expert Pack;
- `Start-DeepSeekCompactPack.ps1`, `Get-DeepSeekCompactPack.ps1`,
  `Invoke-DeepSeekCompactPack*.ps1`, the MTP exporters, and
  `Publish-DeepSeekWorkerBundle.ps1` reproduce the DeepSeek deployment
  artifact;
- `Start-ExpertServer.ps1` plus the two model-specific launchers and
  `Install-ExpertServerTask.ps1` implement foreground and scheduled service
  startup;
- `Get-ExpertServerStatus.ps1`, `Stop-ExpertServer.ps1`,
  `Uninstall-ExpertServerTask.ps1`, and the model smoke/gate scripts implement
  bounded status, verification, shutdown, and performance checks.

Intermediate phase-admission scripts used while developing the DeepSeek
backend are not part of the supported operator surface. Their outcomes are
preserved in [Engineering history](../docs/history.md), while current
correctness belongs in the build/test suites.

Generated logs/artifacts/work directories are ignored by Git. Model deletion is
not part of normal automation. Source shard reclamation exists only behind an
explicit literal confirmation and is documented as destructive.

See [Getting started](../docs/getting-started.md) and
[Operations](../docs/operations.md) for complete procedures.
