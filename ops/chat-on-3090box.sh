#!/usr/bin/env bash
set -euo pipefail

remote_host="${QUANTUM_LLM_REMOTE:-vladi@10.10.88.4}"
remote_root="${QUANTUM_LLM_REMOTE_ROOT:-C:/Users/vladi/quantum-llm}"

ssh -t "${remote_host}" \
  "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${remote_root}/ops/windows/Invoke-ColibriChat.ps1\" $*"
