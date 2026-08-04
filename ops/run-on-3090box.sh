#!/usr/bin/env bash
set -euo pipefail

script_name="${1:-Invoke-Inventory.ps1}"
shift || true

if [[ "${script_name}" == */* || "${script_name}" != *.ps1 ]]; then
  echo "Expected a PowerShell filename from ops/windows, for example Invoke-Inventory.ps1" >&2
  exit 2
fi

remote_host="${QUANTUM_LLM_REMOTE:-}"
remote_root="${QUANTUM_LLM_REMOTE_ROOT:-C:/quantum-llm}"
if [[ -z "${remote_host}" ]]; then
  echo "Set QUANTUM_LLM_REMOTE=user@host" >&2
  exit 2
fi

ssh -o BatchMode=yes "${remote_host}" \
  "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${remote_root}/ops/windows/${script_name}\" $*"
