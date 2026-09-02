#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
env_file="${repo_root}/.env"
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

script_name="${1:-Invoke-Inventory.ps1}"
shift || true

if [[ "${script_name}" == */* || "${script_name}" != *.ps1 ]]; then
  echo "Expected a PowerShell filename from ops/windows, for example Invoke-Inventory.ps1" >&2
  exit 2
fi

remote_host="${QUANTUM_LLM_REMOTE:-}"
remote_root="${QUANTUM_LLM_REMOTE_ROOT:-}"
model_root="${MODEL_ROOT:-}"
if [[ -z "${remote_host}" ]]; then
  echo "Set QUANTUM_LLM_REMOTE=user@host" >&2
  exit 2
fi
if [[ -z "${remote_root}" ]]; then
  echo "Set QUANTUM_LLM_REMOTE_ROOT to the remote project root" >&2
  exit 2
fi

remote_environment=""
if [[ -n "${model_root}" ]]; then
  if [[ ! "${model_root}" =~ ^[A-Za-z]:[/\\][A-Za-z0-9._/\\\ -]+$ ]]; then
    echo "MODEL_ROOT contains unsupported Windows path characters" >&2
    exit 2
  fi
  remote_environment="set \"MODEL_ROOT=${model_root}\" && "
fi

ssh -o BatchMode=yes "${remote_host}" \
  "${remote_environment}powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${remote_root}/ops/windows/${script_name}\" $*"
