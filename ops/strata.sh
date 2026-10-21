#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
if [[ -f "${repo_root}/.env" ]]; then
  set -a
  source "${repo_root}/.env"
  set +a
fi

action="${1:-status}"
model="${2:-swift}"
case "${action}" in
  help|-h|--help)
    echo 'Usage: ./ops/strata.sh <install|publish|deploy|start|restart|stop|status|config> [swift]'
    echo 'Independent Strata service on the RTX 3090. deploy never restarts it.'
    exit ;;
  install|publish|deploy|start|restart|stop|status|config) ;;
  *) echo "Unknown Strata action: ${action}" >&2; exit 2 ;;
esac
[[ "${model}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo 'Invalid model selector' >&2; exit 2; }
: "${QUANTUM_LLM_REMOTE:?Set QUANTUM_LLM_REMOTE in .env}"
: "${QUANTUM_LLM_REMOTE_ROOT:?Set QUANTUM_LLM_REMOTE_ROOT in .env}"
: "${MODEL_ROOT:?Set MODEL_ROOT in .env}"
port="${STRATA_PORT:-8082}"
[[ "${port}" =~ ^[1-9][0-9]*$ ]] && (( port <= 65535 )) || { echo 'Invalid STRATA_PORT' >&2; exit 2; }

scp -q "${script_dir}/strata-models.json" "${QUANTUM_LLM_REMOTE}:${QUANTUM_LLM_REMOTE_ROOT}/ops/"
scp -q "${script_dir}/windows/Invoke-StrataServer.ps1" "${QUANTUM_LLM_REMOTE}:${QUANTUM_LLM_REMOTE_ROOT}/ops/windows/"
scp -q "${script_dir}/windows/Start-StrataServer.ps1" "${QUANTUM_LLM_REMOTE}:${QUANTUM_LLM_REMOTE_ROOT}/ops/windows/"
scp -q "${script_dir}/python/prepare_strata.py" "${QUANTUM_LLM_REMOTE}:${QUANTUM_LLM_REMOTE_ROOT}/ops/python/"
args=(-Action "${action}" -Model "${model}" -Port "${port}")
if [[ -n "${EXPERT_API_KEY:-}" ]]; then
  key_b64="$(python3 -c 'import base64,os; print(base64.b64encode(os.environ["EXPERT_API_KEY"].encode()).decode())')"
  args+=(-ApiKeyBase64 "${key_b64}")
fi
exec "${script_dir}/run-on-windows-host.sh" Invoke-StrataServer.ps1 "${args[@]}"
