#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
if [[ -f "${repo_root}/.env" ]]; then
  set -a
  source "${repo_root}/.env"
  set +a
fi
backend="${1:-}"
[[ "${backend}" == quantum || "${backend}" == llama ]] || {
  echo 'Usage: fetch.sh quantum|llama' >&2
  exit 2
}
[[ -n "${QUANTUM_LLM_REMOTE:-}" && -n "${QUANTUM_LLM_REMOTE_ROOT:-}" ]] || {
  echo 'Remote host and root must be configured' >&2
  exit 2
}
local_parent="${repo_root}/out/benchmarks/pi256k"
[[ ! -e "${local_parent}/${backend}" ]] || {
  echo 'Local result already exists; refusing to overwrite it' >&2
  exit 2
}
remote_rel="out/benchmarks/pi256k/${backend}"
if ! ssh -o BatchMode=yes "${QUANTUM_LLM_REMOTE}" \
    "if not exist \"${QUANTUM_LLM_REMOTE_ROOT}/${remote_rel}/COMPLETE.json\" exit /b 3"; then
  echo "Remote ${backend} benchmark has no completion marker yet" >&2
  exit 3
fi
mkdir -p "${local_parent}"
ssh -o BatchMode=yes "${QUANTUM_LLM_REMOTE}" \
  "tar -cf - -C \"${QUANTUM_LLM_REMOTE_ROOT}\" \"${remote_rel}\"" |
  tar -xf - -C "${repo_root}"
[[ -f "${local_parent}/${backend}/COMPLETE.json" ]] || {
  echo 'Copied result has no completion marker' >&2
  exit 1
}
echo "Copied ${local_parent}/${backend}"
