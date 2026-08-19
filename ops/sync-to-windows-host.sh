#!/usr/bin/env bash
set -euo pipefail

remote_host="${1:-${QUANTUM_LLM_REMOTE:-}}"
remote_root="${2:-${QUANTUM_LLM_REMOTE_ROOT:-}}"
if [[ -z "${remote_host}" ]]; then
  echo "Pass user@host or set QUANTUM_LLM_REMOTE" >&2
  exit 2
fi
if [[ -z "${remote_root}" ]]; then
  echo "Pass the remote project root or set QUANTUM_LLM_REMOTE_ROOT" >&2
  exit 2
fi
remote_native="${remote_root//\//\\}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

ssh -o BatchMode=yes "${remote_host}" \
  "if not exist \"${remote_root}\" mkdir \"${remote_root}\""

(
  cd "${repo_root}"
  git ls-files -z --cached --others --exclude-standard |
    while IFS= read -r -d '' path; do
      [[ -e "${path}" ]] && printf '%s\0' "${path}"
    done
) | COPYFILE_DISABLE=1 tar \
  --format=ustar \
  --null \
  --no-xattrs \
  --no-acls \
  --no-fflags \
  -C "${repo_root}" -T - -cf - \
  | ssh -o BatchMode=yes "${remote_host}" "tar -xf - -C \"${remote_root}\""

ssh -o BatchMode=yes "${remote_host}" \
  "del /f /q \"\\\\?\\${remote_native}\\._.\" 2>nul || ver >nul"

ssh -o BatchMode=yes "${remote_host}" \
  "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${remote_root}/ops/windows/Invoke-Bootstrap.ps1\""

echo "Synchronized ${repo_root} to ${remote_host}:${remote_root}"
