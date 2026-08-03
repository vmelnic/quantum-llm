#!/usr/bin/env bash
set -euo pipefail

remote_host="${1:-vladi@10.10.88.4}"
remote_root="${2:-C:/Users/vladi/quantum-llm}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

ssh -o BatchMode=yes "${remote_host}" \
  "if not exist \"${remote_root}\" mkdir \"${remote_root}\""

COPYFILE_DISABLE=1 tar \
  --exclude='./.git' \
  --exclude='./artifacts' \
  --exclude='./logs' \
  --exclude='./work' \
  --exclude='._*' \
  --exclude='.DS_Store' \
  -C "${repo_root}" -cf - . \
  | ssh -o BatchMode=yes "${remote_host}" "tar -xf - -C \"${remote_root}\""

ssh -o BatchMode=yes "${remote_host}" \
  "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${remote_root}/ops/windows/Invoke-Bootstrap.ps1\""

echo "Synchronized ${repo_root} to ${remote_host}:${remote_root}"
