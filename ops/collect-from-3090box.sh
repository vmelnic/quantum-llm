#!/usr/bin/env bash
set -euo pipefail

remote_host="${1:-vladi@10.10.88.4}"
remote_root="${2:-C:/Users/vladi/quantum-llm}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

mkdir -p "${repo_root}/artifacts"
ssh -o BatchMode=yes "${remote_host}" \
  "tar -cf - -C \"${remote_root}\" artifacts" \
  | tar -xf - -C "${repo_root}"

echo "Collected artifacts into ${repo_root}/artifacts"
