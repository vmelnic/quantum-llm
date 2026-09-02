#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
env_file="${repo_root}/.env"

if [[ ! -f "${env_file}" ]]; then
  echo "claude-api-key.sh: missing ${env_file}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

if [[ -z "${EXPERT_API_KEY:-}" ]]; then
  echo "claude-api-key.sh: EXPERT_API_KEY is not configured" >&2
  exit 2
fi

printf '%s\n' "${EXPERT_API_KEY}"
