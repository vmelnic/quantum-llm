#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
env_file="${QUANTUM_LLM_ENV_FILE:-${repo_root}/.env}"
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

chat_args=(
  --base-url "${CHAT_BASE_URL:-http://127.0.0.1:8080}"
  --model "${CHAT_MODEL:-deepseek-v4-flash}"
  --max-tokens "${CHAT_MAX_TOKENS:-32}"
  --local-port "${CHAT_LOCAL_PORT:-18080}"
  --remote-port "${CHAT_REMOTE_PORT:-8080}"
  --ready-timeout "${CHAT_READY_TIMEOUT:-30}"
)
if [[ -n "${CHAT_SSH:-}" ]]; then
  chat_args+=(--ssh "${CHAT_SSH}")
fi
case "${CHAT_SHOW_STATS:-1}" in
  1|true|TRUE|yes|YES) chat_args+=(--show-stats) ;;
  0|false|FALSE|no|NO) chat_args+=(--no-show-stats) ;;
  *) echo "CHAT_SHOW_STATS must be 1/0, true/false, or yes/no" >&2; exit 2 ;;
esac

exec "${CHAT_PYTHON:-python3}" "${script_dir}/python/chat_client.py" \
  "${chat_args[@]}" "$@"
