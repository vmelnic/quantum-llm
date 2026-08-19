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

chat_max_tokens="${CHAT_MAX_TOKENS:-${MODEL_MAX_OUTPUT_TOKENS:-8192}}"
generation_timeout="${MODEL_GENERATION_TIMEOUT_SECONDS:-600}"
if [[ ! "${generation_timeout}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODEL_GENERATION_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
chat_request_timeout="${CHAT_REQUEST_TIMEOUT_SECONDS:-$((generation_timeout + 60))}"
if [[ ! "${chat_request_timeout}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CHAT_REQUEST_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ -n "${MODEL_MAX_OUTPUT_TOKENS:-}" &&
      "${chat_max_tokens}" =~ ^[0-9]+$ &&
      "${MODEL_MAX_OUTPUT_TOKENS}" =~ ^[0-9]+$ ]] &&
   (( chat_max_tokens > MODEL_MAX_OUTPUT_TOKENS )); then
  echo "CHAT_MAX_TOKENS cannot exceed MODEL_MAX_OUTPUT_TOKENS" >&2
  exit 2
fi

chat_args=(
  --base-url "${CHAT_BASE_URL:-http://127.0.0.1:8080}"
  --model "${CHAT_MODEL:-deepseek-v4-flash}"
  --max-tokens "${chat_max_tokens}"
  --local-port "${CHAT_LOCAL_PORT:-18080}"
  --remote-port "${MODEL_PORT:-${CHAT_REMOTE_PORT:-8080}}"
  --ready-timeout "${CHAT_READY_TIMEOUT:-30}"
  --request-timeout "${chat_request_timeout}"
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
