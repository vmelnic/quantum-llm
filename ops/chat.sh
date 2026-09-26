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

model_max_context="${MODEL_MAX_CONTEXT:-65536}"
if [[ ! "${model_max_context}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODEL_MAX_CONTEXT must be a positive integer" >&2
  exit 2
fi
model_max_tokens="${MODEL_MAX_OUTPUT_TOKENS:-$((model_max_context - 1))}"
chat_max_tokens="${CHAT_MAX_TOKENS:-${model_max_tokens}}"
generation_timeout="${MODEL_GENERATION_TIMEOUT_SECONDS:-600}"
if [[ ! "${generation_timeout}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODEL_GENERATION_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
chat_request_timeout="${CHAT_REQUEST_TIMEOUT_SECONDS:-$((generation_timeout + 60))}"
chat_thinking="${CHAT_THINKING_LEVEL:-xhigh}"
if [[ ! "${chat_request_timeout}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CHAT_REQUEST_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${model_max_tokens}" =~ ^[1-9][0-9]*$ ||
      ! "${chat_max_tokens}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODEL_MAX_OUTPUT_TOKENS and CHAT_MAX_TOKENS must be positive integers" >&2
  exit 2
fi
if (( model_max_tokens >= model_max_context )); then
  echo "MODEL_MAX_OUTPUT_TOKENS must be smaller than MODEL_MAX_CONTEXT" >&2
  exit 2
fi
if (( chat_max_tokens > model_max_tokens )); then
  echo "CHAT_MAX_TOKENS cannot exceed MODEL_MAX_OUTPUT_TOKENS" >&2
  exit 2
fi

chat_args=(
  --base-url "${CHAT_BASE_URL:-http://127.0.0.1:8080}"
  --model "${CHAT_MODEL:-qwen3.8-27b-fp4}"
  --max-tokens "${chat_max_tokens}"
  --local-port "${CHAT_LOCAL_PORT:-18080}"
  --remote-port "${MODEL_PORT:-8080}"
  --ready-timeout "${CHAT_READY_TIMEOUT:-30}"
  --request-timeout "${chat_request_timeout}"
  --thinking "${chat_thinking}"
)
if [[ -n "${QUANTUM_LLM_REMOTE:-}" ]]; then
  chat_args+=(--ssh "${QUANTUM_LLM_REMOTE}")
fi
case "${CHAT_SHOW_STATS:-1}" in
  1|true|TRUE|yes|YES) chat_args+=(--show-stats) ;;
  0|false|FALSE|no|NO) chat_args+=(--no-show-stats) ;;
  *) echo "CHAT_SHOW_STATS must be 1/0, true/false, or yes/no" >&2; exit 2 ;;
esac

exec "${CHAT_PYTHON:-python3}" "${script_dir}/python/chat_client.py" \
  "${chat_args[@]}" "$@"
