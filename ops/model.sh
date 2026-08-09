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

usage() {
  cat <<'EOF'
Usage: ./ops/model.sh <install|sync|start|stop|restart|status|chat|config> [deepseek|qwen|all]

The model defaults to CHAT_MODEL from .env. `start` synchronizes Git-visible
files by default, stops the competing model, installs the selected scheduled
task, waits for readiness, and verifies the advertised model/context/output.
EOF
}

die() {
  echo "model.sh: $*" >&2
  exit 2
}

is_true() {
  case "$1" in
    1|true|TRUE|yes|YES) return 0 ;;
    0|false|FALSE|no|NO) return 1 ;;
    *) die "expected a boolean, got '$1'" ;;
  esac
}

require_uint() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer"
}

action="${1:-status}"
selection_explicit=0
if (( $# >= 2 )); then selection_explicit=1; fi
selection="${2:-${CHAT_MODEL:-deepseek-v4-flash}}"
case "${selection}" in
  deepseek|deepseek-v4-flash)
    model_alias="deepseek"
    model_id="deepseek-v4-flash"
    task_name="QuantumLLM-DeepSeekV4Flash"
    ;;
  qwen|qwen3-next|qwen3-next-80b-a3b-expert-pack-int8)
    model_alias="qwen"
    model_id="qwen3-next-80b-a3b-expert-pack-int8"
    task_name="QuantumLLM-P6ExpertServer"
    ;;
  all)
    model_alias="all"
    model_id=""
    task_name=""
    ;;
  *) die "unsupported model '${selection}'; use deepseek or qwen" ;;
esac

remote_host="${QUANTUM_LLM_REMOTE:-${CHAT_SSH:-}}"
remote_root="${QUANTUM_LLM_REMOTE_ROOT:-C:/quantum-llm}"
port="${MODEL_PORT:-${CHAT_REMOTE_PORT:-8080}}"
max_context="${MODEL_MAX_CONTEXT:-65536}"
max_output="${MODEL_MAX_OUTPUT_TOKENS:-8192}"
ready_timeout="${MODEL_READY_TIMEOUT:-600}"
generation_timeout="${MODEL_GENERATION_TIMEOUT_SECONDS:-600}"
sync_on_start="${MODEL_SYNC_ON_START:-1}"
deepseek_bundle="${MODEL_DEEPSEEK_BUNDLE:-C:/quantum-llm/work/models/deepseek-v4-flash/worker-bundle-v3}"
qwen_container="${MODEL_QWEN_CONTAINER:-C:/quantum-llm/work/models/qwen3-next-80b-expert-pack-int8}"

require_uint MODEL_PORT "${port}"
require_uint MODEL_MAX_CONTEXT "${max_context}"
require_uint MODEL_MAX_OUTPUT_TOKENS "${max_output}"
require_uint MODEL_READY_TIMEOUT "${ready_timeout}"
require_uint MODEL_GENERATION_TIMEOUT_SECONDS "${generation_timeout}"
(( max_output < max_context )) || die "MODEL_MAX_OUTPUT_TOKENS must be smaller than MODEL_MAX_CONTEXT"

export QUANTUM_LLM_REMOTE="${remote_host}"
export QUANTUM_LLM_REMOTE_ROOT="${remote_root}"

require_remote() {
  [[ -n "${remote_host}" ]] || die "set QUANTUM_LLM_REMOTE or CHAT_SSH in ${env_file}"
}

run_remote() {
  require_remote
  "${script_dir}/run-on-windows-host.sh" "$@"
}

sync_remote() {
  require_remote
  "${script_dir}/sync-to-windows-host.sh" "${remote_host}" "${remote_root}"
}

stop_task() {
  run_remote Stop-ExpertServer.ps1 -TaskName "$1" -Port "${port}"
}

stop_all() {
  stop_task QuantumLLM-DeepSeekV4Flash
  stop_task QuantumLLM-P6ExpertServer
}

print_config() {
  printf '%s\n' \
    "model=${model_id:-all}" \
    "remote=${remote_host:-<unset>}" \
    "remote_root=${remote_root}" \
    "port=${port}" \
    "max_context=${max_context}" \
    "max_output_tokens=${max_output}" \
    "generation_timeout_seconds=${generation_timeout}" \
    "sync_on_start=${sync_on_start}"
  case "${model_alias}" in
    deepseek) printf 'bundle=%s\n' "${deepseek_bundle}" ;;
    qwen) printf 'container=%s\n' "${qwen_container}" ;;
  esac
}

start_model() {
  [[ "${model_alias}" != all ]] || die "start requires deepseek or qwen"
  if is_true "${sync_on_start}"; then
    sync_remote
  fi
  run_remote Install-ServerEnvironment.ps1 -CheckOnly
  stop_all

  local build_id
  build_id="$(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || printf development)"
  local common=(
    -TaskName "${task_name}"
    -Port "${port}"
    -MaximumContext "${max_context}"
    -MaximumNewTokens "${max_output}"
    -WorkerKvCacheMiB 2048
    -WorkerKvPageTokens 256
    -PlacementProfile balanced
    -GenerationTimeoutSeconds "${generation_timeout}"
    -StartupTimeoutSeconds 600
    -BuildId "${build_id}"
    -Start
  )
  if [[ "${model_alias}" == deepseek ]]; then
    run_remote Install-ExpertServerTask.ps1 \
      -Profile DeepSeekV4Flash \
      -Bundle "${deepseek_bundle}" \
      -MaximumQueue 4 \
      -WorkerCapacity 1 \
      -WorkerRamCacheGiB 40 \
      -WorkerVramCacheGiB 12 \
      -EnableMtp \
      "${common[@]}"
  else
    run_remote Install-ExpertServerTask.ps1 \
      -Profile P6 \
      -Container "${qwen_container}" \
      -MaximumQueue 8 \
      -WorkerCapacity 4 \
      -WorkerRamCacheGiB 48 \
      -WorkerVramCacheGiB 18 \
      "${common[@]}"
  fi
  run_remote Get-ExpertServerStatus.ps1 \
    -Port "${port}" \
    -WaitSeconds "${ready_timeout}" \
    -ExpectedModel "${model_id}" \
    -ExpectedContext "${max_context}" \
    -ExpectedMaximumNewTokens "${max_output}" \
    -ExpectedGenerationTimeoutSeconds "${generation_timeout}"
}

case "${action}" in
  install)
    sync_remote
    run_remote Install-ServerEnvironment.ps1
    ;;
  sync)
    sync_remote
    ;;
  start)
    start_model
    ;;
  stop)
    if [[ "${model_alias}" == all ]]; then stop_all; else stop_task "${task_name}"; fi
    ;;
  restart)
    [[ "${model_alias}" != all ]] || die "restart requires deepseek or qwen"
    stop_task "${task_name}"
    start_model
    ;;
  status)
    run_remote Get-ExpertServerStatus.ps1 -Port "${port}"
    ;;
  chat)
    [[ "${model_alias}" != all ]] || die "chat requires deepseek or qwen"
    if (( selection_explicit )); then
      exec "${script_dir}/chat.sh" --model "${model_id}"
    else
      exec "${script_dir}/chat.sh" --model auto
    fi
    ;;
  config)
    print_config
    ;;
  -h|--help|help)
    usage
    ;;
  *) usage >&2; die "unknown action '${action}'" ;;
esac
