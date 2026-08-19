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
Usage: ./ops/model.sh <install|sync|start|stop|restart|status|chat|config> [deepseek|qwen|<artifact-name>|all]

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
model_id="${selection}"
artifact_name="${selection}"
task_name="QuantumLLM-ExpertVm"
if [[ "${selection}" == all ]]; then
  model_id=""
  artifact_name=""
  task_name=""
else
  [[ "${selection}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    die "model selector must contain only letters, digits, dot, underscore or dash"
  alias_file="${MODEL_ALIAS_FILE:-${script_dir}/model-aliases.tsv}"
  [[ -f "${alias_file}" ]] || die "model alias registry is missing: ${alias_file}"
  while IFS=$'\t' read -r alias advertised_model artifact extra; do
    [[ "${alias}" != model-aliases-v1 && -n "${alias}" ]] || continue
    [[ -z "${extra}" && -n "${advertised_model}" && -n "${artifact}" ]] ||
      die "invalid model alias registry row for '${alias}'"
    if [[ "${alias}" == "${selection}" ]]; then
      model_id="${advertised_model}"
      artifact_name="${artifact}"
      break
    fi
  done < "${alias_file}"
  [[ "${artifact_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$ ]] ||
    die "model alias resolves outside MODEL_ROOT"
fi

remote_host="${QUANTUM_LLM_REMOTE:-${CHAT_SSH:-}}"
remote_root="${QUANTUM_LLM_REMOTE_ROOT:-}"
model_root="${MODEL_ROOT:-}"
port="${MODEL_PORT:-${CHAT_REMOTE_PORT:-8080}}"
max_context="${MODEL_MAX_CONTEXT:-65536}"
max_output="${MODEL_MAX_OUTPUT_TOKENS:-8192}"
ready_timeout="${MODEL_READY_TIMEOUT:-600}"
generation_timeout="${MODEL_GENERATION_TIMEOUT_SECONDS:-600}"
sync_on_start="${MODEL_SYNC_ON_START:-1}"
ram_cache_gib="${MODEL_RAM_CACHE_GIB:-48}"
vram_cache_gib="${MODEL_VRAM_CACHE_GIB:-13}"
worker_capacity="${MODEL_WORKER_CAPACITY:-1}"
maximum_queue="${MODEL_MAXIMUM_QUEUE:-4}"
kv_cache_mib="${MODEL_KV_CACHE_MIB:-2048}"
kv_page_tokens="${MODEL_KV_PAGE_TOKENS:-256}"
placement_profile="${MODEL_PLACEMENT_PROFILE:-balanced}"
[[ -n "${remote_root}" ]] || die "QUANTUM_LLM_REMOTE_ROOT must reference the remote project root"
[[ -n "${model_root}" ]] || die "MODEL_ROOT must reference the remote model store"
container="${model_root}/${artifact_name}"
vm_runner="${MODEL_VM_RUNNER:-${remote_root}/out/build/windows-msvc-release/runtime/Release/expert-moe-vm-runner.exe}"

require_uint MODEL_PORT "${port}"
require_uint MODEL_MAX_CONTEXT "${max_context}"
require_uint MODEL_MAX_OUTPUT_TOKENS "${max_output}"
require_uint MODEL_READY_TIMEOUT "${ready_timeout}"
require_uint MODEL_GENERATION_TIMEOUT_SECONDS "${generation_timeout}"
require_uint MODEL_RAM_CACHE_GIB "${ram_cache_gib}"
require_uint MODEL_VRAM_CACHE_GIB "${vram_cache_gib}"
require_uint MODEL_WORKER_CAPACITY "${worker_capacity}"
require_uint MODEL_MAXIMUM_QUEUE "${maximum_queue}"
require_uint MODEL_KV_CACHE_MIB "${kv_cache_mib}"
require_uint MODEL_KV_PAGE_TOKENS "${kv_page_tokens}"
[[ "${placement_profile}" == latency || "${placement_profile}" == balanced ||
   "${placement_profile}" == capacity ]] ||
  die "MODEL_PLACEMENT_PROFILE must be latency, balanced, or capacity"
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
  stop_task QuantumLLM-ExpertVm
}

print_config() {
  printf '%s\n' \
    "model=${model_id:-all}" \
    "remote=${remote_host:-<unset>}" \
    "remote_root=${remote_root}" \
    "model_root=${model_root}" \
    "port=${port}" \
    "max_context=${max_context}" \
    "max_output_tokens=${max_output}" \
    "generation_timeout_seconds=${generation_timeout}" \
    "sync_on_start=${sync_on_start}"
  [[ -z "${model_id}" ]] || printf '%s\n' \
    "container=${container}" \
    "runner=${vm_runner}" \
    "ram_cache_gib=${ram_cache_gib}" \
    "vram_cache_gib=${vram_cache_gib}" \
    "worker_capacity=${worker_capacity}" \
    "kv_cache_mib=${kv_cache_mib}" \
    "kv_page_tokens=${kv_page_tokens}" \
    "placement_profile=${placement_profile}"
}

start_model() {
  [[ -n "${model_id}" ]] || die "start requires one model"
  if is_true "${sync_on_start}"; then
    sync_remote
  fi
  run_remote Install-ServerEnvironment.ps1 -CheckOnly
  stop_all

  local build_id
  build_id="${MODEL_BUILD_ID:-$(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || printf development)}"
  local common=(
    -TaskName "${task_name}"
    -Port "${port}"
    -MaximumContext "${max_context}"
    -MaximumNewTokens "${max_output}"
    -WorkerKvCacheMiB "${kv_cache_mib}"
    -WorkerKvPageTokens "${kv_page_tokens}"
    -PlacementProfile "${placement_profile}"
    -GenerationTimeoutSeconds "${generation_timeout}"
    -StartupTimeoutSeconds 600
    -BuildId "${build_id}"
    -Start
  )
  run_remote Install-ExpertServerTask.ps1 \
    -Container "${container}" \
    -Runner "${vm_runner}" \
    -ModelId "${model_id}" \
    -MaximumQueue "${maximum_queue}" \
    -WorkerCapacity "${worker_capacity}" \
    -WorkerRamCacheGiB "${ram_cache_gib}" \
    -WorkerVramCacheGiB "${vram_cache_gib}" \
    "${common[@]}"
  run_remote Get-ExpertServerStatus.ps1 \
    -Port "${port}" \
    -WaitSeconds "${ready_timeout}" \
    -ExpectedModel "${model_id}" \
    -ExpectedTaskName "${task_name}" \
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
    if [[ -z "${model_id}" ]]; then stop_all; else stop_task "${task_name}"; fi
    ;;
  restart)
    [[ -n "${model_id}" ]] || die "restart requires one model"
    stop_task "${task_name}"
    start_model
    ;;
  status)
    run_remote Get-ExpertServerStatus.ps1 -Port "${port}"
    ;;
  chat)
    [[ -n "${model_id}" ]] || die "chat requires one model"
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
