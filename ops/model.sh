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

usage() {
  cat <<'EOF'
Usage: ./ops/model.sh <install|sync|start|stop|restart|status|chat|clear-cache|config> [qwen|qwen-abliterated|qwen-f16|qwen-flash|muse|ornith|mistral|<artifact-name>|all]

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
selection="${2:-${CHAT_MODEL:-qwen3.8-27b-fp4}}"
extra_arguments=("${@:3}")
model_id="${selection}"
artifact_name="${selection}"
alias_kv_cache_dtype=""
alias_routed_vram_policy="fixed"
task_name="QuantumLLM-ExpertVm"
if [[ "${selection}" == all ]]; then
  model_id=""
  artifact_name=""
  task_name=""
else
  [[ "${selection}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    die "model selector must contain only letters, digits, dot, underscore or dash"
  alias_file="${script_dir}/model-aliases.tsv"
  [[ -f "${alias_file}" ]] || die "model alias registry is missing: ${alias_file}"
  alias_registry_version=""
  while IFS=$'\t' read -r alias advertised_model artifact declared_kv declared_vram extra; do
    if [[ "${alias}" == model-aliases-v3 ]]; then
      [[ -z "${advertised_model}${artifact}${declared_kv}${declared_vram}${extra}" ]] ||
        die "invalid model alias registry header"
      alias_registry_version="${alias}"
      continue
    fi
    [[ -n "${alias}" ]] || continue
    [[ -z "${extra}" && -n "${advertised_model}" && -n "${artifact}" &&
       ( "${declared_kv}" == artifact ||
         "${declared_kv}" == fp8-e4m3-per-head ||
         "${declared_kv}" == fp4-e2m1-ue8m0-block32-key-outlier1 ||
         "${declared_kv}" == q4-bfp16-block32-key-outlier1 ||
         "${declared_kv}" == q4-bfp16-block32 ||
         "${declared_kv}" == q4-f16-per-head ||
         "${declared_kv}" == q5-q4-bfp16-block32 ||
         "${declared_kv}" == fp16 ) &&
       ( "${declared_vram}" == fixed || "${declared_vram}" == fit ) ]] ||
      die "invalid model alias registry row for '${alias}'"
    if [[ "${alias}" == "${selection}" ]]; then
      model_id="${advertised_model}"
      artifact_name="${artifact}"
      alias_kv_cache_dtype="${declared_kv}"
      alias_routed_vram_policy="${declared_vram}"
      break
    fi
  done < "${alias_file}"
  [[ "${alias_registry_version}" == model-aliases-v3 ]] ||
    die "unsupported model alias registry version"
  [[ "${artifact_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$ ]] ||
    die "model alias resolves outside MODEL_ROOT"
fi

remote_host="${QUANTUM_LLM_REMOTE:-}"
remote_root="${QUANTUM_LLM_REMOTE_ROOT:-}"
model_root="${MODEL_ROOT:-}"
port="${MODEL_PORT:-8080}"
host_address="${MODEL_HOST:-127.0.0.1}"
api_key="${EXPERT_API_KEY:-}"
max_context="${MODEL_MAX_CONTEXT:-65536}"
max_output="${MODEL_MAX_OUTPUT_TOKENS:-}"
ready_timeout="${MODEL_READY_TIMEOUT:-600}"
generation_timeout="${MODEL_GENERATION_TIMEOUT_SECONDS:-600}"
max_body_mib="${MODEL_MAX_BODY_MIB:-16}"
max_image_pixels="${MODEL_MAX_IMAGE_PIXELS:-2097152}"
max_image_patch_tokens="${MODEL_MAX_IMAGE_PATCH_TOKENS:-4096}"
sync_on_start="${MODEL_SYNC_ON_START:-1}"
ram_cache_gib="${MODEL_RAM_CACHE_GIB:-48}"
vram_cache_gib="${MODEL_VRAM_CACHE_GIB:-12}"
routed_vram_policy="${alias_routed_vram_policy}"
active_expert_devices="${MODEL_ACTIVE_EXPERT_DEVICES:-}"
active_expert_device_cache_gib="${MODEL_ACTIVE_EXPERT_DEVICE_CACHE_GIB:-0}"
active_expert_host_cache_gib="${MODEL_ACTIVE_EXPERT_HOST_CACHE_GIB:-0}"
worker_capacity="${MODEL_WORKER_CAPACITY:-1}"
maximum_queue="${MODEL_MAXIMUM_QUEUE:-4}"
kv_cache_mib="${MODEL_KV_CACHE_MIB:-2048}"
kv_page_tokens="${MODEL_KV_PAGE_TOKENS:-256}"
kv_cache_dtype="${alias_kv_cache_dtype:-${MODEL_KV_CACHE_DTYPE:-artifact}}"
placement_profile="${MODEL_PLACEMENT_PROFILE:-balanced}"
profile_gpu_phases="${MODEL_PROFILE_GPU_PHASES:-0}"
raw_response_trace_file="${MODEL_RAW_RESPONSE_TRACE_FILE:-}"
session_cache_gib="${MODEL_SESSION_CACHE_GIB:-64}"
session_cache_ttl_seconds="${MODEL_SESSION_CACHE_TTL_SECONDS:-604800}"
disable_session_retention="${MODEL_DISABLE_SESSION_RETENTION:-0}"
[[ -n "${remote_root}" ]] || die "QUANTUM_LLM_REMOTE_ROOT must reference the remote project root"
[[ -n "${model_root}" ]] || die "MODEL_ROOT must reference the remote model store"
container="${model_root}/${artifact_name}"
vm_runner="${remote_root}/out/build/windows-msvc-release/runtime/Release/expert-moe-vm-runner.exe"

require_uint MODEL_PORT "${port}"
require_uint MODEL_MAX_CONTEXT "${max_context}"
if [[ -z "${max_output}" ]]; then
  max_output=$((max_context - 1))
fi
require_uint MODEL_MAX_OUTPUT_TOKENS "${max_output}"
require_uint MODEL_READY_TIMEOUT "${ready_timeout}"
require_uint MODEL_GENERATION_TIMEOUT_SECONDS "${generation_timeout}"
require_uint MODEL_MAX_BODY_MIB "${max_body_mib}"
require_uint MODEL_MAX_IMAGE_PIXELS "${max_image_pixels}"
require_uint MODEL_MAX_IMAGE_PATCH_TOKENS "${max_image_patch_tokens}"
require_uint MODEL_RAM_CACHE_GIB "${ram_cache_gib}"
require_uint MODEL_VRAM_CACHE_GIB "${vram_cache_gib}"
require_uint MODEL_WORKER_CAPACITY "${worker_capacity}"
require_uint MODEL_MAXIMUM_QUEUE "${maximum_queue}"
require_uint MODEL_KV_CACHE_MIB "${kv_cache_mib}"
require_uint MODEL_KV_PAGE_TOKENS "${kv_page_tokens}"
[[ "${session_cache_gib}" =~ ^[0-9]+$ ]] ||
  die "MODEL_SESSION_CACHE_GIB must be a non-negative integer"
[[ "${session_cache_ttl_seconds}" =~ ^[0-9]+$ ]] ||
  die "MODEL_SESSION_CACHE_TTL_SECONDS must be a non-negative integer"
if [[ -n "${active_expert_devices}" ]]; then
  [[ "${active_expert_devices}" == auto ||
     "${active_expert_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]] ||
    die "MODEL_ACTIVE_EXPERT_DEVICES must be auto or a comma-separated CUDA ordinal list"
  require_uint MODEL_ACTIVE_EXPERT_DEVICE_CACHE_GIB \
    "${active_expert_device_cache_gib}"
  require_uint MODEL_ACTIVE_EXPERT_HOST_CACHE_GIB \
    "${active_expert_host_cache_gib}"
  (( active_expert_host_cache_gib < ram_cache_gib )) ||
    die "MODEL_ACTIVE_EXPERT_HOST_CACHE_GIB must be smaller than MODEL_RAM_CACHE_GIB"
elif [[ "${active_expert_device_cache_gib}" != 0 ||
        "${active_expert_host_cache_gib}" != 0 ]]; then
  die "active expert cache budgets require MODEL_ACTIVE_EXPERT_DEVICES"
fi
[[ "${host_address}" =~ ^[A-Za-z0-9:.%-]+$ ]] ||
  die "MODEL_HOST contains unsupported characters"
if [[ "${host_address}" != 127.0.0.1 && "${host_address}" != ::1 &&
      "${host_address}" != localhost && -z "${api_key}" ]]; then
  die "EXPERT_API_KEY is required when MODEL_HOST is non-loopback"
fi
[[ "${placement_profile}" == latency || "${placement_profile}" == balanced ||
   "${placement_profile}" == capacity ]] ||
  die "MODEL_PLACEMENT_PROFILE must be latency, balanced, or capacity"
case "${profile_gpu_phases}" in
  1|true|TRUE|yes|YES|0|false|FALSE|no|NO) ;;
  *) die "MODEL_PROFILE_GPU_PHASES must be a boolean" ;;
esac
case "${disable_session_retention}" in
  1|true|TRUE|yes|YES|0|false|FALSE|no|NO) ;;
  *) die "MODEL_DISABLE_SESSION_RETENTION must be a boolean" ;;
esac
[[ "${kv_cache_dtype}" == artifact ||
   "${kv_cache_dtype}" == fp8-e4m3-per-head ||
   "${kv_cache_dtype}" == fp4-e2m1-ue8m0-block32-key-outlier1 ||
   "${kv_cache_dtype}" == q4-bfp16-block32-key-outlier1 ||
   "${kv_cache_dtype}" == q4-bfp16-block32 ||
   "${kv_cache_dtype}" == q4-f16-per-head ||
   "${kv_cache_dtype}" == q5-q4-bfp16-block32 ||
   "${kv_cache_dtype}" == fp16 ]] ||
  die "MODEL_KV_CACHE_DTYPE must be artifact, fp8-e4m3-per-head, fp4-e2m1-ue8m0-block32-key-outlier1, q4-bfp16-block32-key-outlier1, q4-bfp16-block32, q4-f16-per-head, q5-q4-bfp16-block32, or fp16"
(( max_output < max_context )) || die "MODEL_MAX_OUTPUT_TOKENS must be smaller than MODEL_MAX_CONTEXT"

export QUANTUM_LLM_REMOTE="${remote_host}"
export QUANTUM_LLM_REMOTE_ROOT="${remote_root}"

require_remote() {
  [[ -n "${remote_host}" ]] || die "set QUANTUM_LLM_REMOTE in ${env_file}"
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
    "host=${host_address}" \
    "port=${port}" \
    "max_context=${max_context}" \
    "max_output_tokens=${max_output}" \
    "generation_timeout_seconds=${generation_timeout}" \
    "max_body_mib=${max_body_mib}" \
    "max_image_pixels=${max_image_pixels}" \
    "max_image_patch_tokens=${max_image_patch_tokens}" \
    "sync_on_start=${sync_on_start}" \
    "api_key_configured=$([[ -n "${api_key}" ]] && printf yes || printf no)"
  [[ -z "${model_id}" ]] || printf '%s\n' \
    "container=${container}" \
    "runner=${vm_runner}" \
    "ram_cache_gib=${ram_cache_gib}" \
    "vram_cache_gib=${vram_cache_gib}" \
    "routed_vram_policy=${routed_vram_policy}" \
    "active_expert_devices=${active_expert_devices}" \
    "active_expert_device_cache_gib=${active_expert_device_cache_gib}" \
    "active_expert_host_cache_gib=${active_expert_host_cache_gib}" \
    "worker_capacity=${worker_capacity}" \
    "kv_cache_mib=${kv_cache_mib}" \
    "kv_page_tokens=${kv_page_tokens}" \
    "kv_cache_dtype=${kv_cache_dtype}" \
    "placement_profile=${placement_profile}" \
    "profile_gpu_phases=${profile_gpu_phases}" \
    "session_cache_gib=${session_cache_gib}" \
    "session_cache_ttl_seconds=${session_cache_ttl_seconds}" \
    "disable_session_retention=${disable_session_retention}"
}

start_model() {
  [[ -n "${model_id}" ]] || die "start requires one model"
  if is_true "${sync_on_start}"; then
    sync_remote
  fi
  local contract_json artifact_max_context artifact_max_thinking_output
  local artifact_kv_bytes_per_token
  local artifact_mtp_layers artifact_exact_decode_abi
  local artifact_draft_depth artifact_draft_vocabulary_size
  local artifact_mtp_kv_encoding artifact_maximum_emitted_tokens
  contract_json="$(run_remote Get-ModelArtifactContract.ps1 -Container "${container}")"
  read -r artifact_max_context artifact_max_thinking_output artifact_kv_bytes_per_token \
    artifact_mtp_layers artifact_exact_decode_abi artifact_draft_depth \
    artifact_draft_vocabulary_size artifact_mtp_kv_encoding \
    artifact_maximum_emitted_tokens < <(python3 -c '
import json, sys
contract = json.load(sys.stdin)
maximum = contract.get("maximum_context")
maximum_thinking_output = contract.get("maximum_thinking_tokens", 0)
per_token = contract.get("minimum_exact_kv_bytes_per_token", 0)
if not isinstance(maximum, int) or maximum <= 1:
    raise SystemExit("artifact contract has no valid maximum_context")
if (isinstance(maximum_thinking_output, bool) or
        not isinstance(maximum_thinking_output, int) or
        maximum_thinking_output < 0 or maximum_thinking_output >= maximum):
    raise SystemExit("artifact contract has invalid maximum_thinking_tokens")
if not isinstance(per_token, int) or per_token < 0:
    raise SystemExit("artifact contract has invalid exact-KV geometry")
names = (
    "mtp_layers", "exact_decode_abi", "exact_decode_draft_depth",
    "exact_decode_draft_vocabulary_size", "exact_decode_mtp_kv_encoding",
    "exact_decode_maximum_emitted_tokens",
)
values = []
for name in names:
    value = contract.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit(f"artifact contract has invalid {name}")
    values.append(value)
print(maximum, maximum_thinking_output, per_token, *values)
' <<<"${contract_json}")
  if [[ "${kv_cache_dtype}" == q4-f16-per-head ]] &&
     (( artifact_mtp_layers > 0 )) &&
     ! (( artifact_exact_decode_abi == 2 &&
          artifact_draft_depth >= 3 && artifact_draft_depth <= 4 &&
          artifact_draft_vocabulary_size > 0 &&
          artifact_draft_vocabulary_size <= 65536 &&
          artifact_mtp_kv_encoding == 1 &&
          artifact_maximum_emitted_tokens == artifact_draft_depth + 1 )); then
    die "Q4H artifact with MTP requires exact-decode ABI 2, MTP-3/4, a <=65,536 draft vocabulary, and Q8 MTP KV (got ABI=${artifact_exact_decode_abi} depth=${artifact_draft_depth} vocabulary=${artifact_draft_vocabulary_size} MTP-KV=${artifact_mtp_kv_encoding} emitted=${artifact_maximum_emitted_tokens})"
  fi
  if (( max_context > artifact_max_context )); then
    printf 'Using artifact context limit %s instead of configured %s\n' \
      "${artifact_max_context}" "${max_context}"
    max_context="${artifact_max_context}"
  fi
  if (( artifact_kv_bytes_per_token > 0 )) &&
     [[ "${kv_cache_dtype}" == fp16 || "${kv_cache_dtype}" == artifact ]]; then
    local artifact_kv_mib
    artifact_kv_mib=$((
      (((max_context + kv_page_tokens - 1) / kv_page_tokens) *
        kv_page_tokens * artifact_kv_bytes_per_token + 1048575) / 1048576
    ))
    if (( kv_cache_mib < artifact_kv_mib )); then
      printf 'Using artifact exact-KV minimum %s MiB instead of configured %s MiB\n' \
        "${artifact_kv_mib}" "${kv_cache_mib}"
      kv_cache_mib="${artifact_kv_mib}"
    fi
  fi
  if (( max_output >= max_context )); then
    max_output=$((max_context - 1))
  fi
  run_remote Install-ServerEnvironment.ps1 -CheckOnly
  stop_all

  local build_id
  build_id="$(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || printf development)"
  local common=(
    -TaskName "${task_name}"
    -HostAddress "${host_address}"
    -Port "${port}"
    -MaximumContext "${max_context}"
    -MaximumNewTokens "${max_output}"
    -WorkerKvCacheMiB "${kv_cache_mib}"
    -WorkerKvPageTokens "${kv_page_tokens}"
    -WorkerKvCacheDtype "${kv_cache_dtype}"
    -PlacementProfile "${placement_profile}"
    -GenerationTimeoutSeconds "${generation_timeout}"
    -MaximumBodyMiB "${max_body_mib}"
    -MaximumImagePixels "${max_image_pixels}"
    -MaximumImagePatchTokens "${max_image_patch_tokens}"
    -StartupTimeoutSeconds 600
    -BuildId "${build_id}"
    -Start
  )
  if (( session_cache_gib > 0 )); then
    common+=(
      -SessionCacheRoot "${model_root}/.session-cache/${artifact_name}"
      -SessionCacheGiB "${session_cache_gib}"
      -SessionCacheTtlSeconds "${session_cache_ttl_seconds}"
    )
  fi
  if is_true "${disable_session_retention}"; then
    common+=(-DisableSessionRetention)
  fi
  if [[ -n "${api_key}" ]]; then
    common+=(-ApiKey "${api_key}")
  fi
  if is_true "${profile_gpu_phases}"; then
    common+=(-ProfileGpuPhases)
  fi
  if [[ -n "${active_expert_devices}" ]]; then
    common+=(
      -WorkerActiveExpertDevices "${active_expert_devices}"
      -WorkerActiveExpertDeviceCacheGiB "${active_expert_device_cache_gib}"
      -WorkerActiveExpertHostCacheGiB "${active_expert_host_cache_gib}"
    )
  fi
  if [[ -n "${raw_response_trace_file}" ]]; then
    common+=(-RawResponseTraceFile "${raw_response_trace_file}")
  fi
  run_remote Install-ExpertServerTask.ps1 \
    -Container "${container}" \
    -Runner "${vm_runner}" \
    -ModelId "${model_id}" \
    -MaximumQueue "${maximum_queue}" \
    -WorkerCapacity "${worker_capacity}" \
    -WorkerRamCacheGiB "${ram_cache_gib}" \
    -WorkerVramCacheGiB "${vram_cache_gib}" \
    -WorkerRoutedVramPolicy "${routed_vram_policy}" \
    "${common[@]}"
  local status_args=(
    -Port "${port}"
    -WaitSeconds "${ready_timeout}"
    -ExpectedModel "${model_id}"
    -ExpectedTaskName "${task_name}"
    -ExpectedContext "${max_context}"
    -ExpectedMaximumNewTokens "${max_output}"
    -ExpectedGenerationTimeoutSeconds "${generation_timeout}"
    -ExpectedMaximumBodyMiB "${max_body_mib}"
  )
  if [[ -n "${api_key}" ]]; then
    status_args+=(-ApiKey "${api_key}")
  fi
  run_remote Get-ExpertServerStatus.ps1 "${status_args[@]}"
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
    if [[ -n "${api_key}" ]]; then
      run_remote Get-ExpertServerStatus.ps1 -Port "${port}" -ApiKey "${api_key}"
    else
      run_remote Get-ExpertServerStatus.ps1 -Port "${port}"
    fi
    ;;
  chat)
    [[ -n "${model_id}" ]] || die "chat requires one model"
    if (( selection_explicit )); then
      chat_command_args=(--model "${model_id}")
    else
      chat_command_args=(--model auto)
    fi
    if (( ${#extra_arguments[@]} )); then
      chat_command_args+=("${extra_arguments[@]}")
    fi
    exec "${script_dir}/chat.sh" "${chat_command_args[@]}"
    ;;
  clear-cache)
    [[ -n "${model_id}" ]] || die "clear-cache requires one model"
    [[ -n "${artifact_name}" ]] || die "clear-cache requires one model artifact"
    run_remote Clear-ExpertSessionCache.ps1 \
      -ModelRoot "${model_root}" -ArtifactName "${artifact_name}"
    ;;
  config)
    print_config
    ;;
  -h|--help|help)
    usage
    ;;
  *) usage >&2; die "unknown action '${action}'" ;;
esac
