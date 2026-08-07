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

remote_host="${QUANTUM_LLM_REMOTE:-${CHAT_SSH:-}}"
remote_root="${QUANTUM_LLM_REMOTE_ROOT:-C:/quantum-llm}"
dataset_name="${MEMORY_DATASET_NAME:-memory-dataset}"
local_dataset_root="${repo_root}/work/memory-data/${dataset_name}"
remote_dataset_root="${remote_root}/work/memory-data/${dataset_name}"
config_path="${MEMORY_INGEST_CONFIG:-${repo_root}/experiments/memory_expert/configs/json-article-v1.json}"
if [[ "${config_path}" != /* ]]; then
  config_path="${repo_root}/${config_path}"
fi
encoder_model="${MEMORY_ENCODER_MODEL:-BAAI/bge-m3}"
encoder_revision="${MEMORY_ENCODER_REVISION:-5617a9f61b028005a4858fdac845db406aefb181}"
action="${1:-status}"

[[ "${dataset_name}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid MEMORY_DATASET_NAME" >&2; exit 2; }
if [[ "${action}" != "ingest" ]]; then
  [[ -n "${remote_host}" ]] || { echo "Set QUANTUM_LLM_REMOTE in ${env_file}" >&2; exit 2; }
  export QUANTUM_LLM_REMOTE="${remote_host}"
  export QUANTUM_LLM_REMOTE_ROOT="${remote_root}"
fi

sync_data() {
  [[ -f "${local_dataset_root}/ingest/manifest.json" ]] || {
    echo "Run ./ops/memory-data.sh ingest first" >&2; exit 2;
  }
  "${script_dir}/sync-to-windows-host.sh" "${remote_host}" "${remote_root}"
  ssh -o BatchMode=yes "${remote_host}" \
    "if not exist \"${remote_dataset_root}\" mkdir \"${remote_dataset_root}\""
  tar --format=ustar --no-xattrs --no-acls --no-fflags \
    -C "${local_dataset_root}" -cf - ingest \
    | ssh -o BatchMode=yes "${remote_host}" \
      "tar -xf - -C \"${remote_dataset_root}\""
  if [[ -f "${local_dataset_root}/questions.jsonl" ]]; then
    tar --format=ustar --no-xattrs --no-acls --no-fflags \
      -C "${local_dataset_root}" -cf - questions.jsonl \
      | ssh -o BatchMode=yes "${remote_host}" \
        "tar -xf - -C \"${remote_dataset_root}\""
  fi
}

case "${action}" in
  ingest)
    [[ -n "${MEMORY_INGEST_SOURCE:-}" ]] || {
      echo "Set MEMORY_INGEST_SOURCE in ${env_file}" >&2; exit 2;
    }
    mkdir -p "${local_dataset_root}"
    python3 "${repo_root}/experiments/memory_expert/ingest.py" \
      --input "${MEMORY_INGEST_SOURCE}" \
      --config "${config_path}" \
      --source-uri "${MEMORY_SOURCE_URI:-source://${dataset_name}}" \
      --output "${local_dataset_root}/ingest"
    ;;
  sync)
    sync_data
    ;;
  encoder-download)
    "${script_dir}/sync-to-windows-host.sh" "${remote_host}" "${remote_root}"
    "${script_dir}/run-on-windows-host.sh" Install-MemoryEncoder.ps1 \
      -ModelId "${encoder_model}" -Revision "${encoder_revision}"
    ;;
  index|query|status|selftest)
    remote_arguments=(Invoke-MemoryData.ps1 \
      -Action "${action}" -DatasetName "${dataset_name}" \
      -EncoderModel "${encoder_model}" -EncoderRevision "${encoder_revision}" \
      -EncoderMaximumTokens "${MEMORY_ENCODER_MAX_TOKENS:-1024}" \
      -EncoderBatchSize "${MEMORY_ENCODER_BATCH_SIZE:-16}" \
      -TopK "${MEMORY_RETRIEVAL_TOP_K:-2}" \
      -MaximumMemoryTokens "${MEMORY_MAX_MEMORY_TOKENS:-768}" \
      -MaximumNewTokens "${MEMORY_MAX_NEW_TOKENS:-96}")
    if [[ -n "${MEMORY_QUERY_LANGUAGE:-}" ]]; then
      remote_arguments+=(-Language "${MEMORY_QUERY_LANGUAGE}")
    fi
    "${script_dir}/run-on-windows-host.sh" "${remote_arguments[@]}"
    ;;
  *)
    echo "Usage: ./ops/memory-data.sh <ingest|sync|encoder-download|selftest|index|query|status>" >&2
    exit 2
    ;;
esac
