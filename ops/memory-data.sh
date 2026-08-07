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
capability_corpus="${MEMORY_CAPABILITY_CORPUS:-work/memory-capability/xquad-coherent-v2.jsonl}"
if [[ "${capability_corpus}" != /* ]]; then
  capability_corpus="${repo_root}/${capability_corpus}"
fi
capability_corpus_name="$(basename "${capability_corpus}")"
remote_capability_dir="${remote_root}/work/memory-capability"
remote_capability_corpus="${remote_capability_dir}/${capability_corpus_name}"
capability_rewrite_jobs="${MEMORY_CAPABILITY_REWRITE_JOBS:-${repo_root}/work/memory-capability/counterfactual-rewrite-jobs.jsonl}"
capability_rewrites="${MEMORY_CAPABILITY_REWRITES:-${repo_root}/work/memory-capability/counterfactual-rewrites.jsonl}"
capability_source_cache="${MEMORY_CAPABILITY_SOURCE_CACHE:-${repo_root}/work/memory-capability/source-cache}"
if [[ "${capability_rewrite_jobs}" != /* ]]; then
  capability_rewrite_jobs="${repo_root}/${capability_rewrite_jobs}"
fi
if [[ "${capability_rewrites}" != /* ]]; then
  capability_rewrites="${repo_root}/${capability_rewrites}"
fi
if [[ "${capability_source_cache}" != /* ]]; then
  capability_source_cache="${repo_root}/${capability_source_cache}"
fi

[[ "${dataset_name}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid MEMORY_DATASET_NAME" >&2; exit 2; }
if [[ "${action}" != "ingest" && "${action}" != "capability-prepare" \
   && "${action}" != "capability-rewrite" && "${action}" != "capability-build" ]]; then
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

sync_capability_data() {
  [[ -f "${capability_corpus}" ]] || {
    echo "Run capability-prepare, capability-rewrite, and capability-build first" >&2; exit 2;
  }
  python3 "${repo_root}/experiments/memory_expert/prepare_capability_data.py" \
    --output "${capability_corpus}" --validate-only >/dev/null
  "${script_dir}/sync-to-windows-host.sh" "${remote_host}" "${remote_root}"
  ssh -o BatchMode=yes "${remote_host}" \
    "if not exist \"${remote_capability_dir}\" mkdir \"${remote_capability_dir}\""
  capability_files=("${capability_corpus_name}")
  if [[ -f "${capability_corpus}.manifest.json" ]]; then
    capability_files+=("${capability_corpus_name}.manifest.json")
  fi
  tar --format=ustar --no-xattrs --no-acls --no-fflags \
    -C "$(dirname "${capability_corpus}")" -cf - "${capability_files[@]}" \
    | ssh -o BatchMode=yes "${remote_host}" \
      "tar -xf - -C \"${remote_capability_dir}\""
}

case "${action}" in
  capability-prepare)
    python3 "${repo_root}/experiments/memory_expert/prepare_capability_data.py" \
      --output "${capability_corpus}" \
      --source-cache "${capability_source_cache}" \
      --rewrite-jobs "${capability_rewrite_jobs}"
    ;;
  capability-rewrite)
    [[ -f "${capability_rewrite_jobs}" ]] || {
      echo "Run ./ops/memory-data.sh capability-prepare first" >&2; exit 2;
    }
    python3 "${repo_root}/experiments/memory_expert/generate_counterfactual_rewrites.py" \
      --jobs "${capability_rewrite_jobs}" \
      --output "${capability_rewrites}" \
      --limit "${MEMORY_REWRITE_LIMIT:-0}" \
      --attempts "${MEMORY_REWRITE_ATTEMPTS:-3}" \
      --timeout "${MEMORY_REWRITE_TIMEOUT:-180}"
    ;;
  capability-build)
    [[ -f "${capability_rewrites}" ]] || {
      echo "Run ./ops/memory-data.sh capability-rewrite first" >&2; exit 2;
    }
    python3 "${repo_root}/experiments/memory_expert/prepare_capability_data.py" \
      --output "${capability_corpus}" \
      --source-cache "${capability_source_cache}" \
      --rewrite-jobs "${capability_rewrite_jobs}" \
      --rewrites "${capability_rewrites}"
    ;;
  capability-sync)
    sync_capability_data
    ;;
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
  capability-train|capability-probe|capability-evaluate|capability-run)
    capability_action="${action#capability-}"
    sync_capability_data
    capability_arguments=(Invoke-MemoryCapability.ps1 \
      -Action "${capability_action}" \
      -Corpus "${remote_capability_corpus}" \
      -Steps "${MEMORY_CAPABILITY_STEPS:-4096}" \
      -BatchSize "${MEMORY_CAPABILITY_BATCH_SIZE:-2}" \
      -GradientAccumulation "${MEMORY_CAPABILITY_GRADIENT_ACCUMULATION:-16}" \
      -LearningRate "${MEMORY_CAPABILITY_LEARNING_RATE:-0.0002}" \
      -EvaluationLimit "${MEMORY_CAPABILITY_EVALUATION_LIMIT:-30}" \
      -GateRank "${MEMORY_CAPABILITY_GATE_RANK:-16}" \
      -GateAlpha "${MEMORY_CAPABILITY_GATE_ALPHA:-32}" \
      -KnowledgeDropout "${MEMORY_CAPABILITY_KNOWLEDGE_DROPOUT:-0.2}" \
      -MaximumMemoryTokens "${MEMORY_CAPABILITY_MEMORY_TOKENS:-384}" \
      -MaximumNewTokens "${MEMORY_CAPABILITY_NEW_TOKENS:-128}" \
      -MemoryCacheBytes "${MEMORY_CAPABILITY_CACHE_BYTES:-4294967296}" \
      -OutputName "${MEMORY_CAPABILITY_OUTPUT_NAME:-memory-expert-capability-natural-v2}")
    if [[ -f "${local_dataset_root}/questions.jsonl" ]]; then
      ssh -o BatchMode=yes "${remote_host}" \
        "if not exist \"${remote_dataset_root}\" mkdir \"${remote_dataset_root}\""
      tar --format=ustar --no-xattrs --no-acls --no-fflags \
        -C "${local_dataset_root}" -cf - questions.jsonl \
        | ssh -o BatchMode=yes "${remote_host}" \
          "tar -xf - -C \"${remote_dataset_root}\""
      capability_arguments+=(
        -ForbiddenFile "${remote_dataset_root}/questions.jsonl"
      )
    fi
    "${script_dir}/run-on-windows-host.sh" "${capability_arguments[@]}"
    ;;
  index|query|status|selftest)
    remote_arguments=(Invoke-MemoryData.ps1 \
      -Action "${action}" -DatasetName "${dataset_name}" \
      -EncoderModel "${encoder_model}" -EncoderRevision "${encoder_revision}" \
      -EncoderMaximumTokens "${MEMORY_ENCODER_MAX_TOKENS:-1024}" \
      -EncoderBatchSize "${MEMORY_ENCODER_BATCH_SIZE:-16}" \
      -TopK "${MEMORY_RETRIEVAL_TOP_K:-2}" \
      -MaximumMemoryTokens "${MEMORY_MAX_MEMORY_TOKENS:-768}" \
      -MaximumNewTokens "${MEMORY_MAX_NEW_TOKENS:-128}" \
      -Checkpoint "${MEMORY_ADAPTER_CHECKPOINT:-work/memory-expert-capability-natural-v2/memory-expert.pt}")
    if [[ -n "${MEMORY_QUERY_LANGUAGE:-}" ]]; then
      remote_arguments+=(-Language "${MEMORY_QUERY_LANGUAGE}")
    fi
    "${script_dir}/run-on-windows-host.sh" "${remote_arguments[@]}"
    ;;
  *)
    echo "Usage: ./ops/memory-data.sh <ingest|sync|encoder-download|selftest|index|query|status|capability-prepare|capability-rewrite|capability-build|capability-sync|capability-train|capability-probe|capability-evaluate|capability-run>" >&2
    exit 2
    ;;
esac
