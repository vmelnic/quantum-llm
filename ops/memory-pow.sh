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
action="${1:-status}"

[[ -n "${remote_host}" ]] || { echo "Set QUANTUM_LLM_REMOTE in ${env_file}" >&2; exit 2; }
export QUANTUM_LLM_REMOTE="${remote_host}"
export QUANTUM_LLM_REMOTE_ROOT="${remote_root}"

case "${action}" in
  download)
    "${script_dir}/sync-to-windows-host.sh" "${remote_host}" "${remote_root}"
    "${script_dir}/run-on-windows-host.sh" Start-HuggingFaceModelDownload.ps1 \
      -ModelId Qwen/Qwen3-4B \
      -Revision 1cfa9a7208912126459214e8b04321603b3df60c \
      -ExpectedDownloadBytes 8060926626 \
      -ExpectedTensorBytes 8044936192 \
      -ExpectedShards 3 \
      -MaxWorkers 4
    ;;
  download-status)
    "${script_dir}/run-on-windows-host.sh" Get-HuggingFaceModelDownload.ps1
    ;;
  train|probe|evaluate|run)
    "${script_dir}/sync-to-windows-host.sh" "${remote_host}" "${remote_root}"
    "${script_dir}/run-on-windows-host.sh" Install-MemoryExpertEnvironment.ps1
    "${script_dir}/run-on-windows-host.sh" Invoke-MemoryExpertPow.ps1 \
      -Action "${action}" \
      -Epochs "${MEMORY_POW_EPOCHS:-3}" \
      -BatchSize "${MEMORY_POW_BATCH_SIZE:-2}" \
      -GradientAccumulation "${MEMORY_POW_GRADIENT_ACCUMULATION:-16}" \
      -LearningRate "${MEMORY_POW_LEARNING_RATE:-0.0002}" \
      -Optimizer "${MEMORY_POW_OPTIMIZER:-adamw}" \
      -EvaluationLimit "${MEMORY_POW_EVALUATION_LIMIT:-16}" \
      -GateRank "${MEMORY_POW_GATE_RANK:-16}" \
      -GateAlpha "${MEMORY_POW_GATE_ALPHA:-32}" \
      -InjectionEvery "${MEMORY_POW_INJECTION_EVERY:-1}" \
      -KnowledgeDropout "${MEMORY_POW_KNOWLEDGE_DROPOUT:-0.2}"
    ;;
  status)
    ssh -o BatchMode=yes "${remote_host}" \
      "if exist \"${remote_root}/artifacts/memory-expert-pow-status.json\" (type \"${remote_root}/artifacts/memory-expert-pow-status.json\") else (echo Memory Expert PoW has not run)"
    ;;
  *)
    echo "Usage: ./ops/memory-pow.sh <download|download-status|train|probe|evaluate|run|status>" >&2
    exit 2
    ;;
esac
