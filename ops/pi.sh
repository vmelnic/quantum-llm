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

die() {
  echo "pi.sh: $*" >&2
  exit 2
}

selection="${1:-${CHAT_MODEL:-qwen}}"
if (( $# > 0 )); then
  shift
fi

[[ "${selection}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
  die "model selector must contain only letters, digits, dot, underscore or dash"
[[ -n "${EXPERT_API_KEY:-}" ]] ||
  die "EXPERT_API_KEY is required in ${env_file}"
command -v pi >/dev/null 2>&1 || die "pi CLI is not installed or not on PATH"

alias_file="${MODEL_ALIAS_FILE:-${script_dir}/model-aliases.tsv}"
[[ -f "${alias_file}" ]] || die "model alias registry is missing: ${alias_file}"
model_id=""
while IFS=$'\t' read -r alias advertised_model artifact extra; do
  [[ "${alias}" != model-aliases-v1 && -n "${alias}" ]] || continue
  [[ -z "${extra}" && -n "${advertised_model}" && -n "${artifact}" ]] ||
    die "invalid model alias registry row for '${alias}'"
  if [[ "${alias}" == "${selection}" ]]; then
    model_id="${advertised_model}"
    break
  fi
done < "${alias_file}"

[[ -n "${model_id}" ]] || die "unknown model selector '${selection}'"

exec pi \
  --approve \
  --provider quantum-llm \
  --model "${model_id}" \
  --thinking xhigh \
  "$@"
