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

alias_file="${script_dir}/model-aliases.tsv"
[[ -f "${alias_file}" ]] || die "model alias registry is missing: ${alias_file}"
model_id=""
alias_registry_version=""
while IFS=$'\t' read -r alias advertised_model artifact declared_kv extra; do
  if [[ "${alias}" == model-aliases-v2 ]]; then
    [[ -z "${advertised_model}${artifact}${declared_kv}${extra}" ]] ||
      die "invalid model alias registry header"
    alias_registry_version="${alias}"
    continue
  fi
  [[ -n "${alias}" ]] || continue
  [[ -z "${extra}" && -n "${advertised_model}" && -n "${artifact}" &&
     ( "${declared_kv}" == artifact ||
       "${declared_kv}" == fp8-e4m3-per-head ||
       "${declared_kv}" == fp16 ) ]] ||
    die "invalid model alias registry row for '${alias}'"
  if [[ "${alias}" == "${selection}" ]]; then
    model_id="${advertised_model}"
    break
  fi
done < "${alias_file}"

[[ "${alias_registry_version}" == model-aliases-v2 ]] ||
  die "unsupported model alias registry version"

[[ -n "${model_id}" ]] || die "unknown model selector '${selection}'"

exec pi \
  --approve \
  --provider quantum-llm \
  --model "${model_id}" \
  --thinking xhigh \
  "$@"
