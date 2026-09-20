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
command -v python3 >/dev/null 2>&1 || die "python3 is required for runtime validation"

alias_file="${script_dir}/model-aliases.tsv"
[[ -f "${alias_file}" ]] || die "model alias registry is missing: ${alias_file}"
model_id=""
expected_kv=""
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
    expected_kv="${declared_kv}"
    break
  fi
done < "${alias_file}"

[[ "${alias_registry_version}" == model-aliases-v3 ]] ||
  die "unsupported model alias registry version"

[[ -n "${model_id}" ]] || die "unknown model selector '${selection}'"

models_file="${PI_MODELS_FILE:-${HOME}/.pi/agent/models.json}"
[[ -f "${models_file}" ]] || die "Pi model registry is missing: ${models_file}"
if ! python3 - "${models_file}" "${model_id}" "${expected_kv}" <<'PY'
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

models_file, expected_model, expected_kv = sys.argv[1:]
with open(models_file, encoding="utf-8") as handle:
    registry = json.load(handle)
try:
    base_url = registry["providers"]["quantum-llm"]["baseUrl"]
except (KeyError, TypeError) as error:
    raise SystemExit(f"Pi quantum-llm provider has no baseUrl: {error}")
parsed = urllib.parse.urlsplit(str(base_url).rstrip("/"))
path = parsed.path
if path.endswith("/v1"):
    path = path[:-3]
model_info_url = urllib.parse.urlunsplit(
    (parsed.scheme, parsed.netloc, path + "/model-info", "", "")
)
request = urllib.request.Request(
    model_info_url,
    headers={"Authorization": f"Bearer {os.environ['EXPERT_API_KEY']}"},
)
try:
    with urllib.request.urlopen(request, timeout=10.0) as response:
        info = json.load(response)
except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
    raise SystemExit(f"cannot validate the running Quantum LLM service: {error}")
actual_model = info.get("model")
if actual_model != expected_model:
    raise SystemExit(
        f"running model is {actual_model!r}, expected {expected_model!r}"
    )
actual_kv = (info.get("worker_kv") or {}).get("dtype")
if expected_kv != "artifact" and actual_kv != expected_kv:
    raise SystemExit(
        f"running KV codec is {actual_kv!r}, expected {expected_kv!r}"
    )
PY
then
  die "start the selected model/codec before launching Pi"
fi

exec pi \
  --approve \
  --provider quantum-llm \
  --model "${model_id}" \
  --thinking xhigh \
  "$@"
