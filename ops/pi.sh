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
    provider = registry["providers"]["quantum-llm"]
    base_url = provider["baseUrl"]
except (KeyError, TypeError) as error:
    raise SystemExit(f"Pi quantum-llm provider has no baseUrl: {error}")
models = provider.get("models")
if not isinstance(models, list):
    raise SystemExit("Pi quantum-llm provider has no model registry")
configured = next(
    (model for model in models
     if isinstance(model, dict) and model.get("id") == expected_model),
    None,
)
if configured is None:
    raise SystemExit(f"Pi has no Quantum LLM model {expected_model!r}")
client_maximum = configured.get("maxTokens")
if (isinstance(client_maximum, bool) or
        not isinstance(client_maximum, int) or client_maximum <= 0):
    raise SystemExit(
        f"Pi model {expected_model!r} has invalid maxTokens"
    )
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
service_maximum = (info.get("runtime_config") or {}).get(
    "maximum_new_tokens"
)
if (isinstance(service_maximum, bool) or
        not isinstance(service_maximum, int) or
        client_maximum > service_maximum):
    raise SystemExit(
        "Pi output limit exceeds the running service limit: "
        f"Pi={client_maximum!r}, service={service_maximum!r}"
    )
actual_kv = (info.get("worker_kv") or {}).get("dtype")
if expected_kv != "artifact" and actual_kv != expected_kv:
    raise SystemExit(
        f"running KV codec is {actual_kv!r}, expected {expected_kv!r}"
    )
execution = info.get("worker_execution") or {}
runtime = info.get("worker_runtime") or {}
if expected_kv == "q4-f16-per-head" and execution.get("mtp_enabled") is True:
    abi = runtime.get("provider_exact_decode_abi")
    depth = runtime.get("provider_mtp_draft_depth")
    draft_vocabulary = runtime.get("provider_mtp_draft_vocabulary_size")
    mtp_q8 = runtime.get("provider_mtp_q8_kv")
    valid = (
        abi == 2 and depth in (3, 4) and
        isinstance(draft_vocabulary, int) and
        not isinstance(draft_vocabulary, bool) and
        0 < draft_vocabulary <= 65_536 and mtp_q8 == 1 and
        execution.get("mtp_resource_available") is True and
        execution.get("mtp_runtime_ready") is True
    )
    if not valid:
        raise SystemExit(
            "running Q4H MTP contract is incomplete: "
            f"ABI={abi!r}, depth={depth!r}, "
            f"draft_vocabulary={draft_vocabulary!r}, Q8={mtp_q8!r}"
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
