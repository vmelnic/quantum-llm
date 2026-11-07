#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
if [[ -f "${repo_root}/.env" ]]; then
  set -a
  source "${repo_root}/.env"
  set +a
fi

die() { echo "llama.sh: $*" >&2; exit 2; }
usage() {
  cat <<'EOF'
Usage: ./ops/llama.sh <update|publish|deploy|start|restart|stop|status|pi|chat|config> [swift|swith|swift-heretic|kat|gpt-oss|arex] [Pi arguments...]

Runs the selected published GGUF on the Windows RTX 3090 with llama.cpp.
One model is active at a time. QPack operations remain in ops/model.sh.
`deploy` prepares the next launch without stopping or restarting the server.
`pi` without a selector uses the currently running llama.cpp model.
`update` installs the latest stable CUDA build without restarting the server.
EOF
}
action="${1:-status}"
selection="${2:-}"
case "${action}" in -h|--help|help) usage; exit ;; esac
command -v python3 >/dev/null || die 'python3 is required'
port="${LLAMA_PORT:-8081}"
provider="llama-3090"
[[ "${port}" =~ ^[1-9][0-9]*$ ]] && (( port <= 65535 )) || die 'LLAMA_PORT must be a valid TCP port'
[[ -n "${QUANTUM_LLM_REMOTE:-}" && -n "${QUANTUM_LLM_REMOTE_ROOT:-}" && -n "${MODEL_ROOT:-}" ]] ||
  die 'QUANTUM_LLM_REMOTE, QUANTUM_LLM_REMOTE_ROOT and MODEL_ROOT are required'
model_id=""
if [[ -n "${selection}" ]]; then
  model_id="$(python3 - "${script_dir}/llama-models.json" "${selection}" <<'PY'
import json, sys
registry = json.load(open(sys.argv[1], encoding="utf-8"))
if registry.get("schema") != "llama-models-v1":
    raise SystemExit("invalid llama model registry")
matches = [m for m in registry["models"] if sys.argv[2] in m["aliases"]]
if len(matches) != 1:
    raise SystemExit(f"unknown llama model: {sys.argv[2]}")
print(matches[0]["model_id"])
PY
)"
fi
remote() { "${script_dir}/run-on-windows-host.sh" "$@"; }
auth_args=()
if [[ -n "${EXPERT_API_KEY:-}" ]]; then auth_args+=(-ApiKey "${EXPERT_API_KEY}"); fi
server_args=(-Port "${port}" "${auth_args[@]}")

case "${action}" in
  update)
    [[ -z "${selection}" ]] || die 'update does not take a model selector'
    scp -q "${script_dir}/windows/Update-LlamaCpp.ps1" "${script_dir}/windows/Invoke-LlamaServer.ps1" \
      "${QUANTUM_LLM_REMOTE}:${QUANTUM_LLM_REMOTE_ROOT}/ops/windows/"
    remote Update-LlamaCpp.ps1
    ;;
  publish|deploy|start|restart)
    [[ -n "${selection}" ]] || die "${action} requires a model selector (swift, swift-heretic, kat, gpt-oss or arex)"
    "${script_dir}/sync-to-windows-host.sh" "${QUANTUM_LLM_REMOTE}" "${QUANTUM_LLM_REMOTE_ROOT}"
    remote Publish-LlamaModel.ps1 -Model "${selection}"
    if [[ "${action}" != publish ]]; then
      launch_action=start
      if [[ "${action}" == deploy ]]; then launch_action=prepare; fi
      start_args=(-Action "${launch_action}" -Model "${selection}" -HostAddress "${LLAMA_HOST:-0.0.0.0}"
        -WaitSeconds "${MODEL_READY_TIMEOUT:-600}" -TimeoutSeconds "${MODEL_GENERATION_TIMEOUT_SECONDS:-14400}")
      if [[ -n "${LLAMA_BINARY:-}" ]]; then start_args+=(-Binary "${LLAMA_BINARY}"); fi
      remote Invoke-LlamaServer.ps1 "${start_args[@]}" "${server_args[@]}"
    fi
    ;;
  stop|status)
    remote Invoke-LlamaServer.ps1 -Action "${action}" "${server_args[@]}"
    ;;
  pi|chat)
    command -v pi >/dev/null || die 'pi is not installed'
    models_file="${PI_MODELS_FILE:-${HOME}/.pi/agent/models.json}"
    model_id="$(python3 - "${models_file}" "${provider}" "${model_id}" "${script_dir}/llama-models.json" <<'PY'
import json, sys, urllib.request
path, provider_name, expected, policy_path = sys.argv[1:]
registry = json.load(open(path, encoding="utf-8"))
provider = registry["providers"][provider_name]
base = provider["baseUrl"].removesuffix("/v1").rstrip("/")
key = provider.get("apiKey", "")
headers = {"Authorization": f"Bearer {key}"} if key else {}
def get(path):
    with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers), timeout=10) as r:
        return json.load(r)
try:
    models, props = get("/v1/models"), get("/props")
except OSError as error:
    raise SystemExit(f"cannot reach llama.cpp; run ./ops/llama.sh start <model>: {error}")
actual = models["data"][0]["id"]
if expected and actual != expected:
    raise SystemExit(f"running model is {actual!r}, expected {expected!r}; start the selected model first")
configured = next((m for m in provider["models"] if m["id"] == actual), None)
if configured is None:
    raise SystemExit(f"Pi does not know running llama.cpp model {actual!r}")
policy_registry = json.load(open(policy_path, encoding="utf-8"))
if policy_registry.get("schema") != "llama-models-v1":
    raise SystemExit("invalid llama model registry")
matches = [m for m in policy_registry["models"] if m["model_id"] == actual]
if len(matches) != 1:
    raise SystemExit(f"missing or ambiguous launch policy for {actual!r}")
policy = matches[0]
slots = policy.get("parallel", 1)
if type(slots) is not int or not 1 <= slots <= 2 or policy["context"] % slots:
    raise SystemExit("invalid llama.cpp context/slot policy")
context_per_slot = policy["context"] // slots
if (props["default_generation_settings"]["n_ctx"] != context_per_slot
        or configured["contextWindow"] != context_per_slot
        or props["total_slots"] != slots):
    raise SystemExit("Pi and llama.cpp context/slot settings differ")
print(actual)
PY
)"
    if (( $# >= 2 )); then shift 2; else shift; fi
    exec pi --approve --provider "${provider}" --model "${model_id}" --thinking xhigh "$@"
    ;;
  config)
    policy_fields="$(python3 - "${script_dir}/llama-models.json" "${model_id}" <<'PY'
import json, sys
registry = json.load(open(sys.argv[1], encoding="utf-8"))
model = next((m for m in registry["models"] if m["model_id"] == sys.argv[2]), None)
if model is None:
    print("context=<model policy>\nkv=<model policy>\nparallel=<model policy>")
else:
    print(f"context={model['context']}\nkv={model['cache_type_k']}/{model['cache_type_v']}\nparallel={model.get('parallel', 1)}")
PY
)"
    printf '%s\n' "backend=llama.cpp" "model=${model_id:-<running>}" "port=${port}" \
      "provider=${provider}" "model_root=${MODEL_ROOT}" "${policy_fields}" "gpu=RTX3090:0"
    ;;
  *) usage >&2; die "unknown action '${action}'" ;;
esac
