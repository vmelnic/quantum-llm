#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "Usage: Cache-PublishedModel.sh <published-artifact> [cache-root]" >&2
  exit 2
fi

source_root="$(realpath -e -- "$1")"
cache_root="${2:-${MODEL_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/quantum-llm/models}}"
manifest_path="${source_root}/manifest.json"

[[ -f "${manifest_path}" ]] || {
  echo "Published artifact manifest is missing: ${manifest_path}" >&2
  exit 2
}

records="$({
  python3 - "${manifest_path}" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("format") != "huggingface-exact-files-v1":
    raise SystemExit("unsupported published artifact manifest")
files = manifest.get("files")
if not isinstance(files, list) or not files:
    raise SystemExit("published artifact manifest has no files")
payload_bytes = 0
for record in files:
    path = record.get("path")
    size = record.get("bytes")
    digest = record.get("sha256")
    if not isinstance(path, str) or "\t" in path or "\n" in path:
        raise SystemExit("manifest contains an invalid file path")
    relative = pathlib.PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SystemExit(f"manifest file escapes the artifact: {path}")
    if not isinstance(size, int) or size < 0:
        raise SystemExit(f"manifest has invalid byte size: {path}")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest.lower()
    ):
        raise SystemExit(f"manifest has invalid SHA-256: {path}")
    payload_bytes += size
    print(f"{path}\t{size}\t{digest.lower()}")
if payload_bytes != manifest.get("payload_bytes"):
    raise SystemExit("manifest payload byte total does not match its file records")
PY
} )"

validate_payload() {
  local root="$1" path expected_bytes expected_hash actual_bytes actual_hash
  while IFS=$'\t' read -r path expected_bytes expected_hash; do
    [[ -n "${path}" ]] || continue
    [[ -f "${root}/${path}" ]] || {
      echo "Artifact file is missing: ${root}/${path}" >&2
      return 1
    }
    actual_bytes="$(stat -c %s -- "${root}/${path}")"
    [[ "${actual_bytes}" == "${expected_bytes}" ]] || {
      echo "Artifact file size mismatch: ${root}/${path}" >&2
      return 1
    }
    actual_hash="$(sha256sum -- "${root}/${path}")"
    actual_hash="${actual_hash%% *}"
    [[ "${actual_hash}" == "${expected_hash}" ]] || {
      echo "Artifact file hash mismatch: ${root}/${path}" >&2
      return 1
    }
  done <<< "${records}"
}

validate_source_layout() {
  local path expected_bytes expected_hash actual_bytes
  while IFS=$'\t' read -r path expected_bytes expected_hash; do
    [[ -n "${path}" ]] || continue
    [[ -f "${source_root}/${path}" ]] || {
      echo "Published file is missing: ${source_root}/${path}" >&2
      return 1
    }
    actual_bytes="$(stat -c %s -- "${source_root}/${path}")"
    [[ "${actual_bytes}" == "${expected_bytes}" ]] || {
      echo "Published file size mismatch: ${source_root}/${path}" >&2
      return 1
    }
  done <<< "${records}"
}

validate_source_layout

stable_name="$(basename -- "${source_root}")"
manifest_hash="$(sha256sum -- "${manifest_path}")"
manifest_hash="${manifest_hash%% *}"
candidate="${cache_root}/${stable_name}.candidate-${manifest_hash:0:12}"
destination="${cache_root}/${stable_name}"

mkdir -p -- "${cache_root}"
[[ ! -e "${destination}" ]] || {
  echo "Validated cache destination already exists; refusing to overwrite: ${destination}" >&2
  exit 2
}
[[ ! -e "${candidate}" ]] || {
  echo "Cache candidate already exists; inspect it before retrying: ${candidate}" >&2
  exit 2
}

mkdir -- "${candidate}"
while IFS=$'\t' read -r path expected_bytes expected_hash; do
  [[ -n "${path}" ]] || continue
  mkdir -p -- "$(dirname -- "${candidate}/${path}")"
  cp --reflink=auto --sparse=always -- "${source_root}/${path}" \
    "${candidate}/${path}"
done <<< "${records}"
cp -- "${manifest_path}" "${candidate}/manifest.json"

validate_payload "${candidate}"
mv -- "${candidate}" "${destination}"
if ! validate_payload "${destination}"; then
  mv -- "${destination}" "${candidate}"
  exit 1
fi

printf '%s\n' "${destination}"
