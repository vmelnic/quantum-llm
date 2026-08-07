#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
destination="${MEMORY_HF_DATASET_ROOT:-${repo_root}/work/memory-capability/hf-datasets}"

export HF_XET_HIGH_PERFORMANCE=1

download_dataset() {
  local repo_id="$1"
  local revision="$2"
  local include="$3"
  local local_name="$4"
  uvx --from 'huggingface_hub[hf_xet]' hf download "${repo_id}" \
    --repo-type dataset \
    --revision "${revision}" \
    --include "${include}" \
    --local-dir "${destination}/${local_name}" \
    --max-workers "${MEMORY_HF_MAX_WORKERS:-8}"
}

mkdir -p "${destination}"
download_dataset \
  "osunlp/ConflictQA" \
  "056384049e63c1ddae853891c24610fa07d85744" \
  "conflictQA-popQA-chatgpt.json" \
  "conflictqa"
download_dataset \
  "Salesforce/FaithEval-counterfactual-v1.0" \
  "e655f7c8750aabe431eeeda63fa5c019a2567b18" \
  "data/*.parquet" \
  "faitheval-counterfactual"
download_dataset \
  "Salesforce/FaithEval-unanswerable-v1.0" \
  "4a14e0e9860e270a8c9ff57c511821f4f50dce57" \
  "data/*.parquet" \
  "faitheval-unanswerable"
download_dataset \
  "gaotang/ParaConfilct" \
  "9929c64cd89838d1c00459e8ff94859de7b06827" \
  "data/*.parquet" \
  "paraconflict"

echo "Pinned Memory Expert datasets downloaded under ${destination}"
