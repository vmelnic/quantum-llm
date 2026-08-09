# AGENTS.md

## User preferences (durable)

- No effort estimates. The user is not interested in "engineering days",
  story points, or stage durations. Do not estimate; plan work as a flat
  list of items that all need to be done, ordered by dependency, and then
  implement and test them.
- Do not split work into endless small patches ("peticiri"); implement the
  real change, then verify it.
- Communicate in Romanian; repository artifacts (code comments, commit
  messages, docs) stay in English.
- Hugging Face downloads only via xet through the repo scripts
  (`ops/windows/Start-HuggingFaceModelDownload.ps1` /
  `Get-HuggingFaceModelDownload.ps1`). Never curl/parallel ad-hoc
  downloads, never invented scheduled tasks, and do not set
  `HF_HUB_DISABLE_XET`.

## Project pointers

- `docs/inference-30toks-plan.md` — the serving-speed work plan for the
  3090 host (Qwen3-Next-80B / DeepSeek-V4-Flash).
- `docs/memory-kv-attach-v1.md` — the KV-attach memory architecture
  (milestones 1–2 validated); `docs/memory-expert.md` — the closed
  rank-16 gate direction (v4/v5).
