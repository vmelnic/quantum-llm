$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../../../Run-Qwen27BSpeedRound.ps1') `
    -ArtifactName 'qwen3.8-27b-abliterated-fp4-activation-v3' `
    -KvDtype 'q4-f16-per-head' `
    -RoundId 'v3_q4h'
