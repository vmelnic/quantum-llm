$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../../../Run-Qwen27BSpeedRound.ps1') `
    -ArtifactName 'qwen3.8-27b-abliterated-fp4' `
    -KvDtype 'q4-f16-per-head' `
    -RoundId 'fp4_q4h'
