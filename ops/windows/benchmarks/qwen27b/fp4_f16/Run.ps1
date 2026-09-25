$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../../../Run-Qwen27BSpeedRound.ps1') `
    -ArtifactName 'qwen3.8-27b-abliterated-fp4' `
    -KvDtype 'fp16' `
    -RoundId 'fp4_f16'
