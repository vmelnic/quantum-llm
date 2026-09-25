$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../../../Run-Qwen27BSpeedRound.ps1') `
    -ArtifactName 'qwen3.8-27b-abliterated-fp4-activation-v3' `
    -KvDtype 'fp16' `
    -RoundId 'v3_f16'
