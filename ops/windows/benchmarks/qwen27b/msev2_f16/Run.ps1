$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../../../Run-Qwen27BSpeedRound.ps1') `
    -ArtifactName 'qwen3.8-27b-abliterated-fp4-mse-v2' `
    -KvDtype 'fp16' `
    -RoundId 'msev2_f16'
