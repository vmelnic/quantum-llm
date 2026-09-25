$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../../../Run-Qwen27BSpeedRound.ps1') `
    -ArtifactName 'qwen3.8-27b-abliterated-fp4-mse-v2' `
    -KvDtype 'q4-f16-per-head' `
    -RoundId 'msev2_q4h'
