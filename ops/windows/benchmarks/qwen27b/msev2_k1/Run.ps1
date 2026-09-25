$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../../../Run-Qwen27BSpeedRound.ps1') `
    -ArtifactName 'qwen3.8-27b-abliterated-fp4-mse-v2' `
    -KvDtype 'fp4-e2m1-ue8m0-block32-key-outlier1' `
    -RoundId 'msev2_k1'
