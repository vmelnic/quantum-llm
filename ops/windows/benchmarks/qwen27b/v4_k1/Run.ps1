$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../../../Run-Qwen27BSpeedRound.ps1') `
    -ArtifactName 'qwen3.8-27b-abliterated-fp4-activation-v4' `
    -KvDtype 'fp4-e2m1-ue8m0-block32-key-outlier1' `
    -RoundId 'v4_k1'
