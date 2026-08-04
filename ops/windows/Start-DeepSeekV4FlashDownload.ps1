param(
    [int]$MaxWorkers = 4,
    [int64]$SafetyBytes = 8GB
)

& (Join-Path $PSScriptRoot "Start-HuggingFaceModelDownload.ps1") `
    -ModelId "deepseek-ai/DeepSeek-V4-Flash" `
    -Revision "60d8d70770c6776ff598c94bb586a859a38244f1" `
    -ExpectedDownloadBytes 159630041626 `
    -ExpectedTensorBytes 159609485896 `
    -ExpectedShards 46 `
    -MaxWorkers $MaxWorkers `
    -SafetyBytes $SafetyBytes
