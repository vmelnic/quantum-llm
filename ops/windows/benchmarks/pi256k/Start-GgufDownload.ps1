param([Parameter(Mandatory = $true)][string]$ModelRoot)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
$root = [System.IO.Path]::GetFullPath($ModelRoot)
if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw 'MODEL_ROOT is missing' }
$hfHome = Join-Path $root '.hf-cache'
[System.IO.Directory]::CreateDirectory($hfHome) | Out-Null
& (Join-Path $repo 'ops/windows/Start-HuggingFaceModelDownload.ps1') `
    -ModelId 'ggml-org/Qwen3.8-27B-GGUF' `
    -Revision 'efbb3b1f70a21d97fd4495240648405f7228554f' `
    -Files 'Qwen3.8-27B-Q4_K_M.gguf,mtp-Qwen3.8-27B-Q4_0.gguf' `
    -ExpectedShards 2 -ExpectedDownloadBytes 20654142304 `
    -ExpectedTensorBytes 20654142304 -HfHome $hfHome
