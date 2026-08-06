param(
    [Parameter(Mandatory = $true)][string]$Oracle,
    [Parameter(Mandatory = $true)][string]$TargetBundle,
    [Parameter(Mandatory = $true)][string]$MtpSet,
    [Parameter(Mandatory = $true)][string]$MtpPack,
    [Parameter(Mandatory = $true)][string]$Snapshot
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$binary = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-deepseek-mtp-block-smoke.exe"
if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
    throw "DeepSeek MTP block smoke binary is missing; build the runtime first"
}

& $binary `
    ([System.IO.Path]::GetFullPath($Oracle)) `
    ([System.IO.Path]::GetFullPath($TargetBundle)) `
    ([System.IO.Path]::GetFullPath($MtpSet)) `
    ([System.IO.Path]::GetFullPath($MtpPack)) `
    ([System.IO.Path]::GetFullPath($Snapshot))
if ($LASTEXITCODE -ne 0) { throw "DeepSeek MTP block smoke failed" }
