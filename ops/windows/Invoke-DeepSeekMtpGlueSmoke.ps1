param(
    [Parameter(Mandatory = $true)][string]$Oracle,
    [Parameter(Mandatory = $true)][string]$MtpSet,
    [Parameter(Mandatory = $true)][string]$Snapshot
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$oracleRoot = [System.IO.Path]::GetFullPath($Oracle)
$mtpRoot = [System.IO.Path]::GetFullPath($MtpSet)
$sourceRoot = [System.IO.Path]::GetFullPath($Snapshot)
$executable = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-deepseek-mtp-glue-smoke.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "DeepSeek MTP glue smoke executable is missing; build the runtime first"
}

Push-Location $script:RepoRoot
try {
    & $executable $oracleRoot $mtpRoot $sourceRoot
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek MTP glue smoke failed" }
}
finally {
    Pop-Location
}
