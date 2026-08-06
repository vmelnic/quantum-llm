param(
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [Parameter(Mandatory = $true)][string]$Output
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$source = [System.IO.Path]::GetFullPath($Snapshot)
$destination = [System.IO.Path]::GetFullPath($Output)
$python = Get-PythonCommand
if (-not (Test-Path -LiteralPath (Join-Path $source `
        "model.safetensors.index.json") -PathType Leaf)) {
    throw "DeepSeek snapshot index is missing"
}
if ((Test-Path -LiteralPath $destination) -or
    (Test-Path -LiteralPath ($destination + ".partial"))) {
    throw "DeepSeek MTP output or partial already exists: $destination"
}

Push-Location $script:RepoRoot
try {
    & $python.Source -m compiler export-deepseek-mtp-set `
        --source $source --output $destination
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek MTP export failed" }
}
finally {
    Pop-Location
}

$result = [PSCustomObject]@{
    schema_version = 1
    snapshot = $source
    output = $destination
    status = "authenticated"
}
$artifact = Write-JsonArtifact -Value $result `
    -Name "deepseek-mtp-set-latest.json"
Write-Output "DeepSeek MTP set artifact: $artifact"
