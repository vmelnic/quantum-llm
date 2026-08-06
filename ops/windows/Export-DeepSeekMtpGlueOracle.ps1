param(
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [Parameter(Mandatory = $true)][string]$Output,
    [int]$Token = 42,
    [int]$Position = 1,
    [string]$PreviousStreams
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
    throw "DeepSeek MTP oracle output or partial already exists: $destination"
}

$arguments = @(
    "-m", "compiler", "export-deepseek-mtp-glue-oracle",
    "--source", $source, "--output", $destination,
    "--token", $Token, "--position", $Position
)
if ($PreviousStreams) {
    $arguments += @("--previous-streams", `
        [System.IO.Path]::GetFullPath($PreviousStreams))
}

Push-Location $script:RepoRoot
try {
    & $python.Source @arguments
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek MTP glue oracle failed" }
}
finally {
    Pop-Location
}

$result = [PSCustomObject]@{
    schema_version = 1
    snapshot = $source
    output = $destination
    token = $Token
    position = $Position
    status = "qualified"
}
$artifact = Write-JsonArtifact -Value $result `
    -Name "deepseek-mtp-glue-oracle-latest.json"
Write-Output "DeepSeek MTP glue oracle artifact: $artifact"
