param(
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [Parameter(Mandatory = $true)][string]$Output,
    [int[]]$Tokens = @(0, 128803, 23166, 19923),
    [int]$RowChunk = 512
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ($Tokens.Count -ne 4) {
    throw "DeepSeek MTP block oracle requires exactly four token IDs"
}
$source = [System.IO.Path]::GetFullPath($Snapshot)
$destination = [System.IO.Path]::GetFullPath($Output)
$python = Get-PythonCommand
if ((Test-Path -LiteralPath $destination) -or
    (Test-Path -LiteralPath ($destination + ".partial"))) {
    throw "DeepSeek MTP block oracle output or partial exists: $destination"
}

Push-Location $script:RepoRoot
try {
    & $python.Source -m compiler export-deepseek-mtp-block-oracle `
        --source $source --output $destination --tokens $Tokens `
        --row-chunk $RowChunk
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek MTP block oracle failed" }
}
finally {
    Pop-Location
}

$result = [PSCustomObject]@{
    schema_version = 1
    snapshot = $source
    output = $destination
    status = "qualified"
}
$artifact = Write-JsonArtifact -Value $result `
    -Name "deepseek-mtp-block-oracle-latest.json"
Write-Output "DeepSeek MTP block oracle artifact: $artifact"
