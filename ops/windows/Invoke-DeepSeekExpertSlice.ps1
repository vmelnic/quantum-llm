param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [int]$Layer = 0,
    [int]$Expert = 0,
    [int]$RowChunk = 128,
    [string]$Snapshot
)

. (Join-Path $PSScriptRoot "Common.ps1")

Initialize-ExperimentDirectories
$source = if ($Snapshot) {
    [System.IO.Path]::GetFullPath($Snapshot)
} else {
    Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
}
$python = Get-PythonCommand

Push-Location $script:RepoRoot
try {
    $raw = & $python.Source -m compiler qualify-deepseek-expert `
        --source $source --layer $Layer --expert $Expert --row-chunk $RowChunk |
        Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Output $raw
        throw "DeepSeek expert qualification exited with code $LASTEXITCODE"
    }
    $parsed = $raw | ConvertFrom-Json
    $artifact = Write-JsonArtifact -Value $parsed.result `
        -Name "deepseek-expert-slice-latest.json"
    Write-Output $raw.Trim()
    Write-Output "Qualification artifact: $artifact"
}
finally {
    Pop-Location
}
