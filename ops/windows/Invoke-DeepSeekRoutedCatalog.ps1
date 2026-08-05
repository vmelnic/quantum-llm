param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [string]$Snapshot,
    [string]$Output
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$source = if ($Snapshot) {
    [System.IO.Path]::GetFullPath($Snapshot)
} else {
    Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
}
$catalog = if ($Output) {
    [System.IO.Path]::GetFullPath($Output)
} else {
    Join-Path (Join-Path $script:RepoRoot "work") "deepseek-routed-catalog"
}
$python = Get-PythonCommand
$executable = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-deepseek-catalog-smoke.exe"
if (-not (Test-Path $executable -PathType Leaf)) {
    throw "Build the CUDA runtime before generating the routed catalog"
}

Push-Location $script:RepoRoot
try {
    & $python.Source -m compiler export-deepseek-routed-catalog `
        --source $source --output $catalog
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek routed catalog export failed" }
    $nativeRaw = & $executable $catalog $source | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Output $nativeRaw
        throw "DeepSeek routed catalog runtime validation failed"
    }
    $native = $nativeRaw | ConvertFrom-Json
    $result = [PSCustomObject]@{
        schema_version = 1
        model_id = $ModelId
        revision = $Revision
        catalog = $catalog
        runtime = $native
    }
    $artifact = Write-JsonArtifact -Value $result `
        -Name "deepseek-routed-catalog-latest.json"
    $nativeRaw.Trim()
    Write-Output "Routed catalog artifact: $artifact"
}
finally {
    Pop-Location
}
