param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [string]$Snapshot,
    [ValidateSet(0, 2)][int]$OracleLayer = 2
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories
$source = if ($Snapshot) {
    [System.IO.Path]::GetFullPath($Snapshot)
} else {
    Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
}
$python = Get-PythonCommand
$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
$bundle = Join-Path (Join-Path $script:RepoRoot "work") "deepseek-model-$stamp"
$dense = Join-Path $bundle "dense"
$typed = Join-Path $bundle "typed"
$oracle = Join-Path $bundle "attention-oracle"
$executable = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-deepseek-model-residency.exe"
if (-not (Test-Path $executable -PathType Leaf)) {
    throw "Build the CUDA runtime before running model residency"
}

Push-Location $script:RepoRoot
try {
    & $python.Source -m compiler export-deepseek-dense-set --source $source `
        --output $dense | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek dense-set export failed" }
    & $python.Source -m compiler export-deepseek-typed-set --source $source `
        --output $typed | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek typed-set export failed" }
    & $python.Source -m compiler export-deepseek-attention-oracle --source $source `
        --output $oracle --layer $OracleLayer | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek attention oracle export failed" }
    $nativeRaw = & $executable $dense $typed $source $oracle | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Output $nativeRaw
        throw "DeepSeek model residency failed"
    }
    $native = $nativeRaw | ConvertFrom-Json
    $result = [PSCustomObject]@{
        schema_version = 1
        model_id = $ModelId
        revision = $Revision
        descriptor_bundle = $bundle
        runtime = $native
    }
    $artifact = Write-JsonArtifact -Value $result `
        -Name "deepseek-model-residency-latest.json"
    $nativeRaw.Trim()
    Write-Output "Model residency artifact: $artifact"
}
finally {
    Pop-Location
}
