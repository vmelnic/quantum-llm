param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
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
$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
$bundle = Join-Path (Join-Path $script:RepoRoot "work") `
    "deepseek-shared-set-${stamp}"
$executable = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-deepseek-shared-residency.exe"
if (-not (Test-Path $executable -PathType Leaf)) {
    throw "Build the CUDA runtime before running shared residency"
}

Push-Location $script:RepoRoot
try {
    & $python.Source -m compiler export-deepseek-shared-set --source $source `
        --output $bundle | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek shared-set export failed" }
    $nativeRaw = & $executable $bundle $source | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Output $nativeRaw
        throw "DeepSeek shared residency failed"
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
        -Name "deepseek-shared-residency-latest.json"
    $nativeRaw.Trim()
    Write-Output "Shared residency artifact: $artifact"
}
finally {
    Pop-Location
}
