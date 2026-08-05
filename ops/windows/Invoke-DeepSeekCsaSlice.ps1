param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [int]$Layer = 2,
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
$bundle = Join-Path (Join-Path $script:RepoRoot "work") "deepseek-csa-$Layer-$stamp"
$executable = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-deepseek-csa-smoke.exe"
if (-not (Test-Path $executable -PathType Leaf)) {
    throw "Build the CUDA runtime before running the CSA slice"
}

Push-Location $script:RepoRoot
try {
    & $python.Source -m compiler export-deepseek-csa --source $source `
        --output $bundle --layer $Layer | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek CSA export failed" }
    $manifest = Get-Content (Join-Path $bundle "manifest.json") -Raw |
        ConvertFrom-Json
    $ratio = [string]$manifest.oracle.compress_ratio
    $nativeRaw = & $executable $bundle $source `
        ([string]$manifest.combined.sha256) ([string]$Layer) $ratio | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Output $nativeRaw
        throw "DeepSeek CSA CUDA slice failed"
    }
    $native = $nativeRaw | ConvertFrom-Json
    $result = [PSCustomObject]@{
        schema_version = 1
        model_id = $ModelId
        revision = $Revision
        layer = $Layer
        descriptor_bundle = $bundle
        reference = $manifest.oracle
        runtime = $native
    }
    $artifact = Write-JsonArtifact -Value $result -Name "deepseek-csa-slice-latest.json"
    $nativeRaw.Trim()
    Write-Output "CSA slice artifact: $artifact"
}
finally {
    Pop-Location
}
