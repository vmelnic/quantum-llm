param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [int]$Layer = 0,
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
$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
$bundle = Join-Path (Join-Path $script:RepoRoot "work") `
    "deepseek-l${Layer}-shared-${stamp}.compact"
$executable = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-deepseek-admission-smoke.exe"
if (-not (Test-Path $executable -PathType Leaf)) {
    throw "Build the CUDA runtime before running shared-expert admission"
}

Push-Location $script:RepoRoot
try {
    $qualificationRaw = & $python.Source -m compiler qualify-deepseek-shared `
        --source $source --layer $Layer --row-chunk $RowChunk | Out-String
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek shared reference qualification failed" }
    $qualification = $qualificationRaw | ConvertFrom-Json

    & $python.Source -m compiler export-deepseek-shared --source $source `
        --output $bundle --layer $Layer | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek shared extent export failed" }
    $manifest = Get-Content (Join-Path $bundle "manifest.json") -Raw |
        ConvertFrom-Json

    $cudaRaw = & $executable $bundle $source `
        ([string]$qualification.result.candidate_sha256) `
        ([string]$manifest.combined.sha256) shared | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Output $cudaRaw
        throw "DeepSeek shared CUDA admission failed"
    }
    $cuda = $cudaRaw | ConvertFrom-Json
    $result = [PSCustomObject]@{
        schema_version = 1
        model_id = $ModelId
        revision = $Revision
        layer = $Layer
        compact_bundle = $bundle
        reference = $qualification.result
        cuda = $cuda
    }
    $artifact = Write-JsonArtifact -Value $result `
        -Name "deepseek-shared-admission-latest.json"
    $cudaRaw.Trim()
    Write-Output "Shared admission artifact: $artifact"
}
finally {
    Pop-Location
}
