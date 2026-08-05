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
$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
$bundle = Join-Path (Join-Path $script:RepoRoot "work") `
    "deepseek-l${Layer}-e${Expert}-${stamp}.compact"
$executable = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-deepseek-admission-smoke.exe"
if (-not (Test-Path $executable -PathType Leaf)) {
    throw "Build the CUDA runtime before running DeepSeek admission"
}

Push-Location $script:RepoRoot
try {
    $qualificationRaw = & $python.Source -m compiler qualify-deepseek-expert `
        --source $source --layer $Layer --expert $Expert --row-chunk $RowChunk |
        Out-String
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek reference qualification failed" }
    $qualification = $qualificationRaw | ConvertFrom-Json

    & $python.Source -m compiler export-deepseek-expert --source $source `
        --output $bundle --layer $Layer --expert $Expert | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek compact fixture export failed" }
    $manifest = Get-Content (Join-Path $bundle "manifest.json") -Raw |
        ConvertFrom-Json

    $cudaRaw = & $executable $bundle `
        ([string]$qualification.result.candidate_sha256) `
        ([string]$manifest.combined.sha256) | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Output $cudaRaw
        throw "DeepSeek CUDA admission failed"
    }
    $cuda = $cudaRaw | ConvertFrom-Json
    $result = [PSCustomObject]@{
        schema_version = 1
        model_id = $ModelId
        revision = $Revision
        layer = $Layer
        expert = $Expert
        compact_bundle = $bundle
        reference = $qualification.result
        cuda = $cuda
    }
    $artifact = Write-JsonArtifact -Value $result `
        -Name "deepseek-cuda-admission-latest.json"
    $cudaRaw.Trim()
    Write-Output "Admission artifact: $artifact"
}
finally {
    Pop-Location
}
