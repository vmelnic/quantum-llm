param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [string]$Name = "layers.0.attn.wq_a",
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
$safeName = $Name -replace '[^A-Za-z0-9_.-]', '_'
$bundle = Join-Path (Join-Path $script:RepoRoot "work") `
    "deepseek-dense-${safeName}-${stamp}"
$executable = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-deepseek-dense-smoke.exe"
if (-not (Test-Path $executable -PathType Leaf)) {
    throw "Build the CUDA runtime before running dense admission"
}

Push-Location $script:RepoRoot
try {
    $qualificationRaw = & $python.Source -m compiler qualify-deepseek-fp8-matrix `
        --source $source --name $Name --row-chunk $RowChunk | Out-String
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek dense qualification failed" }
    $qualification = $qualificationRaw | ConvertFrom-Json
    & $python.Source -m compiler export-deepseek-fp8-matrix --source $source `
        --output $bundle --name $Name | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek dense extent export failed" }
    $manifest = Get-Content (Join-Path $bundle "manifest.json") -Raw |
        ConvertFrom-Json
    $rows = [string]$qualification.result.shape[0]
    $columns = [string]$qualification.result.shape[1]
    $nativeRaw = & $executable $bundle $source $rows $columns `
        ([string]$manifest.combined.sha256) `
        ([string]$qualification.result.candidate_sha256) | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Output $nativeRaw
        throw "DeepSeek dense CUDA admission failed"
    }
    $native = $nativeRaw | ConvertFrom-Json
    $result = [PSCustomObject]@{
        schema_version = 1
        model_id = $ModelId
        revision = $Revision
        name = $Name
        descriptor_bundle = $bundle
        reference = $qualification.result
        runtime = $native
    }
    $artifact = Write-JsonArtifact -Value $result `
        -Name "deepseek-dense-admission-latest.json"
    $nativeRaw.Trim()
    Write-Output "Dense admission artifact: $artifact"
}
finally {
    Pop-Location
}
