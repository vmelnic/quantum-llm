param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [int]$Layer = 0,
    [ValidateSet("attn", "ffn")][string]$Site = "attn",
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
    "deepseek-hca-$Layer-$Site-$stamp"
$executable = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-deepseek-hca-smoke.exe"
if (-not (Test-Path $executable -PathType Leaf)) {
    throw "Build the CUDA runtime before running the HCA slice"
}

Push-Location $script:RepoRoot
try {
    $exportRaw = & $python.Source -m compiler export-deepseek-hca `
        --source $source --output $bundle --layer $Layer --site $Site | Out-String
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek HCA export failed: $exportRaw" }
    $manifest = Get-Content (Join-Path $bundle "manifest.json") -Raw |
        ConvertFrom-Json
    $nativeRaw = & $executable $bundle $source `
        ([string]$manifest.combined.sha256) | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Output $nativeRaw
        throw "DeepSeek HCA CUDA slice failed"
    }
    $native = $nativeRaw | ConvertFrom-Json
    $result = [PSCustomObject]@{
        schema_version = 1
        model_id = $ModelId
        revision = $Revision
        layer = $Layer
        site = $Site
        descriptor_bundle = $bundle
        reference = $manifest.oracle
        runtime = $native
    }
    $artifact = Write-JsonArtifact -Value $result `
        -Name "deepseek-hca-slice-latest.json"
    $nativeRaw.Trim()
    Write-Output "HCA slice artifact: $artifact"
}
finally {
    Pop-Location
}
