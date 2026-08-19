param(
    [string]$Snapshot = "",
    [string]$Output = "",
    [ValidateSet("NO", "DELETE_CONSUMED_SHARDS")]
    [string]$SourceReclamationConfirmation = "NO",
    [switch]$Resume
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if (-not $Output) {
    if (-not $env:MODEL_ROOT) { throw "MODEL_ROOT is required" }
    $Output = Join-Path $env:MODEL_ROOT "qwen3-next-80b-expert-pack-fp4"
}
if (-not $Snapshot) {
    $Snapshot = Resolve-HuggingFaceSnapshot `
        -ModelId "Qwen/Qwen3-Next-80B-A3B-Instruct"
}

$snapshotPath = [System.IO.Path]::GetFullPath($Snapshot)
$outputPath = [System.IO.Path]::GetFullPath($Output)
$shards = @(Get-ChildItem $snapshotPath -Filter "model-*.safetensors" -File `
    -ErrorAction SilentlyContinue)
$incomplete = @(Get-ChildItem (Join-Path (Split-Path (Split-Path $snapshotPath)) "blobs") `
    -Filter "*.incomplete" -File -ErrorAction SilentlyContinue)
if ($shards.Count -ne 41 -or $incomplete.Count -ne 0) {
    throw "P6 checkpoint is incomplete: $($shards.Count)/41 shards, $($incomplete.Count) incomplete transfers"
}

$preflightText = & (Join-Path $PSScriptRoot "Invoke-P6Preflight.ps1") `
    -Snapshot $snapshotPath 6>&1 | Out-String
$preflightPath = Join-Path (Join-Path $script:RepoRoot "artifacts") "p6-preflight-latest.json"
$preflight = Get-Content $preflightPath -Raw | ConvertFrom-Json
$reclaim = $SourceReclamationConfirmation -eq "DELETE_CONSUMED_SHARDS"
if ($reclaim -and -not $preflight.can_convert_with_opt_in_reclaim) {
    throw "P6 preflight rejects conversion even with source reclamation"
}
if (-not $reclaim -and -not $preflight.can_convert_without_reclaim) {
    throw "Insufficient disk without reclamation. Re-run only after reviewing preflight and pass -SourceReclamationConfirmation DELETE_CONSUMED_SHARDS."
}

$arguments = @{
    Action = "Compile"
    Path = $snapshotPath
    Output = $outputPath
    SourceId = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    SourceRevision = Split-Path $snapshotPath -Leaf
    Adapter = "qwen3_next"
    QuantProfile = "fp4-e2m1-ue8m0-block32-v1"
    MaxExpertPackBytes = 4GB
    Resume = $Resume
    ReclaimSourceShards = $reclaim
}
Write-Output $preflightText.Trim()
& (Join-Path $PSScriptRoot "Invoke-ExpertPack.ps1") @arguments
