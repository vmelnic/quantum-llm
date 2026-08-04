param(
    [string]$Snapshot = "C:\Users\vladi\.cache\huggingface\hub\models--Qwen--Qwen3-Next-80B-A3B-Instruct\snapshots\9c7f2fbe84465e40164a94cc16cd30b6999b0cc7",
    [string]$Output = "C:\Users\vladi\quantum-llm\work\models\qwen3-next-80b-expert-pack-int8",
    [ValidateSet("NO", "DELETE_CONSUMED_SHARDS")]
    [string]$SourceReclamationConfirmation = "NO",
    [switch]$Resume
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

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
    SourceRevision = "9c7f2fbe84465e40164a94cc16cd30b6999b0cc7"
    Adapter = "qwen3_next"
    MaxExpertPackBytes = 4GB
    Resume = $Resume
    ReclaimSourceShards = $reclaim
}
Write-Output $preflightText.Trim()
& (Join-Path $PSScriptRoot "Invoke-ExpertPack.ps1") @arguments
