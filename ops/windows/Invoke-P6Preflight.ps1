param(
    [string]$Snapshot = "C:\Users\vladi\.cache\huggingface\hub\models--Qwen--Qwen3-Next-80B-A3B-Instruct\snapshots\9c7f2fbe84465e40164a94cc16cd30b6999b0cc7",
    [int64]$SafetyBytes = 8GB,
    [int64]$MaximumExpertPackBytes = 4GB
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$snapshotPath = [System.IO.Path]::GetFullPath($Snapshot)
$indexPath = Join-Path $snapshotPath "model.safetensors.index.json"
if (-not (Test-Path $indexPath -PathType Leaf)) {
    throw "P6 snapshot has no model.safetensors.index.json: $snapshotPath"
}
$index = Get-Content $indexPath -Raw | ConvertFrom-Json
$sourceBytes = [int64]$index.metadata.total_size
if ($sourceBytes -le 0) { throw "SafeTensors index has no positive metadata.total_size" }

# Exact Expert Pack v1 estimate for the pinned Qwen3-Next 80B geometry,
# including the optional 1-layer MTP head retained in dense.qpack.
$estimatedDenseBytes = [int64]4036993024
$estimatedExpertBytes = [int64]77712064512
$estimatedContainerBytes = $estimatedDenseBytes + $estimatedExpertBytes
$shards = @(Get-ChildItem $snapshotPath -Filter "model-*.safetensors" -File `
    -ErrorAction SilentlyContinue)
$completedSourceBytes = [int64]0
$completedShardBytes = @{}
foreach ($shard in $shards) {
    $length = [int64]$shard.Length
    if (($shard.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        $target = [string]$shard.Target
        if (-not [System.IO.Path]::IsPathRooted($target)) {
            $target = Join-Path $shard.DirectoryName $target
        }
        $length = [int64](Get-Item -LiteralPath $target -ErrorAction Stop).Length
    }
    $completedSourceBytes += $length
    $completedShardBytes[$shard.Name] = $length
}
$remainingDownloadBytes = [Math]::Max([int64]0, $sourceBytes - $completedSourceBytes)
$drive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($snapshotPath).Substring(0, 1))
$freeNow = [int64]$drive.Free
$freeAfterDownload = $freeNow - $remainingDownloadBytes
$physicalRam = [int64](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$firstShard = $shards | Select-Object -First 1
$firstShardIsReparsePoint = $false
if ($null -ne $firstShard) {
    $firstShardIsReparsePoint = ($firstShard.Attributes -band `
        [System.IO.FileAttributes]::ReparsePoint) -ne 0
}

$requiredWithoutReclaim = $estimatedContainerBytes + $SafetyBytes
$requiredWithReclaim = $estimatedDenseBytes + $MaximumExpertPackBytes + $SafetyBytes
$reclaimScheduleVerified = $shards.Count -eq 41
$projectedMinimumFree = $null
$reclaimEvents = @()
if ($reclaimScheduleVerified) {
    # The compiler writes all dense tensors first, then main routed experts in
    # deterministic (layer, expert) order. MTP experts are dense records and do
    # not keep their source shard live after dense.qpack commits.
    $lastExpertByShard = @{}
    foreach ($property in $index.weight_map.PSObject.Properties) {
        $shardName = [string]$property.Value
        if (-not $lastExpertByShard.ContainsKey($shardName)) {
            $lastExpertByShard[$shardName] = [int64]-1
        }
        if ($property.Name -match '^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$') {
            $ordinal = [int64]$Matches[1] * 512 + [int64]$Matches[2]
            $lastExpertByShard[$shardName] = [Math]::Max(
                [int64]$lastExpertByShard[$shardName], $ordinal)
        }
    }
    $expertRecordBytes = [int64]3162112
    $totalExperts = [int64](48 * 512)
    $recordsPerPack = [int64][Math]::Floor(
        [double]$MaximumExpertPackBytes / $expertRecordBytes)
    if ($recordsPerPack -lt 1) { throw "MaximumExpertPackBytes fits no Qwen expert" }
    $free = [int64]($freeAfterDownload - $estimatedDenseBytes)
    $projectedMinimumFree = $free
    $reclaimed = @{}
    $nextExpert = [int64]0
    while ($true) {
        $reclaimedBytes = [int64]0
        $reclaimedNames = @()
        foreach ($entry in $lastExpertByShard.GetEnumerator()) {
            if (-not $reclaimed.ContainsKey($entry.Key) -and
                [int64]$entry.Value -lt $nextExpert) {
                $bytes = [int64]$completedShardBytes[$entry.Key]
                $free += $bytes
                $reclaimedBytes += $bytes
                $reclaimedNames += [string]$entry.Key
                $reclaimed[$entry.Key] = $true
            }
        }
        if ($reclaimedNames.Count -gt 0) {
            $reclaimEvents += [PSCustomObject]@{
                after_experts = $nextExpert
                reclaimed_shards = @($reclaimedNames | Sort-Object)
                reclaimed_bytes = $reclaimedBytes
                projected_free_bytes = $free
            }
        }
        if ($nextExpert -ge $totalExperts) { break }
        $records = [Math]::Min($recordsPerPack, $totalExperts - $nextExpert)
        $free -= [int64]$records * $expertRecordBytes
        $projectedMinimumFree = [Math]::Min([int64]$projectedMinimumFree, $free)
        $nextExpert += $records
    }
}
$canConvertWithReclaim = if ($reclaimScheduleVerified) {
    [int64]$projectedMinimumFree -ge $SafetyBytes
} else {
    $freeAfterDownload -ge $requiredWithReclaim
}
$result = [PSCustomObject]@{
    schema_version = 1
    model_id = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    snapshot = $snapshotPath
    complete_shards = $shards.Count
    expected_shards = 41
    source_tensor_bytes = $sourceBytes
    completed_source_bytes = $completedSourceBytes
    remaining_download_bytes = $remainingDownloadBytes
    estimated_dense_pack_bytes = $estimatedDenseBytes
    estimated_expert_pack_bytes = $estimatedExpertBytes
    estimated_container_bytes = $estimatedContainerBytes
    physical_ram_bytes = $physicalRam
    container_larger_than_physical_ram = $estimatedContainerBytes -gt $physicalRam
    free_bytes_now = $freeNow
    projected_free_bytes_after_download = $freeAfterDownload
    required_free_without_reclaim = $requiredWithoutReclaim
    required_free_with_reclaim = $requiredWithReclaim
    can_finish_download = $freeNow -ge ($remainingDownloadBytes + $SafetyBytes)
    can_convert_without_reclaim = $freeAfterDownload -ge $requiredWithoutReclaim
    can_convert_with_opt_in_reclaim = $canConvertWithReclaim
    reclaim_schedule_verified = $reclaimScheduleVerified
    projected_minimum_free_during_conversion = $projectedMinimumFree
    reclaim_schedule = $reclaimEvents
    first_complete_shard_is_reparse_point = $firstShardIsReparsePoint
    reclaim_is_destructive_and_requires_explicit_switch = $true
}
$path = Write-JsonArtifact -Value $result -Name "p6-preflight-latest.json"
$result | ConvertTo-Json -Depth 4
Write-Output "P6 preflight artifact: $path"
