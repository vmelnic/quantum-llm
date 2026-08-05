. (Join-Path $PSScriptRoot "Common.ps1")

$artifact = Join-Path (Join-Path $script:RepoRoot "artifacts") `
    "hf-download-process.json"
if (-not (Test-Path $artifact -PathType Leaf)) {
    throw "No Hugging Face download artifact"
}
$state = Get-Content $artifact -Raw | ConvertFrom-Json
$escapedModel = [WildcardPattern]::Escape([string]$state.model_id)
$worker = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -in @("python.exe", "hf.exe", "powershell.exe") -and
        ($_.CommandLine -like "*download*$escapedModel*" -or
         $_.CommandLine -like "*$escapedModel*--max-workers*")
    } | Select-Object -First 1
$task = Get-ScheduledTask -TaskName ([string]$state.task_name) `
    -ErrorAction SilentlyContinue
$taskRunning = $null -ne $task -and $task.State -eq "Running"

$cacheName = "models--" + ($state.model_id -replace "/", "--")
$root = Join-Path (Join-Path $env:USERPROFILE ".cache\huggingface\hub") $cacheName
$blobBytes = [int64]0
foreach ($blob in @(Get-ChildItem (Join-Path $root "blobs") -File `
        -ErrorAction SilentlyContinue)) {
    $blobBytes += [int64]$blob.Length
}
$incomplete = @(Get-ChildItem (Join-Path $root "blobs") -Filter "*.incomplete" `
    -File -ErrorAction SilentlyContinue)
$snapshot = Join-Path (Join-Path $root "snapshots") ([string]$state.revision)
$shards = @(Get-ChildItem $snapshot -Filter "model-*.safetensors" -File `
    -ErrorAction SilentlyContinue)
$completeShardBytes = [int64](($shards | Measure-Object Length -Sum).Sum)
$indexPath = Join-Path $snapshot "model.safetensors.index.json"
$indexTensorBytes = [int64]0
$missingReferencedFiles = @()
if (Test-Path $indexPath -PathType Leaf) {
    $index = Get-Content $indexPath -Raw | ConvertFrom-Json
    $indexTensorBytes = [int64]$index.metadata.total_size
    $referenced = @($index.weight_map.PSObject.Properties.Value | Sort-Object -Unique)
    $missingReferencedFiles = @($referenced | Where-Object {
        -not (Test-Path (Join-Path $snapshot ([string]$_)) -PathType Leaf)
    })
}
$exit = if (Test-Path ([string]$state.exit_status) -PathType Leaf) {
    Get-Content ([string]$state.exit_status) -Raw | ConvertFrom-Json
} else { $null }
$startedUtc = [DateTimeOffset]::Parse([string]$state.started_utc).UtcDateTime
$elapsed = [Math]::Max(0.001, ([DateTime]::UtcNow - $startedUtc).TotalSeconds)
$initialCompleteShardBytes = if ($null -ne $state.PSObject.Properties[
        "initial_complete_shard_bytes"]) {
    [int64]$state.initial_complete_shard_bytes
} else { [int64]0 }
$downloadedThisRun = [Math]::Max([int64]0,
    $completeShardBytes - $initialCompleteShardBytes)
$bytesPerSecond = [double]$downloadedThisRun / $elapsed
$remaining = [Math]::Max([int64]0,
    [int64]$state.expected_tensor_bytes - $completeShardBytes)
$complete = $null -ne $exit -and [int]$exit.exit_code -eq 0 -and
    $shards.Count -eq [int]$state.expected_shards -and
    $indexTensorBytes -eq [int64]$state.expected_tensor_bytes -and
    $missingReferencedFiles.Count -eq 0 -and $incomplete.Count -eq 0

[PSCustomObject]@{
    model_id = $state.model_id
    revision = $state.revision
    xet_high_performance = [bool]$state.xet_high_performance
    pid = if ($null -ne $worker) { [int]$worker.ProcessId } else { $null }
    task_name = if ($null -ne $task) { $task.TaskName } else { $null }
    task_state = if ($null -ne $task) { [string]$task.State } else { $null }
    running = ($null -ne $worker) -or $taskRunning
    completed = $complete
    complete_shards = $shards.Count
    expected_shards = [int]$state.expected_shards
    incomplete_transfers = $incomplete.Count
    missing_referenced_files = $missingReferencedFiles.Count
    index_tensor_bytes = $indexTensorBytes
    expected_tensor_bytes = [int64]$state.expected_tensor_bytes
    complete_shard_bytes = $completeShardBytes
    remaining_tensor_bytes = $remaining
    cached_blob_bytes = $blobBytes
    expected_download_bytes = [int64]$state.expected_download_bytes
    bytes_per_second_since_start = $bytesPerSecond
    estimated_seconds_remaining = if ($bytesPerSecond -gt 0) {
        [double]$remaining / $bytesPerSecond
    } else { $null }
    exit_code = if ($null -ne $exit) { $exit.exit_code } else { $null }
    exit_error = if ($null -ne $exit) { $exit.error } else { $null }
    stderr_tail = @(Get-Content ([string]$state.stderr) -Tail 8 `
        -ErrorAction SilentlyContinue | ForEach-Object { [string]$_ })
} | ConvertTo-Json -Depth 4
