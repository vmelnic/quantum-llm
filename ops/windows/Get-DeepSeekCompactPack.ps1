. (Join-Path $PSScriptRoot "Common.ps1")

$artifact = Join-Path (Join-Path $script:RepoRoot "artifacts") `
    "deepseek-pack-process.json"
if (-not (Test-Path $artifact -PathType Leaf)) {
    throw "No DeepSeek compact-pack artifact"
}
$state = Get-Content $artifact -Raw | ConvertFrom-Json
$task = Get-ScheduledTask -TaskName ([string]$state.task_name) `
    -ErrorAction SilentlyContinue
$partial = [string]$state.partial
$output = [string]$state.output
$activeRoot = if (Test-Path $output -PathType Container) { $output } else { $partial }
$shards = @(Get-ChildItem $activeRoot -Filter "experts-*.dsc" -File `
    -ErrorAction SilentlyContinue)
$commitMarkers = @(Get-ChildItem $activeRoot -Filter "experts-*.dsc.commit.json" `
    -File -ErrorAction SilentlyContinue)
$packedBytes = [int64]0
foreach ($shard in $shards) { $packedBytes += [int64]$shard.Length }
$exit = if (Test-Path ([string]$state.exit_status) -PathType Leaf) {
    Get-Content ([string]$state.exit_status) -Raw | ConvertFrom-Json
} else { $null }
$complete = $null -ne $exit -and [int]$exit.exit_code -eq 0 -and
    (Test-Path (Join-Path $output "manifest.json") -PathType Leaf) -and
    $shards.Count -eq 43 -and $commitMarkers.Count -eq 43 -and
    $packedBytes -eq [int64]$state.expected_payload_bytes

[PSCustomObject]@{
    task_name = if ($null -ne $task) { $task.TaskName } else { $null }
    task_state = if ($null -ne $task) { [string]$task.State } else { $null }
    running = $null -ne $task -and $task.State -eq "Running"
    completed = $complete
    output = $output
    complete_layer_shards = $shards.Count
    commit_markers = $commitMarkers.Count
    packed_bytes = $packedBytes
    expected_payload_bytes = [int64]$state.expected_payload_bytes
    remaining_bytes = [Math]::Max([int64]0,
        [int64]$state.expected_payload_bytes - $packedBytes)
    exit_code = if ($null -ne $exit) { $exit.exit_code } else { $null }
    exit_error = if ($null -ne $exit) { $exit.error } else { $null }
    stdout_tail = @(Get-Content -LiteralPath ([string]$state.stdout) -Tail 8 `
        -ErrorAction SilentlyContinue | ForEach-Object { [string]$_ })
    stderr_tail = @(Get-Content -LiteralPath ([string]$state.stderr) -Tail 8 `
        -ErrorAction SilentlyContinue | ForEach-Object { [string]$_ })
} | ConvertTo-Json -Depth 4
