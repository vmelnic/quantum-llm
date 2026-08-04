. (Join-Path $PSScriptRoot "Common.ps1")

$artifact = Join-Path (Join-Path $script:RepoRoot "artifacts") "p6-download-process.json"
if (-not (Test-Path $artifact)) { throw "No P6 download process artifact" }
$state = Get-Content $artifact -Raw | ConvertFrom-Json
$process = if ([int]$state.pid -gt 0) {
    Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
} else { $null }
if ($null -eq $process -or $process.ProcessName -notin @("python", "hf", "powershell")) {
    # hf.exe is a launcher; Xet work may continue in its Python child after
    # the launcher PID disappears. Recover the tracked PID from the command
    # line instead of declaring a live resumable download dead.
    $escapedModel = [WildcardPattern]::Escape([string]$state.model_id)
    $worker = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -in @("python.exe", "hf.exe", "powershell.exe") -and
            $_.CommandLine -like "*download*$escapedModel*" -or
            ($_.Name -in @("python.exe", "hf.exe", "powershell.exe") -and
             $_.CommandLine -like "*$escapedModel*--max-workers*")
        } | Select-Object -First 1
    if ($null -ne $worker) {
        $state.pid = [int]$worker.ProcessId
        $state | ConvertTo-Json | Set-Content $artifact -Encoding UTF8
        $process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
    }
}
$task = if ($null -ne $state.PSObject.Properties["task_name"]) {
    Get-ScheduledTask -TaskName ([string]$state.task_name) -ErrorAction SilentlyContinue
} else { $null }
$taskRunning = $null -ne $task -and $task.State -eq "Running"
$cacheName = "models--" + ($state.model_id -replace "/", "--")
$root = Join-Path (Join-Path $env:USERPROFILE ".cache\huggingface\hub") $cacheName
$bytes = if (Test-Path $root) {
    [int64]((Get-ChildItem $root -File -Recurse -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum)
} else { [int64]0 }
$shards = @(Get-ChildItem (Join-Path $root "snapshots") -Filter "model-*.safetensors" `
    -File -Recurse -ErrorAction SilentlyContinue)
$incomplete = @(Get-ChildItem (Join-Path $root "blobs") -Filter "*.incomplete" `
    -File -ErrorAction SilentlyContinue)
$exitPath = if ($null -ne $state.PSObject.Properties["exit_status"]) {
    [string]$state.exit_status
} else {
    Join-Path (Join-Path $script:RepoRoot "artifacts") "p6-download-exit.json"
}
$exit = if (Test-Path -LiteralPath $exitPath -PathType Leaf) {
    Get-Content -LiteralPath $exitPath -Raw | ConvertFrom-Json
} else { $null }
[PSCustomObject]@{
    model_id = $state.model_id
    pid = $state.pid
    task_name = if ($null -ne $task) { $task.TaskName } else { $null }
    task_state = if ($null -ne $task) { [string]$task.State } else { $null }
    running = ($null -ne $process) -or $taskRunning
    completed = ($null -eq $process) -and (-not $taskRunning) -and `
        $shards.Count -eq 41 -and $incomplete.Count -eq 0
    complete_shards = $shards.Count
    expected_shards = 41
    incomplete_transfers = $incomplete.Count
    cached_bytes = $bytes
    exit_code = if ($null -ne $exit) { $exit.exit_code } else { $null }
    exit_error = if ($null -ne $exit) { $exit.error } else { $null }
    stderr_tail = @(Get-Content $state.stderr -Tail 5 -ErrorAction SilentlyContinue)
} | ConvertTo-Json -Depth 4
