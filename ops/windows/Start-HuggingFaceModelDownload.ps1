param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [Parameter(Mandatory = $true)][int64]$ExpectedDownloadBytes,
    [Parameter(Mandatory = $true)][int64]$ExpectedTensorBytes,
    [Parameter(Mandatory = $true)][int]$ExpectedShards,
    [int]$MaxWorkers = 4,
    [int64]$SafetyBytes = 8GB
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ($ModelId -match "\s" -or $Revision -notmatch '^[0-9a-f]{40}$') {
    throw "ModelId or pinned revision is invalid"
}
if ($ExpectedDownloadBytes -lt 1 -or $ExpectedTensorBytes -lt 1 -or
    $ExpectedShards -lt 1 -or $MaxWorkers -lt 1 -or $SafetyBytes -lt 1) {
    throw "Download limits must be positive"
}

$hf = Join-Path $env:USERPROFILE ".hf-cli\venv\Scripts\hf.exe"
if (-not (Test-Path $hf -PathType Leaf)) {
    $hf = (Get-Command hf.exe -ErrorAction Stop).Source
}
$hfPython = Join-Path (Split-Path (Split-Path $hf -Parent) -Parent) "Scripts\python.exe"
if (-not (Test-Path $hfPython -PathType Leaf)) {
    $hfPython = Join-Path (Split-Path $hf -Parent) "python.exe"
}
if (-not (Test-Path $hfPython -PathType Leaf)) {
    throw "Cannot verify hf_xet beside $hf"
}
& $hfPython -c "import hf_xet" 2>$null
if ($LASTEXITCODE -ne 0) { throw "hf_xet is not installed in the hf environment" }

$artifactDirectory = Join-Path $script:RepoRoot "artifacts"
$logDirectory = Join-Path $script:RepoRoot "logs"
$artifact = Join-Path $artifactDirectory "hf-download-process.json"
$stdout = Join-Path $logDirectory "hf-download.stdout.log"
$stderr = Join-Path $logDirectory "hf-download.stderr.log"
$status = Join-Path $artifactDirectory "hf-download-exit.json"
$taskName = "QuantumLLM-HuggingFaceDownload"

$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $existingTask -and $existingTask.State -eq "Running") {
    throw "A Hugging Face download task is already running: $taskName"
}
$escapedModel = [WildcardPattern]::Escape($ModelId)
$existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.ProcessId -ne $PID -and
        $_.Name -in @("python.exe", "hf.exe", "powershell.exe") -and
        $_.CommandLine -like "*download*$escapedModel*"
    } | Select-Object -First 1
if ($null -ne $existing) {
    throw "Model download is already running as PID $($existing.ProcessId)"
}

$cacheName = "models--" + ($ModelId -replace "/", "--")
$cacheRoot = Join-Path (Join-Path $env:USERPROFILE ".cache\huggingface\hub") $cacheName
$snapshot = Join-Path (Join-Path $cacheRoot "snapshots") $Revision
$initialShardBytes = [int64]0
foreach ($shard in @(Get-ChildItem $snapshot -Filter "model-*.safetensors" `
        -File -ErrorAction SilentlyContinue)) {
    $initialShardBytes += [int64]$shard.Length
}
$cachedBytes = [int64]0
foreach ($blob in @(Get-ChildItem (Join-Path $cacheRoot "blobs") -File `
        -ErrorAction SilentlyContinue)) {
    $cachedBytes += [int64]$blob.Length
}
$remainingBytes = [Math]::Max([int64]0, $ExpectedDownloadBytes - $cachedBytes)
$driveName = ([System.IO.Path]::GetPathRoot($env:USERPROFILE)).Substring(0, 1)
$freeBytes = [int64](Get-PSDrive -Name $driveName).Free
if ($freeBytes -lt $remainingBytes + $SafetyBytes) {
    throw "Insufficient disk: free=$freeBytes remaining=$remainingBytes safety=$SafetyBytes"
}

$worker = Join-Path $PSScriptRoot "Invoke-HuggingFaceDownloadWorker.ps1"
if (-not (Test-Path $worker -PathType Leaf)) {
    throw "Download worker is missing: $worker"
}
Remove-Item -LiteralPath $status -Force -ErrorAction SilentlyContinue
$workerArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$worker`" " +
    "-HfPath `"$hf`" -ModelId $ModelId -Revision $Revision " +
    "-MaxWorkers $MaxWorkers -StdoutPath `"$stdout`" " +
    "-StderrPath `"$stderr`" -StatusPath `"$status`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $workerArguments
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$currentIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $currentIdentity `
    -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings `
    -Principal $principal -Force | Out-Null

[PSCustomObject]@{
    schema_version = 1
    model_id = $ModelId
    revision = $Revision
    expected_download_bytes = $ExpectedDownloadBytes
    expected_tensor_bytes = $ExpectedTensorBytes
    expected_shards = $ExpectedShards
    initial_cached_bytes = $cachedBytes
    initial_complete_shard_bytes = $initialShardBytes
    free_bytes_before = $freeBytes
    safety_bytes = $SafetyBytes
    max_workers = $MaxWorkers
    xet_high_performance = $true
    task_name = $taskName
    started_utc = [DateTime]::UtcNow.ToString("o")
    stdout = $stdout
    stderr = $stderr
    exit_status = $status
} | ConvertTo-Json | Set-Content $artifact -Encoding UTF8

Start-ScheduledTask -TaskName $taskName
Start-Sleep -Milliseconds 750
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ($task.State -ne "Running" -and -not (Test-Path $status -PathType Leaf)) {
    $detail = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction SilentlyContinue
    throw "Download task failed to start (state=$($task.State), result=$($detail.LastTaskResult))"
}
Get-Content $artifact -Raw
