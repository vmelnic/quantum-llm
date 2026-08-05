param(
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [Parameter(Mandatory = $true)][string]$RoutedCatalog,
    [Parameter(Mandatory = $true)][string]$Output,
    [int64]$ExpectedPayloadBytes = 147169738752,
    [int64]$SafetyBytes = 8GB
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$source = [System.IO.Path]::GetFullPath($Snapshot)
$catalog = [System.IO.Path]::GetFullPath($RoutedCatalog)
$destination = [System.IO.Path]::GetFullPath($Output)
$workRoot = [System.IO.Path]::GetFullPath((Join-Path $script:RepoRoot "work"))
if (-not (Test-Path $source -PathType Container) -or
    -not (Test-Path (Join-Path $catalog "catalog.tsv") -PathType Leaf)) {
    throw "Snapshot or routed catalog is incomplete"
}
if ($destination.StartsWith($workRoot + [System.IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase) -or $destination -eq $workRoot) {
    throw "Durable compact packs cannot be published under repository work/"
}
if ($destination.StartsWith($source + [System.IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase) -or $destination -eq $source) {
    throw "Compact-pack output cannot overlap the source checkpoint"
}
if ($ExpectedPayloadBytes -ne 147169738752 -or $SafetyBytes -lt 1GB) {
    throw "DeepSeek compact-pack size or safety reserve is invalid"
}
if (Test-Path (Join-Path $destination "manifest.json") -PathType Leaf) {
    throw "A published compact pack already exists at $destination"
}

$partial = $destination + ".partial"
$partialBytes = [int64]0
if (Test-Path $partial -PathType Container) {
    foreach ($shard in @(Get-ChildItem $partial -Filter "experts-*.dsc" `
            -File -ErrorAction SilentlyContinue)) {
        $partialBytes += [int64]$shard.Length
    }
}
$remainingBytes = [Math]::Max([int64]0, $ExpectedPayloadBytes - $partialBytes)
$root = [System.IO.Path]::GetPathRoot($destination)
$driveName = $root.Substring(0, 1)
$freeBytes = [int64](Get-PSDrive -Name $driveName).Free
if ($freeBytes -lt $remainingBytes + $SafetyBytes) {
    throw "Insufficient disk for compact pack: free=$freeBytes remaining=$remainingBytes safety=$SafetyBytes"
}

$taskName = "QuantumLLM-DeepSeekCompactPack"
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $existingTask -and $existingTask.State -eq "Running") {
    throw "A DeepSeek compact-pack task is already running"
}
$artifactDirectory = Join-Path $script:RepoRoot "artifacts"
$logDirectory = Join-Path $script:RepoRoot "logs"
$artifact = Join-Path $artifactDirectory "deepseek-pack-process.json"
$stdout = Join-Path $logDirectory "deepseek-pack.stdout.log"
$stderr = Join-Path $logDirectory "deepseek-pack.stderr.log"
$status = Join-Path $artifactDirectory "deepseek-pack-exit.json"
$worker = Join-Path $PSScriptRoot "Invoke-DeepSeekCompactPackWorker.ps1"
Remove-Item -LiteralPath $status -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stdout -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderr -Force -ErrorAction SilentlyContinue

$workerArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$worker`" " +
    "-Snapshot `"$source`" -RoutedCatalog `"$catalog`" " +
    "-Output `"$destination`" -StdoutPath `"$stdout`" " +
    "-StderrPath `"$stderr`" -StatusPath `"$status`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $workerArguments
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 1 -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$currentIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $currentIdentity `
    -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings `
    -Principal $principal -Force | Out-Null

[PSCustomObject]@{
    schema_version = 1
    snapshot = $source
    routed_catalog = $catalog
    output = $destination
    partial = $partial
    expected_payload_bytes = $ExpectedPayloadBytes
    initial_partial_bytes = $partialBytes
    free_bytes_before = $freeBytes
    safety_bytes = $SafetyBytes
    task_name = $taskName
    started_utc = [DateTime]::UtcNow.ToString("o")
    stdout = $stdout
    stderr = $stderr
    exit_status = $status
} | ConvertTo-Json | Set-Content -LiteralPath $artifact -Encoding UTF8

Start-ScheduledTask -TaskName $taskName
Start-Sleep -Milliseconds 750
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ($task.State -ne "Running" -and -not (Test-Path $status -PathType Leaf)) {
    $detail = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction SilentlyContinue
    throw "Compact-pack task failed to start (state=$($task.State), result=$($detail.LastTaskResult))"
}
Get-Content $artifact -Raw
