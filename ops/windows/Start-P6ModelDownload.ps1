param(
    [string]$ModelId = "Qwen/Qwen3-Next-80B-A3B-Instruct",
    [int]$MaxWorkers = 4
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$hf = "C:\Users\vladi\.hf-cli\venv\Scripts\hf.exe"
if (-not (Test-Path $hf -PathType Leaf)) { $hf = (Get-Command hf.exe -ErrorAction Stop).Source }
$artifact = Join-Path (Join-Path $script:RepoRoot "artifacts") "p6-download-process.json"
$stdout = Join-Path (Join-Path $script:RepoRoot "logs") "p6-download.stdout.log"
$stderr = Join-Path (Join-Path $script:RepoRoot "logs") "p6-download.stderr.log"
$status = Join-Path (Join-Path $script:RepoRoot "artifacts") "p6-download-exit.json"
$taskName = "QuantumLLM-P6ModelDownload"
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $existingTask -and $existingTask.State -eq "Running") {
    throw "P6 download task is already running: $taskName"
}
if (Test-Path $artifact) {
    $previous = Get-Content $artifact -Raw | ConvertFrom-Json
    $running = if ([int]$previous.pid -gt 0) {
        Get-Process -Id ([int]$previous.pid) -ErrorAction SilentlyContinue
    } else { $null }
    if ($null -ne $running) { throw "P6 download is already running as PID $($previous.pid)" }
}
# Windows PowerShell 5.1 flattens ArgumentList itself and does not reliably
# preserve an item containing spaces. Reject those here because Hugging Face
# repository ids cannot contain whitespace, then pass one deterministic string.
if ($ModelId -match "\s") { throw "ModelId cannot contain whitespace: $ModelId" }
$escapedModel = [WildcardPattern]::Escape($ModelId)
$existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -in @("python.exe", "hf.exe") -and
        $_.CommandLine -like "*download*$escapedModel*"
    } | Select-Object -First 1
if ($null -ne $existing) {
    throw "P6 download is already running as PID $($existing.ProcessId)"
}
$worker = Join-Path $PSScriptRoot "Invoke-P6ModelDownloadWorker.ps1"
if (-not (Test-Path $worker -PathType Leaf)) { throw "Download worker is missing: $worker" }
Remove-Item -LiteralPath $status -Force -ErrorAction SilentlyContinue
$workerArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$worker`" " +
    "-HfPath `"$hf`" -ModelId $ModelId -MaxWorkers $MaxWorkers " +
    "-StatusPath `"$status`""
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
    pid = 0
    task_name = $taskName
    started_utc = [DateTime]::UtcNow.ToString("o")
    stdout = $stdout
    stderr = $stderr
    exit_status = $status
} | ConvertTo-Json | Set-Content $artifact -Encoding UTF8
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Milliseconds 500
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ($task.State -ne "Running") {
    $detail = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction SilentlyContinue
    throw "P6 download task did not remain running (state=$($task.State), result=$($detail.LastTaskResult))"
}
Get-Content $artifact -Raw
