param([Parameter(Mandatory = $true)][ValidateSet('quantum', 'llama')][string]$Backend)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
$name = "QuantumLLM-Pi256k-$Backend"
$task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
$taskInfo = if ($task) { Get-ScheduledTaskInfo -TaskName $name } else { $null }
$root = Join-Path $repo "out/benchmarks/pi256k/$Backend"
$progress = Join-Path $root 'PROGRESS.json'
$complete = Join-Path $root 'COMPLETE.json'
$failure = Join-Path $repo "out/benchmarks/pi256k/$Backend-FAILED.json"
@{
    task = $name
    task_state = if ($task) { [string]$task.State } else { 'absent' }
    task_result = if ($taskInfo) { $taskInfo.LastTaskResult } else { $null }
    progress = if (Test-Path -LiteralPath $progress) {
        Get-Content -LiteralPath $progress -Raw | ConvertFrom-Json
    } else { $null }
    complete = if (Test-Path -LiteralPath $complete) {
        Get-Content -LiteralPath $complete -Raw | ConvertFrom-Json
    } else { $null }
    launch_failure = if (Test-Path -LiteralPath $failure) {
        Get-Content -LiteralPath $failure -Raw | ConvertFrom-Json
    } else { $null }
} | ConvertTo-Json -Depth 8 -Compress
