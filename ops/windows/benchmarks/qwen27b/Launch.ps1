param([string]$TaskName = 'QuantumLLM-Qwen27B-EnglishSpeed-Single-20260924')

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($env:MODEL_ROOT) -or
    -not (Test-Path -LiteralPath $env:MODEL_ROOT -PathType Container)) {
    throw 'MODEL_ROOT is required and must exist'
}
if ($TaskName -notmatch '^QuantumLLM-Qwen27B-EnglishSpeed-[A-Za-z0-9-]+$') {
    throw 'TaskName is invalid'
}
$runAll = Join-Path $PSScriptRoot 'Run-All.ps1'
if (-not (Test-Path -LiteralPath $runAll -PathType Leaf)) { throw 'Run-All.ps1 is missing' }
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task already exists: $TaskName"
}
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument (
    '-NoProfile -ExecutionPolicy Bypass -File "{0}" -ModelRoot "{1}"' -f
    $runAll, [System.IO.Path]::GetFullPath($env:MODEL_ROOT)
)
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2)
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId (
    [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
) -LogonType S4U -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -ErrorAction Stop | Out-Null
Start-ScheduledTask -TaskName $TaskName
@{
    task = $TaskName
    status = 'started'
    result_root = 'out/benchmarks/qwen27b-english-speed'
} | ConvertTo-Json -Compress
