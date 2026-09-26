param(
    [Parameter(Mandatory = $true)][string]$ModelRoot,
    [Parameter(Mandatory = $true)][ValidateSet('quantum', 'llama')][string]$Backend
)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
$root = [System.IO.Path]::GetFullPath($ModelRoot)
if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw 'MODEL_ROOT is missing' }
$taskName = "QuantumLLM-Pi256k-$Backend"
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw "Benchmark task already exists: $taskName"
}
if (Test-Path -LiteralPath (Join-Path $repo "out/benchmarks/pi256k/$Backend")) {
    throw 'Benchmark result directory already exists'
}
$script = Join-Path $PSScriptRoot 'Run.ps1'
$arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -ModelRoot "{1}" -Backend {2}' -f `
    $script, $root, $Backend
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId (
    [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action `
    -Settings $settings -Principal $principal -ErrorAction Stop | Out-Null
Start-ScheduledTask -TaskName $taskName
@{ task = $taskName; status = 'started'; backend = $Backend;
   result_root = "out/benchmarks/pi256k/$Backend" } | ConvertTo-Json -Compress
