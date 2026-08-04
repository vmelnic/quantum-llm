param(
    [ValidateSet("OLMoE", "P6")][string]$Profile = "OLMoE",
    [string]$TaskName = "",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [int]$MaximumQueue = 8,
    [int]$MaximumContext = 4096,
    [int]$WorkerCapacity = 0,
    [int]$WorkerRamCacheGiB = 48,
    [int]$WorkerVramCacheGiB = 14,
    [double]$MicrobatchWindowMs = 2.0,
    [int]$LatencyWindow = 4096,
    [int]$StartupTimeoutSeconds = 0,
    [string]$BuildId = "development",
    [switch]$Start
)

. (Join-Path $PSScriptRoot "Common.ps1")

$isP6 = $Profile -eq "P6"
if ($WorkerCapacity -eq 0) { $WorkerCapacity = if ($isP6) { 4 } else { 1 } }
if ($StartupTimeoutSeconds -eq 0) {
    $StartupTimeoutSeconds = if ($isP6) { 600 } else { 120 }
}
if ($WorkerCapacity -lt 1 -or $StartupTimeoutSeconds -lt 1) {
    throw "Invalid worker capacity or startup timeout"
}
if (-not $TaskName) {
    $TaskName = if ($isP6) { "QuantumLLM-P6ExpertServer" } else { "QuantumLLM-ExpertServer" }
}
$startScript = Join-Path $PSScriptRoot $(if ($isP6) {
    "Start-P6ExpertServer.ps1"
} else {
    "Start-ExpertServer.ps1"
})
if (-not (Test-Path -LiteralPath $startScript -PathType Leaf)) {
    throw "Start script is missing: $startScript"
}
$arguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", "`"$startScript`"",
    "-HostAddress", $HostAddress,
    "-Port", $Port,
    "-MaximumQueue", $MaximumQueue,
    "-MaximumContext", $MaximumContext,
    "-WorkerCapacity", $WorkerCapacity,
    "-WorkerRamCacheGiB", $WorkerRamCacheGiB,
    "-WorkerVramCacheGiB", $WorkerVramCacheGiB,
    "-MicrobatchWindowMs", $MicrobatchWindowMs.ToString([Globalization.CultureInfo]::InvariantCulture),
    "-LatencyWindow", $LatencyWindow,
    "-StartupTimeoutSeconds", $StartupTimeoutSeconds,
    "-BuildId", $BuildId
) -join " "
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Out-Null
if ($Start) { Start-ScheduledTask -TaskName $TaskName }
[PSCustomObject]@{
    status = "installed"
    task = $TaskName
    profile = $Profile
    start_script = $startScript
    endpoint = "http://${HostAddress}:$Port"
    started = [bool]$Start
} | ConvertTo-Json
