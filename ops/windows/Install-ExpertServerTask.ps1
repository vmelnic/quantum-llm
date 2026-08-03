param(
    [string]$TaskName = "QuantumLLM-ExpertServer",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [int]$MaximumQueue = 8,
    [string]$BuildId = "development",
    [switch]$Start
)

. (Join-Path $PSScriptRoot "Common.ps1")

$startScript = Join-Path $PSScriptRoot "Start-ExpertServer.ps1"
$arguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", "`"$startScript`"",
    "-HostAddress", $HostAddress,
    "-Port", $Port,
    "-MaximumQueue", $MaximumQueue,
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
    endpoint = "http://${HostAddress}:$Port"
    started = [bool]$Start
} | ConvertTo-Json
