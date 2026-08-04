param(
    [string]$TaskName = "QuantumLLM-P6ExpertServer",
    [int]$Port = 8080
)

. (Join-Path $PSScriptRoot "Common.ps1")

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $task) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}
$stopped = @(Stop-ExpertServerProcessTree -Port $Port)
if ($null -ne $task) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}
[PSCustomObject]@{
    status = "removed"
    task = $TaskName
    stopped_process_ids = $stopped
} | ConvertTo-Json
