param([string]$TaskName = "QuantumLLM-ExpertServer")

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $task) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}
[PSCustomObject]@{ status = "removed"; task = $TaskName } | ConvertTo-Json
