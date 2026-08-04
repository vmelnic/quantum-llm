param(
    [string]$TaskName = "QuantumLLM-P6ExpertServer",
    [int]$Port = 8080
)

. (Join-Path $PSScriptRoot "Common.ps1")

Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$stopped = @(Stop-ExpertServerProcessTree -Port $Port)
[PSCustomObject]@{
    status = "stopped"
    task = $TaskName
    stopped_process_ids = $stopped
} | ConvertTo-Json
