param([string]$TaskName = 'QuantumLLM-Qwen27B-EnglishSpeed-Single-20260924')

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
$root = Join-Path $repoRoot 'out/benchmarks/qwen27b-english-speed'
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$result = @{
    task = $TaskName
    task_state = if ($task) { [string]$task.State } else { 'absent' }
    started = Test-Path -LiteralPath (Join-Path $root 'STARTED.json')
    complete = Test-Path -LiteralPath (Join-Path $root 'COMPLETE.json')
    rounds = @()
}
$resultsPath = Join-Path $root 'RESULTS.json'
if (Test-Path -LiteralPath $resultsPath -PathType Leaf) {
    $results = Get-Content -LiteralPath $resultsPath -Raw | ConvertFrom-Json
    $result.rounds = @($results.rounds | ForEach-Object {
        @{ round_id = $_.round_id; status = $_.status }
    })
}
if ($result.complete) {
    $complete = Get-Content -LiteralPath (Join-Path $root 'COMPLETE.json') -Raw |
        ConvertFrom-Json
    $result.end_to_end_winner = $complete.end_to_end_winner
    $result.decode_winner = $complete.decode_winner
    $result.completed_rounds = $complete.completed_rounds
    $result.failed_rounds = $complete.failed_rounds
}
$result | ConvertTo-Json -Compress -Depth 6
