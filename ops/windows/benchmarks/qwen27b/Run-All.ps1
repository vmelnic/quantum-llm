param([Parameter(Mandatory = $true)][string]$ModelRoot)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $ModelRoot -PathType Container)) {
    throw 'MODEL_ROOT does not exist'
}
$env:MODEL_ROOT = [System.IO.Path]::GetFullPath($ModelRoot)
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
$resultRoot = Join-Path $repoRoot 'out/benchmarks/qwen27b-english-speed'
$completePath = Join-Path $resultRoot 'COMPLETE.json'
if (Test-Path -LiteralPath $completePath) { throw 'Benchmark already completed; refusing to overwrite it' }
[System.IO.Directory]::CreateDirectory($resultRoot) | Out-Null

$roundIds = @(
    'fp4_q4h', 'fp4_k1', 'fp4_f16',
    'msev2_q4h', 'msev2_k1', 'msev2_f16',
    'v3_q4h', 'v3_k1', 'v3_f16',
    'v4_q4h', 'v4_k1', 'v4_f16'
)

function Write-JsonAtomic([string]$Path, $Value) {
    $temporary = "$Path.partial"
    $json = ConvertTo-Json -InputObject $Value -Depth 30
    [System.IO.File]::WriteAllText($temporary, "$json`n", [System.Text.UTF8Encoding]::new($false))
    if ([System.IO.File]::Exists($Path)) {
        [System.IO.File]::Replace($temporary, $Path, "$Path.backup")
    } else {
        [System.IO.File]::Move($temporary, $Path)
    }
}

function Write-Ledger([object[]]$Results) {
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('# Qwen3.8-27B English xhigh speed matrix')
    $lines.Add('')
    $lines.Add('Each round runs one fresh service profile and exactly one identical English startup-ideas request. MTP-4, Q8 activations, temperature 1.0, top-p 0.95, top-k 20, presence penalty 0.5, seed 314159, and no retained session are fixed. The 32,768-token ceiling is a runaway guard; length-terminated rounds are invalid.')
    $lines.Add('')
    $lines.Add('| Round | Status | Prompt tokens | Completion tokens | Prefill s | TTFT s | Decode tok/s | HTTP end-to-end tok/s |')
    $lines.Add('|---|---|---:|---:|---:|---:|---:|---:|')
    foreach ($item in $Results) {
        if ($item.status -eq 'complete') {
            $m = $item.metrics
            $lines.Add(('| `{0}` | complete | {1} | {2} | {3:N3} | {4:N3} | {5:N2} | {6:N2} |' -f $item.round_id, $m.prompt_tokens, $m.completion_tokens, $m.prefill_seconds, $m.ttft_seconds, $m.decode_tokens_per_second, $m.end_to_end_tokens_per_second))
        } else {
            $lines.Add("| ``$($item.round_id)`` | $($item.status) | - | - | - | - | - | - |")
        }
    }
    $lines.Add('')
    $lines.Add('The raw request, response, and provider-matched measurements are under each round directory. A round is rankable only if its single request has normal stop and visible output.')
    $temporary = Join-Path $resultRoot 'ledger.md.partial'
    [System.IO.File]::WriteAllText($temporary, ($lines -join "`n") + "`n", [System.Text.UTF8Encoding]::new($false))
    $ledgerPath = Join-Path $resultRoot 'ledger.md'
    if ([System.IO.File]::Exists($ledgerPath)) {
        [System.IO.File]::Replace($temporary, $ledgerPath, "$ledgerPath.backup")
    } else {
        [System.IO.File]::Move($temporary, $ledgerPath)
    }
}

function Get-CompletedMetrics([string]$RoundId) {
    $roundRoot = Join-Path $resultRoot $RoundId
    $summaryPath = Join-Path $roundRoot 'summary.json'
    $cleanupPath = Join-Path $roundRoot 'cleanup.json'
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $cleanupPath -PathType Leaf)) {
        throw "Round $RoundId has no complete result and cleanup"
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $cleanup = Get-Content -LiteralPath $cleanupPath -Raw | ConvertFrom-Json
    if ($summary.schema -ne 'qwen-speed-round-v2' -or
        $summary.profile.run_id -ne $RoundId -or
        $summary.profile.requests_per_profile -ne 1 -or
        $summary.metrics.finish_reason -ne 'stop' -or
        $summary.metrics.visible_tokens -le 0 -or
        -not $cleanup.port_free) {
        throw "Round $RoundId failed its single-request or cleanup gate"
    }
    return $summary.metrics
}

$results = [System.Collections.Generic.List[object]]::new()
$startedPath = Join-Path $resultRoot 'STARTED.json'
$resultsPath = Join-Path $resultRoot 'RESULTS.json'
if (Test-Path -LiteralPath $startedPath -PathType Leaf) {
    $started = Get-Content -LiteralPath $startedPath -Raw | ConvertFrom-Json
    if ($started.schema -ne 'qwen-english-speed-matrix-v2' -or
        (@($started.round_ids) -join '|') -ne ($roundIds -join '|') -or
        -not (Test-Path -LiteralPath $resultsPath -PathType Leaf)) {
        throw 'Existing benchmark state does not match this matrix'
    }
    $startedUtc = $started.started_utc
    $existing = Get-Content -LiteralPath $resultsPath -Raw | ConvertFrom-Json
    foreach ($old in @($existing.rounds)) {
        if ($results.Count -ge $roundIds.Count -or
            $old.round_id -ne $roundIds[$results.Count]) {
            throw 'Existing rounds are not a valid matrix prefix'
        }
        $entry = [ordered]@{
            round_id = $old.round_id
            status = $old.status
            started_utc = $old.started_utc
        }
        if ($old.status -eq 'running' -or $old.status -eq 'complete') {
            $entry.metrics = Get-CompletedMetrics $old.round_id
            $entry.status = 'complete'
            $entry.finished_utc = if ($old.finished_utc) {
                $old.finished_utc
            } else {
                [DateTime]::UtcNow.ToString('o')
            }
        } elseif ($old.status -eq 'failed') {
            $entry.error = $old.error
            $entry.finished_utc = $old.finished_utc
        } else {
            throw "Unknown existing round status: $($old.status)"
        }
        $results.Add($entry)
    }
    Write-JsonAtomic $resultsPath @{
        schema = 'qwen-english-speed-matrix-v2'
        started_utc = $startedUtc
        rounds = @($results.ToArray())
    }
    Write-Ledger @($results.ToArray())
} else {
    $startedUtc = [DateTime]::UtcNow.ToString('o')
    Write-JsonAtomic $startedPath @{
        schema = 'qwen-english-speed-matrix-v2'
        started_utc = $startedUtc
        round_ids = $roundIds
        source = 'ops/windows/benchmarks/qwen27b/Run-All.ps1'
    }
}

foreach ($roundId in $roundIds) {
    if (@($results | Where-Object { $_.round_id -eq $roundId }).Count -ne 0) {
        continue
    }
    $entry = [ordered]@{
        round_id = $roundId
        status = 'running'
        started_utc = [DateTime]::UtcNow.ToString('o')
    }
    $results.Add($entry)
    Write-JsonAtomic (Join-Path $resultRoot 'RESULTS.json') @{
        schema = 'qwen-english-speed-matrix-v2'
        started_utc = $startedUtc
        rounds = @($results.ToArray())
    }
    Write-Ledger @($results.ToArray())
    try {
        & (Join-Path $PSScriptRoot "$roundId/Run.ps1")
        $metrics = Get-CompletedMetrics $roundId
        $entry.status = 'complete'
        $entry.metrics = $metrics
    } catch {
        $entry.status = 'failed'
        $entry.error = $_.Exception.Message
        $failure = @(
            "#### English xhigh speed round ``$roundId`` failed",
            '',
            "Reason: $($entry.error)",
            '',
            'The round is excluded from ranking. Partial request files, if any, remain in its directory.',
            ''
        ) -join "`n"
        [System.IO.File]::WriteAllText(
            (Join-Path $resultRoot "$roundId-failure.md"), $failure,
            [System.Text.UTF8Encoding]::new($false))
    }
    $entry.finished_utc = [DateTime]::UtcNow.ToString('o')
    Write-JsonAtomic (Join-Path $resultRoot 'RESULTS.json') @{
        schema = 'qwen-english-speed-matrix-v2'
        started_utc = $startedUtc
        rounds = @($results.ToArray())
    }
    Write-Ledger @($results.ToArray())
}

$rankable = @($results | Where-Object { $_.status -eq 'complete' } |
    Sort-Object { [double]$_.metrics.end_to_end_tokens_per_second } -Descending)
$decodeRanked = @($results | Where-Object { $_.status -eq 'complete' } |
    Sort-Object { [double]$_.metrics.decode_tokens_per_second } -Descending)
Write-JsonAtomic $completePath @{
    schema = 'qwen-english-speed-matrix-v2'
    started_utc = $startedUtc
    finished_utc = [DateTime]::UtcNow.ToString('o')
    completed_rounds = $rankable.Count
    failed_rounds = 12 - $rankable.Count
    end_to_end_winner = if ($rankable.Count) { $rankable[0].round_id } else { $null }
    decode_winner = if ($decodeRanked.Count) { $decodeRanked[0].round_id } else { $null }
    rounds = @($results.ToArray())
}
