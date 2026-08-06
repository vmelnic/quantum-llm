param(
    [Parameter(Mandatory = $true)][string]$Bundle,
    [string]$Runner = "",
    [int]$MaximumContext = 4096,
    [int]$RamCacheGiB = 40,
    [int]$VramCacheGiB = 12,
    [int[]]$ExpectedTokens = @(14, 552, 438, 223, 25)
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$root = [System.IO.Path]::GetFullPath($Bundle)
if (-not $Runner) {
    $Runner = Join-Path $script:RepoRoot `
        "out\build\windows-msvc-release\runtime\Release\expert-deepseek-worker.exe"
}
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "DeepSeek worker is missing: $Runner"
}

function Invoke-WorkerSequence {
    param(
        [Parameter(Mandatory = $true)][bool]$EnableMtp,
        [Parameter(Mandatory = $true)][int]$Steps
    )

    $commands = [System.Collections.Generic.List[string]]::new()
    $commands.Add("BEGIN`t1`t16`t42")
    for ($index = 0; $index -lt $Steps; ++$index) {
        $commands.Add("NEXT`t1`t0")
    }
    $commands.Add("STATS")
    $commands.Add("END`t1")
    $commands.Add("SHUTDOWN")
    $arguments = @(
        $root, "--worker", $MaximumContext, $RamCacheGiB, $VramCacheGiB,
        1, 2048, 256, "balanced"
    )
    if ($EnableMtp) { $arguments += "--enable-mtp" }
    $started = [System.Diagnostics.Stopwatch]::StartNew()
    $lines = @($commands | & $Runner @arguments 2>&1 | ForEach-Object {
        [string]$_
    })
    $started.Stop()
    if ($LASTEXITCODE -ne 0) {
        throw "DeepSeek worker failed with exit code $LASTEXITCODE`n$($lines -join [Environment]::NewLine)"
    }
    $messages = @($lines | ForEach-Object {
        try { $_ | ConvertFrom-Json -ErrorAction Stop } catch { }
    })
    $ready = @($messages | Where-Object { $_.type -eq "ready" })
    $tokens = @($messages | Where-Object { $_.type -eq "token" })
    $stats = @($messages | Where-Object { $_.type -eq "stats" })
    $shutdown = @($messages | Where-Object { $_.type -eq "shutdown" })
    if ($ready.Count -ne 1 -or $tokens.Count -ne $Steps -or
        $stats.Count -ne 1 -or $shutdown.Count -ne 1) {
        throw "DeepSeek worker protocol gate returned an incomplete transcript`n$($lines -join [Environment]::NewLine)"
    }
    $actual = @($tokens | ForEach-Object { @($_.tokens) } |
        ForEach-Object { [int]$_ })
    return [PSCustomObject]@{
        mtp_enabled = $EnableMtp
        tokens = $actual
        worker_model_steps = [uint64]$stats[0].worker_model_steps
        worker_model_rows = [uint64]$stats[0].worker_model_rows
        worker_model_step_ns = [uint64]$stats[0].worker_model_step_ns
        useful_tokens = [uint64]$stats[0].worker_useful_tokens
        verify_pairs = [uint64]$stats[0].worker_verify_pairs
        mtp_accepted = [uint64]$stats[0].worker_mtp_accepted
        mtp_rejected = [uint64]$stats[0].worker_mtp_rejected
        prefetch_predictions = [uint64]$stats[0].scheduler_prefetch_predictions
        prefetch_scheduled = [uint64]$stats[0].scheduler_prefetch_scheduled
        prefetch_completed = [uint64]$stats[0].scheduler_prefetch_completed
        prefetch_useful = [uint64]$stats[0].scheduler_prefetch_useful
        prefetch_late = [uint64]$stats[0].scheduler_prefetch_late
        prefetch_incorrect = [uint64]$stats[0].scheduler_prefetch_incorrect
        prefetch_cancelled = [uint64]$stats[0].scheduler_prefetch_cancelled
        wall_seconds = $started.Elapsed.TotalSeconds
    }
}

$speculative = Invoke-WorkerSequence -EnableMtp $true -Steps 3
$ordinary = Invoke-WorkerSequence -EnableMtp $false -Steps 5
$expected = $ExpectedTokens -join ","
if (($speculative.tokens -join ",") -ne $expected -or
    ($ordinary.tokens -join ",") -ne $expected) {
    throw "Target sequence mismatch: expected=$expected speculative=$($speculative.tokens -join ',') ordinary=$($ordinary.tokens -join ',')"
}
if ($speculative.verify_pairs -ne 3 -or
    $speculative.mtp_accepted + $speculative.mtp_rejected -ne 3) {
    throw "Speculative transaction accounting is incomplete"
}

$result = [PSCustomObject]@{
    schema_version = 1
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    expected_tokens = $ExpectedTokens
    speculative = $speculative
    ordinary = $ordinary
    status = "pass"
}
$path = Write-JsonArtifact -Value $result -Name "deepseek-mtp-pair-gate-latest.json"
Write-Output ($result | ConvertTo-Json -Depth 5 -Compress)
Write-Output "DeepSeek MTP pair gate passed: $path"
