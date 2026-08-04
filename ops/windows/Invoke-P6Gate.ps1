param(
    [ValidateSet("Single", "Batch", "Both")]
    [string]$Mode = "Both",
    [string]$Container = "C:\Users\vladi\quantum-llm\work\models\qwen3-next-80b-expert-pack-int8",
    [string]$PromptTokenIds = "151644,872,374",
    [string]$BatchPromptTokenIds = "151644,872,374;151644,9707,374;151644,17,488,17;151644,3696,13362",
    [int]$NewTokens = 32,
    [int]$Concurrency = 4,
    [int]$RamCacheGiB = 48,
    [int]$VramCacheGiB = 14,
    [int]$MinimumFreePhysicalGiB = 2,
    [int]$WarmupRounds = 1,
    [double]$SingleTokensPerSecond = 10.0,
    [double]$AggregateTokensPerSecond = 30.0
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ($NewTokens -lt 2 -or $Concurrency -lt 1 -or $RamCacheGiB -lt 1 -or
    $VramCacheGiB -lt 1 -or $MinimumFreePhysicalGiB -lt 1 -or
    $WarmupRounds -lt 0) {
    throw "Invalid P6 gate settings"
}
$runner = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-qwen3-next-runner.exe"
if (-not (Test-Path $runner -PathType Leaf)) { throw "P6 runner missing: $runner" }
if (-not (Test-Path (Join-Path $Container "manifest.json") -PathType Leaf)) {
    throw "P6 container missing or incomplete: $Container"
}

function Get-PagefileUsageMiB {
    $usage = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue |
        Measure-Object CurrentUsage -Sum
    return [int64]$usage.Sum
}

function Invoke-GateRun {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Arguments,
        [Parameter(Mandatory = $true)][double]$RequiredTokensPerSecond
    )
    $stdout = Join-Path $script:RepoRoot "logs\p6-gate-$Name.stdout.log"
    $stderr = Join-Path $script:RepoRoot "logs\p6-gate-$Name.stderr.log"
    $pagefileBefore = Get-PagefileUsageMiB
    $pagefilePeak = $pagefileBefore
    $minimumFreePhysical = [int64]::MaxValue
    $peakWorkingSet = [int64]0
    $peakPrivateBytes = [int64]0
    $started = [DateTime]::UtcNow
    $process = Start-Process -FilePath $runner -ArgumentList $Arguments `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    while (-not $process.HasExited) {
        $process.Refresh()
        $peakWorkingSet = [Math]::Max($peakWorkingSet, [int64]$process.WorkingSet64)
        $peakPrivateBytes = [Math]::Max($peakPrivateBytes, [int64]$process.PrivateMemorySize64)
        $freePhysical = [int64](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB
        $minimumFreePhysical = [Math]::Min($minimumFreePhysical, $freePhysical)
        $pagefilePeak = [Math]::Max($pagefilePeak, (Get-PagefileUsageMiB))
        Start-Sleep -Milliseconds 500
    }
    $process.Refresh()
    if ($process.ExitCode -ne 0) {
        $tail = @(Get-Content $stderr -Tail 30 -ErrorAction SilentlyContinue) -join `
            [Environment]::NewLine
        throw "P6 $Name runner failed with $($process.ExitCode):$([Environment]::NewLine)$tail"
    }
    $line = Get-Content $stdout -Tail 1 -ErrorAction Stop
    $model = $line | ConvertFrom-Json
    $pagefileAfter = Get-PagefileUsageMiB
    $pagefilePeak = [Math]::Max($pagefilePeak, $pagefileAfter)
    $pagefileGrowth = [Math]::Max([int64]0, $pagefilePeak - $pagefileBefore)
    $physicalRam = [int64](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    $minimumFreePhysicalRequired = [int64]$MinimumFreePhysicalGiB * 1GB
    $memoryHeadroomPass = $minimumFreePhysical -ge $minimumFreePhysicalRequired
    $swapIndependencePass = $pagefileGrowth -eq 0 -and $memoryHeadroomPass
    return [PSCustomObject]@{
        mode = $Name
        started_utc = $started.ToString("o")
        elapsed_seconds = ([DateTime]::UtcNow - $started).TotalSeconds
        required_tokens_per_second = $RequiredTokensPerSecond
        measured_tokens_per_second = [double]$model.tokens_per_second
        speed_gate_pass = [double]$model.tokens_per_second -ge $RequiredTokensPerSecond
        correctness_gate_pass = if ($Name -eq "batch") {
            [bool]$model.mixed_prompts -and [bool]$model.interleaving_match
        } else { $true }
        container_bytes = [int64]$model.container_bytes
        physical_ram_bytes = $physicalRam
        container_larger_than_ram = [int64]$model.container_bytes -gt $physicalRam
        pagefile_usage_mib_before = $pagefileBefore
        pagefile_usage_mib_peak = $pagefilePeak
        pagefile_usage_mib_after = $pagefileAfter
        pagefile_growth_mib = $pagefileGrowth
        minimum_free_physical_required_bytes = $minimumFreePhysicalRequired
        memory_headroom_gate_pass = $memoryHeadroomPass
        swap_independence_gate_pass = $swapIndependencePass
        # Backward-compatible field name. The invariant is workload independence
        # from swap, not disabling a system-wide Windows safety mechanism.
        no_swap_gate_pass = $swapIndependencePass
        peak_working_set_bytes = $peakWorkingSet
        peak_private_bytes = $peakPrivateBytes
        minimum_free_physical_bytes = $minimumFreePhysical
        output = $model
    }
}

$runs = @()
if ($Mode -in @("Single", "Both")) {
    $arguments = "$Container $PromptTokenIds $NewTokens $RamCacheGiB $VramCacheGiB $WarmupRounds"
    $runs += Invoke-GateRun -Name "single" -Arguments $arguments `
        -RequiredTokensPerSecond $SingleTokensPerSecond
}
if ($Mode -in @("Batch", "Both")) {
    $arguments = "$Container --batch $BatchPromptTokenIds $NewTokens $Concurrency $RamCacheGiB $VramCacheGiB $WarmupRounds"
    $runs += Invoke-GateRun -Name "batch" -Arguments $arguments `
        -RequiredTokensPerSecond $AggregateTokensPerSecond
}

$overall = $true
foreach ($run in $runs) {
    $overall = $overall -and $run.speed_gate_pass -and
        $run.correctness_gate_pass -and $run.container_larger_than_ram -and
        $run.no_swap_gate_pass
}
$result = [PSCustomObject]@{
    schema_version = 2
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    model_id = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    mode = $Mode
    overall_pass = $overall
    runs = $runs
}
$path = Write-JsonArtifact -Value $result -Name "p6-gate-latest.json"
$result | ConvertTo-Json -Depth 10
Write-Output "P6 gate artifact: $path"
if (-not $overall) { exit 2 }
