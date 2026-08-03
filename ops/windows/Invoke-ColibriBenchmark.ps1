param(
    [string]$CacheSizes,
    [string]$PilotModes,
    [string]$DirectModes,
    [int]$Repeats = 1,
    [int]$Threads = 6,
    [ValidateSet(0, 1)][int]$RamCache = 0,
    [string]$ReferencePath,
    [switch]$UseIndependentOracle
)

. (Join-Path $PSScriptRoot "Common.ps1")

Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig
$checkout = Resolve-RepositoryPath -Path ([string]$config.colibri.checkout)
$model = Resolve-RepositoryPath -Path ([string]$config.colibri.converted_model)
$engine = Join-Path $checkout "c\olmoe.exe"
if (-not (Test-Path -LiteralPath $engine -PathType Leaf)) {
    throw "Colibri OLMoE engine is missing: $engine"
}
if (-not (Test-Path -LiteralPath $model -PathType Container)) {
    throw "Converted Colibri model is missing: $model"
}

if ($UseIndependentOracle) {
    $ReferencePath = Join-Path (Join-Path $script:RepoRoot "artifacts") "olmoe-oracle-latest.json"
}
elseif ([string]::IsNullOrWhiteSpace($ReferencePath)) {
    $ReferencePath = Resolve-RepositoryPath -Path ([string]$config.colibri.reference)
}
else {
    $ReferencePath = Resolve-RepositoryPath -Path $ReferencePath
}
if (-not (Test-Path -LiteralPath $ReferencePath -PathType Leaf)) {
    throw "Reference JSON is missing: $ReferencePath"
}
$reference = Get-Content -LiteralPath $ReferencePath -Raw | ConvertFrom-Json
$promptCount = @($reference.prompt_ids).Count
$fullCount = @($reference.full_ids).Count
$expectedTokens = $fullCount - $promptCount
if ($promptCount -lt 1 -or $expectedTokens -lt 1) {
    throw "Reference JSON must contain non-empty prompt_ids and a longer full_ids array."
}

function Convert-IntegerList {
    param(
        [string]$Text,
        [object[]]$Default,
        [int]$Minimum,
        [int]$Maximum,
        [string]$Name
    )
    $values = @(
        if ([string]::IsNullOrWhiteSpace($Text)) {
            $Default | ForEach-Object { [int]$_ }
        }
        else {
            $Text.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries) |
                ForEach-Object { [int]$_.Trim() }
        }
    )
    if ($values.Count -eq 0) {
        throw "$Name must not be empty."
    }
    foreach ($value in $values) {
        if ($value -lt $Minimum -or $value -gt $Maximum) {
            throw "$Name value $value is outside [$Minimum, $Maximum]."
        }
    }
    return @($values | Select-Object -Unique)
}

$cacheValues = @(Convert-IntegerList -Text $CacheSizes `
    -Default @($config.colibri.default_cache_sizes) -Minimum 1 -Maximum 64 -Name "CacheSizes")
$pilotValues = @(Convert-IntegerList -Text $PilotModes `
    -Default @($config.colibri.default_pilot_modes) -Minimum 0 -Maximum 3 -Name "PilotModes")
$directValues = @(Convert-IntegerList -Text $DirectModes `
    -Default @($config.colibri.default_direct_modes) -Minimum 0 -Maximum 1 -Name "DirectModes")
if ($Repeats -lt 1 -or $Repeats -gt 20) {
    throw "Repeats must be between 1 and 20."
}
if ($Threads -lt 1) {
    throw "Threads must be positive."
}

$pinsPath = Join-Path $model "hot_pinned.bin"
if (Test-Path -LiteralPath $pinsPath) {
    throw "Persistent hot pins would invalidate a controlled sweep: $pinsPath"
}

$containerReportPath = Join-Path (Join-Path $script:RepoRoot "artifacts") "colibri-container-latest.json"
if (-not (Test-Path -LiteralPath $containerReportPath -PathType Leaf)) {
    throw "Validated container report is missing: $containerReportPath"
}
$containerReport = Get-Content -LiteralPath $containerReportPath -Raw | ConvertFrom-Json
if ([string]$containerReport.status -ne "pass") {
    throw "Converted container has not passed validation: $containerReportPath"
}
$expertPayloadBytes = [int64]$containerReport.container.expert_payload_bytes_max
$expectedRamCacheExperts = [int64]$containerReport.container.expert_count_expected

$git = Get-Command "git.exe" -ErrorAction Stop
$colibriCommit = (& $git.Source -C $checkout rev-parse HEAD | Out-String).Trim()
if ($colibriCommit -ne [string]$config.colibri.commit) {
    throw "Colibri checkout commit drift: $colibriCommit"
}

function Get-RegexGroup {
    param([string]$Text, [string]$Pattern, [int]$Group = 1)
    $match = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if (-not $match.Success) { return $null }
    return $match.Groups[$Group].Value
}

function Convert-OptionalDouble {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    return [double]::Parse($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Convert-OptionalInt64 {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    return [int64]::Parse($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$runs = @()
$runIndex = 0
$savedEnvironment = @{}
foreach ($name in @(
    "SNAP", "PILOT", "WIDE", "PILOT_EVICT_GUARD", "EXPERT_DROP", "HOT",
    "WARMUP", "OMP_NUM_THREADS", "DIRECT", "RAMCACHE", "CHAT", "PPL"
)) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    $env:SNAP = $model
    $env:WIDE = "1"
    $env:PILOT_EVICT_GUARD = "1"
    $env:EXPERT_DROP = "1"
    $env:HOT = "0"
    $env:WARMUP = "2147483647"
    $env:OMP_NUM_THREADS = [string]$Threads
    $env:RAMCACHE = [string]$RamCache
    Remove-Item Env:CHAT -ErrorAction SilentlyContinue
    Remove-Item Env:PPL -ErrorAction SilentlyContinue

    foreach ($direct in $directValues) {
        $env:DIRECT = [string]$direct
        foreach ($pilot in $pilotValues) {
            foreach ($cache in $cacheValues) {
                foreach ($repeat in 1..$Repeats) {
                $runIndex++
                $env:PILOT = [string]$pilot
                $runName = "colibri-run-$stamp-$('{0:d2}' -f $runIndex)-rc$RamCache-d$direct-c$cache-p$pilot-r$repeat"
                $logPath = Join-Path (Join-Path $script:RepoRoot "logs") "$runName.log"
                $started = [DateTime]::UtcNow
                $rawLines = @(& $engine ([string]$cache) ([string]$config.colibri.quant_bits) $ReferencePath 2>&1)
                $exitCode = $LASTEXITCODE
                $finished = [DateTime]::UtcNow
                $rawText = ($rawLines | ForEach-Object { [string]$_ }) -join [Environment]::NewLine
                $rawText | Set-Content -LiteralPath $logPath -Encoding UTF8

                $matching = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern "Matching tokens:\s*(\d+)/(\d+)" -Group 1)
                $matchTotal = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern "Matching tokens:\s*(\d+)/(\d+)" -Group 2)
                $hits = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern "Expert cache hit rate:\s*[0-9.]+%\s+\(hit=(\d+) miss=(\d+)\)" -Group 1)
                $misses = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern "Expert cache hit rate:\s*[0-9.]+%\s+\(hit=(\d+) miss=(\d+)\)" -Group 2)
                $loadPattern = "Expert loads:\s*(\d+)\s+payload_bytes=(\d+)\s+io_bytes=(\d+)\s+direct_loads=(\d+)\s+direct_bytes=(\d+)\s+prefetch_loads=(\d+)"
                $loads = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern $loadPattern -Group 1)
                $loadBytes = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern $loadPattern -Group 2)
                $ioBytes = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern $loadPattern -Group 3)
                $directLoads = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern $loadPattern -Group 4)
                $directBytes = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern $loadPattern -Group 5)
                $prefetchLoads = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern $loadPattern -Group 6)
                $speed = Convert-OptionalDouble (Get-RegexGroup -Text $rawText -Pattern "Speed:\s*([0-9.]+) tok/s")
                $engineSeconds = Convert-OptionalDouble (Get-RegexGroup -Text $rawText -Pattern "Speed:\s*[0-9.]+ tok/s \(([0-9.]+)s for")
                $rss = Convert-OptionalDouble (Get-RegexGroup -Text $rawText -Pattern "PEAK RSS:\s*([0-9.]+) GB")
                $hitRate = Convert-OptionalDouble (Get-RegexGroup -Text $rawText -Pattern "Expert cache hit rate:\s*([0-9.]+)%")
                $loadSeconds = Convert-OptionalDouble (Get-RegexGroup -Text $rawText -Pattern "resident weights loaded in ([0-9.]+)s")
                $ramCacheExperts = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern "\[RAMCACHE\] ready experts=(\d+)" -Group 1)
                $ramCacheBytes = Convert-OptionalInt64 (Get-RegexGroup -Text $rawText -Pattern "\[RAMCACHE\] ready experts=\d+ payload_bytes=(\d+)" -Group 1)
                $ramCacheSeconds = Convert-OptionalDouble (Get-RegexGroup -Text $rawText -Pattern "\[RAMCACHE\] ready experts=\d+ payload_bytes=\d+ load_seconds=([0-9.]+)" -Group 1)
                $ramCacheRss = Convert-OptionalDouble (Get-RegexGroup -Text $rawText -Pattern "\[RAMCACHE\] ready experts=\d+ payload_bytes=\d+ load_seconds=[0-9.]+ rss_gb=([0-9.]+)" -Group 1)
                $runPassed = (
                    $exitCode -eq 0 -and $null -ne $matching -and $null -ne $matchTotal -and
                    $matching -eq $matchTotal -and $matchTotal -eq $expectedTokens -and
                    $null -ne $loads -and $null -ne $loadBytes -and $null -ne $ioBytes -and
                    ($RamCache -eq 0 -or (
                        $ramCacheExperts -eq $expectedRamCacheExperts -and
                        $loads -eq 0 -and $misses -eq 0
                    )) -and
                    ($RamCache -eq 1 -or $direct -eq 0 -or (
                        $null -ne $directLoads -and $directLoads -gt 0 -and
                        $directLoads -le $loads -and $directBytes -gt 0
                    ))
                )

                $runs += [PSCustomObject]@{
                    run_index = $runIndex
                    status = if ($runPassed) { "pass" } else { "fail" }
                    cache_experts_per_layer = $cache
                    pilot = $pilot
                    direct = $direct
                    ram_cache = $RamCache
                    repeat = $repeat
                    quant_bits = [int]$config.colibri.quant_bits
                    threads = $Threads
                    exit_code = $exitCode
                    matching_tokens = $matching
                    expected_tokens = $matchTotal
                    token_exact = ($null -ne $matching -and $matching -eq $matchTotal)
                    speed_tokens_per_second = $speed
                    engine_seconds = $engineSeconds
                    wall_seconds = ($finished - $started).TotalSeconds
                    resident_load_seconds = $loadSeconds
                    ram_cache_experts = $ramCacheExperts
                    ram_cache_payload_bytes = $ramCacheBytes
                    ram_cache_load_seconds = $ramCacheSeconds
                    ram_cache_ready_rss_gb = $ramCacheRss
                    peak_rss_gb = $rss
                    cache_hit_rate_percent = $hitRate
                    demand_hits = $hits
                    demand_misses = $misses
                    expert_loads_total = $loads
                    expert_prefetch_loads = $prefetchLoads
                    expert_payload_bytes_total = $loadBytes
                    expert_io_bytes_total = $ioBytes
                    expert_direct_loads = $directLoads
                    expert_direct_bytes = $directBytes
                    demand_miss_bytes_estimate = if ($null -ne $misses) { $misses * $expertPayloadBytes } else { $null }
                    physical_disk_read_bytes = $null
                    os_file_cache_state = if ($RamCache -eq 1) { "not_used_for_experts_during_decode" } elseif ($direct -eq 1) { "bypassed_when_aligned_with_buffered_tail_fallback" } else { "uncontrolled" }
                    log = $logPath
                }
                Write-Output "[$runIndex] direct=$direct cache=$cache pilot=$pilot repeat=$repeat status=$(if ($runPassed) { 'PASS' } else { 'FAIL' })"
                }
            }
        }
    }
}
finally {
    foreach ($name in $savedEnvironment.Keys) {
        $value = $savedEnvironment[$name]
        if ($null -eq $value) {
            [Environment]::SetEnvironmentVariable($name, $null, "Process")
        }
        else {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

$failedRuns = @($runs | Where-Object status -ne "pass")
$summary = [PSCustomObject]@{
    schema_version = 1
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    status = if ($failedRuns.Count -eq 0) { "pass" } else { "fail" }
    experiment = "colibri-olmoe-exact-streaming-baseline"
    machine = [string]$config.machine_name
    cpu_only = $true
    engine = [PSCustomObject]@{
        path = $engine
        sha256 = Get-Sha256 -Path $engine
        repository = [string]$config.colibri.repository
        commit = $colibriCommit
        metrics_patch_sha256 = Get-Sha256 -Path (Join-Path $script:OpsRoot "patches\colibri-olmoe-load-metrics.patch")
    }
    model = [PSCustomObject]@{
        path = $model
        source_model = [string]$config.models.recommended_lab_moe
        source_revision = [string]$config.models.recommended_lab_revision
        container_report = $containerReportPath
        expert_payload_bytes = $expertPayloadBytes
    }
    reference = [PSCustomObject]@{
        path = $ReferencePath
        sha256 = Get-Sha256 -Path $ReferencePath
        prompt_tokens = $promptCount
        generated_tokens = $expectedTokens
        independent_teacher = [bool]$UseIndependentOracle
    }
    controls = [PSCustomObject]@{
        cache_sizes = $cacheValues
        pilot_modes = $pilotValues
        direct_modes = $directValues
        repeats = $Repeats
        threads = $Threads
        ram_cache = $RamCache
        hot_pinning = 0
        warmup_tokens = 2147483647
        expert_drop = 1
        process_lru_controlled = $true
        os_file_cache_controlled = $false
        expert_weight_file_cache_bypassed = ($directValues.Count -eq 1 -and $directValues[0] -eq 1)
        caveat = if ($RamCache -eq 1) { "RAMCACHE=1 preloads all expert weights once and performs no expert file reads during decode." } else { "DIRECT=1 bypasses the OS file cache for aligned expert weights. Weights whose aligned range crosses shard EOF, scale arrays, and dense tensors use buffered I/O." }
    }
    runs = $runs
}
$versioned = Write-JsonArtifact -Value $summary -Name "colibri-benchmark-$stamp.json"
$latest = Write-JsonArtifact -Value $summary -Name "colibri-benchmark-latest.json"
Write-Output "Benchmark artifact: $versioned"
Write-Output "Latest benchmark: $latest"
if ($failedRuns.Count -gt 0) {
    throw "$($failedRuns.Count) Colibri benchmark run(s) failed validation."
}
