param(
    [Parameter(Mandatory = $true)][string]$Container,
    [string]$Runner = "",
    [string]$TokenFile = "",
    [string]$PromptTokenIds = "",
    [int]$GeneratedTokens = 128,
    [int]$MaximumContext = 262144,
    [string]$FractionsPpm = "250000,375000",
    [switch]$SkipRankPreview,
    [int]$RamCacheGiB = 48,
    [int]$VramCacheGiB = 13,
    [int]$KvCacheMiB = 512,
    [int]$KvPageTokens = 256,
    [ValidateSet("artifact", "fp8-e4m3-per-head")]
    [string]$KvCacheDtype = "artifact",
    [switch]$DisableMtp,
    [ValidateSet("latency", "balanced", "capacity")]
    [string]$PlacementProfile = "balanced",
    [string]$TraceFile = "",
    [string]$TreeFile = "",
    [int]$TreeHorizon = 32,
    [int]$TreeBeamWidth = 4,
    [int]$BlockJacobiIterations = 0,
    [string]$KvProfileFile = "",
    [string]$ResultFile = ""
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if (-not $Runner) {
    $Runner = Join-Path $script:RepoRoot `
        "out\build\windows-msvc-release\runtime\Release\expert-moe-vm-runner.exe"
}
if (-not $TraceFile) {
    $TraceFile = Join-Path $script:RepoRoot "artifacts\exact-tier-gate.tsv"
}
if (-not $ResultFile) {
    $ResultFile = Join-Path $script:RepoRoot "artifacts\exact-tier-gate-run.json"
}
if ([string]::IsNullOrWhiteSpace($TokenFile) -eq `
        [string]::IsNullOrWhiteSpace($PromptTokenIds)) {
    throw "Pass exactly one of TokenFile or PromptTokenIds"
}
foreach ($path in @($Container, $Runner)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path missing: $path" }
}
if ($TokenFile) {
    if (-not (Test-Path -LiteralPath $TokenFile -PathType Leaf)) {
        throw "Token file missing: $TokenFile"
    }
    $PromptTokenIds = (Get-Content -LiteralPath $TokenFile -Raw).Trim()
}
if ($PromptTokenIds -notmatch '^\d+(,\d+)+$') {
    throw "Prompt token IDs must be a comma-separated integer list"
}
$promptTokens = ([regex]::Matches($PromptTokenIds, ',')).Count + 1
if ($GeneratedTokens -lt 2 -or $MaximumContext -lt 2 -or
    $promptTokens + $GeneratedTokens -gt $MaximumContext) {
    throw "Prompt and generated tokens exceed the context contract"
}
$rankFractions = if ($SkipRankPreview) { "" } else { $FractionsPpm }
if ($rankFractions -and $rankFractions -notmatch '^\d+(,\d+)*$') {
    throw "FractionsPpm must be a comma-separated integer list"
}
if ($TreeFile -and ($TreeHorizon -lt 1 -or $TreeBeamWidth -lt 1 -or
        $TreeHorizon * $TreeBeamWidth -gt 128)) {
    throw "TreeHorizon * TreeBeamWidth must be between 1 and 128"
}
if ($BlockJacobiIterations -lt 0 -or
    ($BlockJacobiIterations -gt 0 -and
        (-not $TreeFile -or $TreeBeamWidth -ne 1 -or
            $TreeHorizon * ($BlockJacobiIterations + 1) -gt 128))) {
    throw "Block Jacobi requires a tree, beam width 1, and at most 128 trajectory nodes"
}

$containerPath = [System.IO.Path]::GetFullPath($Container)
$runnerPath = [System.IO.Path]::GetFullPath($Runner)
$tracePath = if ($rankFractions) {
    [System.IO.Path]::GetFullPath($TraceFile)
} else { "" }
$treePath = if ($TreeFile) { [System.IO.Path]::GetFullPath($TreeFile) } else { "" }
$kvProfilePath = if ($KvProfileFile) {
    [System.IO.Path]::GetFullPath($KvProfileFile)
} else { "" }
$resultPath = [System.IO.Path]::GetFullPath($ResultFile)
$stdoutPath = Join-Path $script:RepoRoot "logs\exact-tier-gate.stdout.log"
$stderrPath = Join-Path $script:RepoRoot "logs\exact-tier-gate.stderr.log"
if ($tracePath) {
    $env:QUANTUM_LLM_EXACT_TIER_GATE_OUTPUT = $tracePath
    $env:QUANTUM_LLM_EXACT_TIER_GATE_FRACTIONS_PPM = $rankFractions
}
if ($treePath) {
    $env:QUANTUM_LLM_EXACT_TIER_TREE_OUTPUT = $treePath
    $env:QUANTUM_LLM_EXACT_TIER_TREE_HORIZON = $TreeHorizon
    $env:QUANTUM_LLM_EXACT_TIER_TREE_BEAM_WIDTH = $TreeBeamWidth
}
if ($BlockJacobiIterations -gt 0) {
    $env:QUANTUM_LLM_EXACT_TIER_BLOCK_JACOBI_ITERATIONS = `
        $BlockJacobiIterations
}
if ($kvProfilePath) {
    $env:QUANTUM_LLM_EXACT_KV_PROFILE_OUTPUT = $kvProfilePath
}

function Quote-NativeArgument([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}

$arguments = @(
    (Quote-NativeArgument $containerPath), "--worker",
    "--max-context=$MaximumContext", "--capacity=1",
    "--ram-cache-gib=$RamCacheGiB", "--vram-cache-gib=$VramCacheGiB",
    "--kv-cache-mib=$KvCacheMiB", "--kv-page-tokens=$KvPageTokens",
    "--kv-cache-dtype=$KvCacheDtype",
    "--placement-profile=$PlacementProfile"
) -join ' '
$startInfo = [Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $runnerPath
$startInfo.Arguments = $arguments
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardInput = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.CreateNoWindow = $true
$process = [Diagnostics.Process]::new()
$process.StartInfo = $startInfo
$started = [DateTime]::UtcNow
if (-not $process.Start()) { throw "Failed to start exact-tier worker" }
$stderrTask = $process.StandardError.ReadToEndAsync()
$transcript = [Collections.Generic.List[string]]::new()

function Read-WorkerResponse([string]$ExpectedType) {
    $line = $process.StandardOutput.ReadLine()
    if ($null -eq $line) {
        throw "Worker closed stdout while waiting for $ExpectedType"
    }
    $transcript.Add($line)
    $response = $line | ConvertFrom-Json -ErrorAction Stop
    if ($response.type -eq "error") { throw "Worker error: $($response.message)" }
    if ($response.type -ne $ExpectedType) {
        throw "Expected worker response $ExpectedType, got $($response.type)"
    }
    return $response
}

try {
    $ready = Read-WorkerResponse "ready"
    $prefillStarted = [DateTime]::UtcNow
    $process.StandardInput.WriteLine("BEGIN`t1`t$MaximumContext`t$PromptTokenIds")
    $process.StandardInput.Flush()
    $null = Read-WorkerResponse "begun"
    $prefillSeconds = ([DateTime]::UtcNow - $prefillStarted).TotalSeconds

    $decodeStarted = [DateTime]::UtcNow
    $emitted = 0
    $emittedTokenIds = [Collections.Generic.List[UInt32]]::new()
    $nextMode = if ($DisableMtp) { 2 } else { 0 }
    for ($index = 0; $index -lt $GeneratedTokens; $index++) {
        $process.StandardInput.WriteLine("NEXT`t1`t$nextMode")
        $process.StandardInput.Flush()
        $response = Read-WorkerResponse "token"
        $emitted += @($response.tokens).Count
        foreach ($token in @($response.tokens)) {
            $emittedTokenIds.Add([UInt32]$token)
        }
    }
    $decodeSeconds = ([DateTime]::UtcNow - $decodeStarted).TotalSeconds
    $process.StandardInput.WriteLine("STATS")
    $process.StandardInput.Flush()
    $stats = Read-WorkerResponse "stats"
    $process.StandardInput.WriteLine("END`t1")
    $process.StandardInput.Flush()
    $null = Read-WorkerResponse "ended"
    $kvProfile = $null
    if ($kvProfilePath) {
        if (-not (Test-Path -LiteralPath $kvProfilePath -PathType Leaf)) {
            throw "Exact FP16 KV profile was not produced"
        }
        $kvProfile = Get-Content -LiteralPath $kvProfilePath -Raw |
            ConvertFrom-Json -ErrorAction Stop
        if ($kvProfile.format -ne "fp16-kv-lossless-profile-v1" -or
            [int]$kvProfile.populated_tokens -ne ($promptTokens + $emitted)) {
            throw "Exact FP16 KV profile has an invalid contract"
        }
    }
    $process.StandardInput.WriteLine("SHUTDOWN")
    $process.StandardInput.Flush()
    $null = Read-WorkerResponse "shutdown"
    $process.StandardInput.Close()
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        throw "Exact-tier worker exited with code $($process.ExitCode)"
    }
    $traceRecords = 0
    $verifiedCalls = [Math]::Min(
        $GeneratedTokens, [Math]::Max(0, $MaximumContext - $promptTokens - 1))
    if ($tracePath) {
        if (-not (Test-Path -LiteralPath $tracePath -PathType Leaf)) {
            throw "Exact-tier trace was not produced"
        }
        $traceRecords = [Math]::Max(
            0, (Get-Content -LiteralPath $tracePath).Count - 1)
        $requiredRecords = $verifiedCalls * $rankFractions.Split(',').Count
        if ($traceRecords -ne $requiredRecords) {
            throw "Exact-tier trace has $traceRecords records; expected $requiredRecords"
        }
    }
    $treeRecords = 0
    if ($treePath) {
        if (-not (Test-Path -LiteralPath $treePath -PathType Leaf)) {
            throw "Exact-tier tree trace was not produced"
        }
        $treeLines = @(Get-Content -LiteralPath $treePath)
        if ($treeLines.Count -lt 2 -or $treeLines[0] -ne
                "start_position`thorizon`tbeam_width`tnodes`tproposer_ns`tpeak_host_bytes`taccepted_depth`ttermination") {
            throw "Exact-tier tree trace is empty or has an invalid header"
        }
        $treeRecords = $treeLines.Count - 1
    }
    $result = [ordered]@{
        schema_version = 1
        format = "exact-tier-gate-run-v1"
        started_utc = $started.ToString("o")
        container = $containerPath
        prompt_tokens = $promptTokens
        generated_calls = $GeneratedTokens
        emitted_tokens = $emitted
        emitted_token_ids = @($emittedTokenIds)
        maximum_context = $MaximumContext
        kv_cache_dtype_requested = $KvCacheDtype
        mtp_enabled_for_decode = -not $DisableMtp
        fractions_ppm = if ($rankFractions) {
            @($rankFractions.Split(',') | ForEach-Object { [int]$_ })
        } else { @() }
        prefill_seconds = $prefillSeconds
        decode_seconds = $decodeSeconds
        trace_records = $traceRecords
        trace_file = $tracePath
        tree_file = $treePath
        tree_horizon = if ($treePath) { $TreeHorizon } else { 0 }
        tree_beam_width = if ($treePath) { $TreeBeamWidth } else { 0 }
        block_jacobi_iterations = $BlockJacobiIterations
        kv_profile_file = $kvProfilePath
        kv_profile = $kvProfile
        tree_records = $treeRecords
        ready = $ready
        stats = $stats
    }
    $result | ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath $resultPath -Encoding UTF8
    $transcript | Set-Content -LiteralPath $stdoutPath -Encoding UTF8
    $stderrTask.Result | Set-Content -LiteralPath $stderrPath -Encoding UTF8
    $result | ConvertTo-Json -Depth 12
}
finally {
    if (-not $process.HasExited) { $process.Kill() }
    $env:QUANTUM_LLM_EXACT_TIER_GATE_OUTPUT = $null
    $env:QUANTUM_LLM_EXACT_TIER_GATE_FRACTIONS_PPM = $null
    $env:QUANTUM_LLM_EXACT_TIER_TREE_OUTPUT = $null
    $env:QUANTUM_LLM_EXACT_TIER_TREE_HORIZON = $null
    $env:QUANTUM_LLM_EXACT_TIER_TREE_BEAM_WIDTH = $null
    $env:QUANTUM_LLM_EXACT_TIER_BLOCK_JACOBI_ITERATIONS = $null
    $env:QUANTUM_LLM_EXACT_KV_PROFILE_OUTPUT = $null
}
