param(
    [string]$Container = "",
    [string]$Tokenizer = "",
    [string]$Runner = "",
    [string]$Python = "",
    [string]$ModelId = "",
    [string]$HostAddress = "127.0.0.1",
    [string]$ApiKey = "",
    [int]$Port = 8080,
    [int]$MaximumQueue = 8,
    [int]$MaximumContext = 4096,
    [int]$MaximumNewTokens = 512,
    [int]$WorkerCapacity = 4,
    [int]$WorkerRamCacheGiB = 48,
    [int]$WorkerVramCacheGiB = 18,
    [ValidateSet("latency", "balanced", "capacity")]
    [string]$PlacementProfile = "balanced",
    [int]$WorkerKvCacheMiB = 2048,
    [int]$WorkerKvPageTokens = 256,
    [ValidateSet("artifact", "fp8-e4m3-per-head", "fp16")]
    [string]$WorkerKvCacheDtype = "artifact",
    [switch]$ProfileGpuPhases,
    [switch]$DisableRetainedRoute,
    [switch]$EnableCpuHybrid,
    [string]$WorkerRouteTraceFile = "",
    [ValidateRange(1, 65536)][int]$WorkerRouteTraceMaxSteps = 4096,
    [double]$MicrobatchWindowMs = 2.0,
    [int]$LatencyWindow = 4096,
    [double]$QueueTimeoutSeconds = 1.0,
    [double]$GenerationTimeoutSeconds = 120.0,
    [int]$MaximumBodyMiB = 16,
    [ValidateRange(65536, 16777216)][int]$MaximumImagePixels = 2097152,
    [ValidateRange(256, 32768)][int]$MaximumImagePatchTokens = 4096,
    [int]$StartupTimeoutSeconds = 600,
    [int]$DrainTimeoutSeconds = 30,
    [string]$BuildId = "development",
    [string]$LogFile = "",
    [string]$RawResponseTraceFile = ""
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if (-not $Container) { throw "Container is required" }
if (-not $Runner) { throw "Runner is required" }
if (-not $ModelId) { throw "ModelId is required" }
if (-not $Tokenizer) { $Tokenizer = Join-Path $Container "tokenizer" }
if (-not $LogFile) { $LogFile = Join-Path $script:RepoRoot "logs\expert-server.jsonl" }

$pythonCommand = $null
if ($Python) {
    if (Test-Path -LiteralPath $Python -PathType Leaf) {
        $pythonCommand = Get-Command ([System.IO.Path]::GetFullPath($Python)) -ErrorAction Stop
    } else {
        $pythonCommand = Get-Command $Python -ErrorAction Stop
    }
} else {
    foreach ($candidate in @(
        (Join-Path $script:RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $script:RepoRoot "work\venv\server\Scripts\python.exe")
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $pythonCommand = Get-Command $candidate -ErrorAction Stop
            break
        }
    }
    if ($null -eq $pythonCommand) { $pythonCommand = Get-PythonCommand }
}

$effectiveApiKey = if ($ApiKey) { $ApiKey } else { $env:EXPERT_API_KEY }
$loopback = $HostAddress -in @("127.0.0.1", "::1", "localhost")
if (-not $loopback -and [string]::IsNullOrWhiteSpace($effectiveApiKey)) {
    throw "EXPERT_API_KEY is required for a non-loopback bind"
}

$worker = [System.IO.Path]::GetFullPath($Runner)
$containerPath = [System.IO.Path]::GetFullPath($Container)
$tokenizerPath = [System.IO.Path]::GetFullPath($Tokenizer)
$server = Join-Path $script:RepoRoot "ops\python\expert_server.py"
if (-not (Test-Path $worker -PathType Leaf)) { throw "CUDA worker missing: $worker" }
if (-not (Test-Path $containerPath -PathType Container)) { throw "Container missing: $containerPath" }
if (-not (Test-Path $tokenizerPath -PathType Container)) { throw "Tokenizer missing: $tokenizerPath" }
[string[]]$profileArguments = if ($ProfileGpuPhases) {
    "--profile-gpu-phases"
} else { @() }
[string[]]$retainedRouteArguments = if ($DisableRetainedRoute) {
    "--disable-worker-retained-route"
} else { @() }
[string[]]$cpuHybridArguments = if ($EnableCpuHybrid) {
    "--enable-worker-cpu-hybrid"
} else { @() }
[string[]]$routeTraceArguments = if ($WorkerRouteTraceFile) {
    @(
        "--worker-route-trace-file", ([System.IO.Path]::GetFullPath($WorkerRouteTraceFile)),
        "--worker-route-trace-max-steps", [string]$WorkerRouteTraceMaxSteps
    )
} else { @() }
[string[]]$apiKeyArguments = if ($effectiveApiKey) {
    @("--api-key", $effectiveApiKey)
} else { @() }
[string[]]$responseTraceArguments = if ($RawResponseTraceFile) {
    @("--raw-response-trace-file", ([System.IO.Path]::GetFullPath(
        $RawResponseTraceFile)))
} else { @() }

& $pythonCommand.Source $server `
    --worker $worker `
    --container $containerPath `
    --tokenizer $tokenizerPath `
    --model $ModelId `
    --host $HostAddress `
    @apiKeyArguments `
    --port $Port `
    --maximum-queue $MaximumQueue `
    --max-context $MaximumContext `
    --maximum-new-tokens $MaximumNewTokens `
    --worker-capacity $WorkerCapacity `
    --worker-ram-cache-gib $WorkerRamCacheGiB `
    --worker-vram-cache-gib $WorkerVramCacheGiB `
    --placement-profile $PlacementProfile `
    --worker-kv-cache-mib $WorkerKvCacheMiB `
    --worker-kv-page-tokens $WorkerKvPageTokens `
    --worker-kv-cache-dtype $WorkerKvCacheDtype `
    @profileArguments `
    @retainedRouteArguments `
    @cpuHybridArguments `
    @routeTraceArguments `
    @responseTraceArguments `
    --microbatch-window-ms $MicrobatchWindowMs `
    --latency-window $LatencyWindow `
    --queue-timeout $QueueTimeoutSeconds `
    --generation-timeout $GenerationTimeoutSeconds `
    --maximum-body-bytes ([int64]$MaximumBodyMiB * 1MB) `
    --maximum-image-pixels $MaximumImagePixels `
    --maximum-image-patch-tokens $MaximumImagePatchTokens `
    --startup-timeout $StartupTimeoutSeconds `
    --drain-timeout $DrainTimeoutSeconds `
    --build-id $BuildId `
    --log-file $LogFile
if ($LASTEXITCODE -ne 0) { throw "Expert server exited with code $LASTEXITCODE" }
