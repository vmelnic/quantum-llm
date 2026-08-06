param(
    [string]$Container = "",
    [string]$Tokenizer = "",
    [string]$Runner = "",
    [string]$Python = "",
    [string]$ModelId = "qwen3-next-80b-a3b-expert-pack-int8",
    [string]$HostAddress = "127.0.0.1",
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
    [bool]$ProfileGpuPhases = $false,
    [bool]$EnableMtp = $false,
    [double]$MicrobatchWindowMs = 2.0,
    [int]$LatencyWindow = 4096,
    [double]$QueueTimeoutSeconds = 1.0,
    [double]$GenerationTimeoutSeconds = 120.0,
    [int]$StartupTimeoutSeconds = 600,
    [int]$DrainTimeoutSeconds = 30,
    [string]$BuildId = "development",
    [string]$LogFile = ""
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if (-not $Container) {
    $Container = Join-Path $script:RepoRoot "work\models\qwen3-next-80b-expert-pack-int8"
}
if (-not $Runner) {
    $Runner = Join-Path $script:RepoRoot `
        "out\build\windows-msvc-release\runtime\Release\expert-qwen3-next-runner.exe"
}
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

$loopback = $HostAddress -in @("127.0.0.1", "::1", "localhost")
if (-not $loopback -and [string]::IsNullOrWhiteSpace($env:EXPERT_API_KEY)) {
    throw "EXPERT_API_KEY is required for a non-loopback bind"
}

$worker = [System.IO.Path]::GetFullPath($Runner)
$containerPath = [System.IO.Path]::GetFullPath($Container)
$tokenizerPath = [System.IO.Path]::GetFullPath($Tokenizer)
$server = Join-Path $script:RepoRoot "ops\python\expert_server.py"
if (-not (Test-Path $worker -PathType Leaf)) { throw "CUDA worker missing: $worker" }
if (-not (Test-Path $containerPath -PathType Container)) { throw "Container missing: $containerPath" }
if (-not (Test-Path $tokenizerPath -PathType Container)) { throw "Tokenizer missing: $tokenizerPath" }
$profileArguments = if ($ProfileGpuPhases) { @("--profile-gpu-phases") } else { @() }
$mtpArguments = if ($EnableMtp) { @("--enable-mtp") } else { @() }

& $pythonCommand.Source $server `
    --worker $worker `
    --container $containerPath `
    --tokenizer $tokenizerPath `
    --model $ModelId `
    --host $HostAddress `
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
    @profileArguments `
    @mtpArguments `
    --microbatch-window-ms $MicrobatchWindowMs `
    --latency-window $LatencyWindow `
    --queue-timeout $QueueTimeoutSeconds `
    --generation-timeout $GenerationTimeoutSeconds `
    --startup-timeout $StartupTimeoutSeconds `
    --drain-timeout $DrainTimeoutSeconds `
    --build-id $BuildId `
    --log-file $LogFile
if ($LASTEXITCODE -ne 0) { throw "Expert server exited with code $LASTEXITCODE" }
