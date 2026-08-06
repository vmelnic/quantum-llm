param(
    [Parameter(Mandatory = $true)][string]$Bundle,
    [string]$Tokenizer = "",
    [string]$Runner = "",
    [string]$Python = "",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [int]$MaximumQueue = 4,
    [int]$MaximumContext = 4096,
    [int]$MaximumNewTokens = 512,
    [ValidateRange(1, 4)][int]$WorkerCapacity = 2,
    [ValidateRange(2, 52)][int]$WorkerRamCacheGiB = 40,
    [ValidateRange(2, 14)][int]$WorkerVramCacheGiB = 12,
    [ValidateSet("latency", "balanced", "capacity")]
    [string]$PlacementProfile = "balanced",
    [int]$WorkerKvCacheMiB = 2048,
    [int]$WorkerKvPageTokens = 256,
    [bool]$ProfileGpuPhases = $false,
    [double]$MicrobatchWindowMs = 2.0,
    [int]$LatencyWindow = 4096,
    [double]$QueueTimeoutSeconds = 1.0,
    [double]$GenerationTimeoutSeconds = 600.0,
    [int]$StartupTimeoutSeconds = 600,
    [int]$DrainTimeoutSeconds = 30,
    [string]$BuildId = "development",
    [string]$LogFile = ""
)

$root = [System.IO.Path]::GetFullPath($Bundle)
$runtime = Join-Path $root "runtime.tsv"
if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) {
    throw "DeepSeek runtime bundle is missing runtime.tsv"
}
$fields = @{}
foreach ($line in @(Get-Content -LiteralPath $runtime | Select-Object -Skip 1)) {
    $parts = $line -split "`t", 2
    if ($parts.Count -eq 2) { $fields[$parts[0]] = $parts[1] }
}
if (-not $Tokenizer) { $Tokenizer = [string]$fields["checkpoint"] }
if (-not $Runner) {
    $repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $Runner = Join-Path $repoRoot `
        "out\build\windows-msvc-release\runtime\Release\expert-deepseek-worker.exe"
}

$arguments = @{
    Container = $root
    Tokenizer = $Tokenizer
    Runner = $Runner
    ModelId = "deepseek-v4-flash"
    HostAddress = $HostAddress
    Port = $Port
    MaximumQueue = $MaximumQueue
    MaximumContext = $MaximumContext
    MaximumNewTokens = $MaximumNewTokens
    WorkerCapacity = $WorkerCapacity
    WorkerRamCacheGiB = $WorkerRamCacheGiB
    WorkerVramCacheGiB = $WorkerVramCacheGiB
    PlacementProfile = $PlacementProfile
    WorkerKvCacheMiB = $WorkerKvCacheMiB
    WorkerKvPageTokens = $WorkerKvPageTokens
    ProfileGpuPhases = $ProfileGpuPhases
    MicrobatchWindowMs = $MicrobatchWindowMs
    LatencyWindow = $LatencyWindow
    QueueTimeoutSeconds = $QueueTimeoutSeconds
    GenerationTimeoutSeconds = $GenerationTimeoutSeconds
    StartupTimeoutSeconds = $StartupTimeoutSeconds
    DrainTimeoutSeconds = $DrainTimeoutSeconds
    BuildId = $BuildId
}
if ($Python) { $arguments.Python = $Python }
if ($LogFile) { $arguments.LogFile = $LogFile }

& (Join-Path $PSScriptRoot "Start-ExpertServer.ps1") @arguments
