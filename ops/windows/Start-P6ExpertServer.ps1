param(
    [string]$Container = "",
    [string]$Tokenizer = "",
    [string]$Runner = "",
    [string]$Python = "",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [int]$MaximumQueue = 8,
    [int]$MaximumContext = 4096,
    [int]$MaximumNewTokens = 512,
    [int]$WorkerCapacity = 4,
    [int]$WorkerRamCacheGiB = 48,
    [int]$WorkerVramCacheGiB = 18,
    [double]$MicrobatchWindowMs = 2.0,
    [int]$LatencyWindow = 4096,
    [double]$QueueTimeoutSeconds = 1.0,
    [double]$GenerationTimeoutSeconds = 120.0,
    [int]$StartupTimeoutSeconds = 600,
    [int]$DrainTimeoutSeconds = 30,
    [string]$BuildId = "development"
)

$arguments = @{
    HostAddress = $HostAddress
    Port = $Port
    MaximumQueue = $MaximumQueue
    MaximumContext = $MaximumContext
    MaximumNewTokens = $MaximumNewTokens
    WorkerCapacity = $WorkerCapacity
    WorkerRamCacheGiB = $WorkerRamCacheGiB
    WorkerVramCacheGiB = $WorkerVramCacheGiB
    MicrobatchWindowMs = $MicrobatchWindowMs
    LatencyWindow = $LatencyWindow
    QueueTimeoutSeconds = $QueueTimeoutSeconds
    GenerationTimeoutSeconds = $GenerationTimeoutSeconds
    StartupTimeoutSeconds = $StartupTimeoutSeconds
    DrainTimeoutSeconds = $DrainTimeoutSeconds
    BuildId = $BuildId
    ModelId = "qwen3-next-80b-a3b-expert-pack-int8"
}
if ($Container) { $arguments.Container = $Container }
if ($Tokenizer) { $arguments.Tokenizer = $Tokenizer }
if ($Runner) { $arguments.Runner = $Runner }
if ($Python) { $arguments.Python = $Python }

& (Join-Path $PSScriptRoot "Start-ExpertServer.ps1") @arguments
