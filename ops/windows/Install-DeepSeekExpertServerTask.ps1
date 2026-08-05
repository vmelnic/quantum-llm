param(
    [Parameter(Mandatory = $true)][string]$Bundle,
    [string]$TaskName = "QuantumLLM-DeepSeekV4Flash",
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
    [double]$MicrobatchWindowMs = 2.0,
    [int]$LatencyWindow = 4096,
    [double]$QueueTimeoutSeconds = 1.0,
    [double]$GenerationTimeoutSeconds = 600.0,
    [int]$StartupTimeoutSeconds = 600,
    [int]$DrainTimeoutSeconds = 30,
    [string]$BuildId = "development",
    [switch]$Start
)

$arguments = @{
    Profile = "DeepSeekV4Flash"
    TaskName = $TaskName
    Bundle = $Bundle
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
    MicrobatchWindowMs = $MicrobatchWindowMs
    LatencyWindow = $LatencyWindow
    QueueTimeoutSeconds = $QueueTimeoutSeconds
    GenerationTimeoutSeconds = $GenerationTimeoutSeconds
    StartupTimeoutSeconds = $StartupTimeoutSeconds
    DrainTimeoutSeconds = $DrainTimeoutSeconds
    BuildId = $BuildId
}
if ($Tokenizer) { $arguments.Tokenizer = $Tokenizer }
if ($Runner) { $arguments.Runner = $Runner }
if ($Python) { $arguments.Python = $Python }
if ($Start) { $arguments.Start = $true }

& (Join-Path $PSScriptRoot "Install-ExpertServerTask.ps1") @arguments
