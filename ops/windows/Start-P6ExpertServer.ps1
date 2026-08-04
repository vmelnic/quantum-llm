param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [int]$MaximumQueue = 8,
    [int]$MaximumContext = 4096,
    [int]$WorkerCapacity = 4,
    [int]$WorkerRamCacheGiB = 48,
    [int]$WorkerVramCacheGiB = 14,
    [double]$MicrobatchWindowMs = 2.0,
    [int]$StartupTimeoutSeconds = 600,
    [string]$BuildId = "development"
)

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$container = Join-Path $repoRoot "work\models\qwen3-next-80b-expert-pack-int8"
$runner = Join-Path $repoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-qwen3-next-runner.exe"

& (Join-Path $PSScriptRoot "Start-ExpertServer.ps1") `
    -Container $container `
    -Runner $runner `
    -ModelId "qwen3-next-80b-a3b-expert-pack-int8" `
    -HostAddress $HostAddress `
    -Port $Port `
    -MaximumQueue $MaximumQueue `
    -MaximumContext $MaximumContext `
    -WorkerCapacity $WorkerCapacity `
    -WorkerRamCacheGiB $WorkerRamCacheGiB `
    -WorkerVramCacheGiB $WorkerVramCacheGiB `
    -MicrobatchWindowMs $MicrobatchWindowMs `
    -StartupTimeoutSeconds $StartupTimeoutSeconds `
    -BuildId $BuildId
