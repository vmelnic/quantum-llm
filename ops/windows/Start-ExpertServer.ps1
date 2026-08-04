param(
    [string]$Container = "C:\Users\vladi\quantum-llm\work\models\olmoe-expert-pack-int8",
    [string]$Tokenizer = "",
    [string]$Runner = "",
    [string]$ModelId = "olmoe-expert-pack-int8",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [int]$MaximumQueue = 8,
    [int]$MaximumContext = 4096,
    [int]$WorkerCapacity = 1,
    [int]$WorkerRamCacheGiB = 48,
    [int]$WorkerVramCacheGiB = 14,
    [double]$MicrobatchWindowMs = 2.0,
    [int]$LatencyWindow = 4096,
    [int]$StartupTimeoutSeconds = 120,
    [string]$BuildId = "development"
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$python = Join-Path $script:RepoRoot "work\venv\colibri\Scripts\python.exe"
$worker = if ($Runner) { [System.IO.Path]::GetFullPath($Runner) } else {
    Join-Path $script:RepoRoot "out\build\windows-msvc-release\runtime\Release\expert-olmoe-runner.exe"
}
$server = Join-Path $script:RepoRoot "ops\python\expert_server.py"
$logFile = Join-Path (Join-Path $script:RepoRoot "logs") "expert-server.jsonl"
if (-not (Test-Path $python -PathType Leaf)) { throw "Python environment missing: $python" }
if (-not (Test-Path $worker -PathType Leaf)) { throw "CUDA worker missing: $worker" }
if (-not (Test-Path $Container -PathType Container)) { throw "Container missing: $Container" }
if (-not $Tokenizer) {
    $containerTokenizer = Join-Path $Container "tokenizer"
    if (Test-Path $containerTokenizer -PathType Container) {
        $Tokenizer = $containerTokenizer
    } else {
        $snapshotRoot = "C:\Users\vladi\.cache\huggingface\hub\models--allenai--OLMoE-1B-7B-0125-Instruct\snapshots"
        $Tokenizer = (Get-ChildItem $snapshotRoot -Directory | Select-Object -First 1).FullName
    }
}

& $python $server `
    --worker $worker `
    --container $Container `
    --tokenizer $Tokenizer `
    --host $HostAddress `
    --port $Port `
    --maximum-queue $MaximumQueue `
    --max-context $MaximumContext `
    --worker-capacity $WorkerCapacity `
    --worker-ram-cache-gib $WorkerRamCacheGiB `
    --worker-vram-cache-gib $WorkerVramCacheGiB `
    --microbatch-window-ms $MicrobatchWindowMs `
    --latency-window $LatencyWindow `
    --startup-timeout $StartupTimeoutSeconds `
    --model $ModelId `
    --build-id $BuildId `
    --log-file $logFile
if ($LASTEXITCODE -ne 0) { throw "Expert server exited with code $LASTEXITCODE" }
