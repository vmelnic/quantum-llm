param(
    [ValidateSet("validate", "train", "probe", "evaluate", "run")][string]$Action = "run",
    [int]$Epochs = 3,
    [int]$BatchSize = 2,
    [int]$GradientAccumulation = 16,
    [double]$LearningRate = 0.0002,
    [int]$ValidationLimit = 256,
    [switch]$Resume,
    [int]$EvaluationLimit = 30,
    [int]$GateRank = 16,
    [double]$GateAlpha = 32.0,
    [double]$KnowledgeDropout = 0.2,
    [int]$MaximumMemoryTokens = 768,
    [int]$MaximumNewTokens = 128,
    [string]$OutputName = "memory-expert-capability-conflictqa-v4",
    [Parameter(Mandatory = $true)][string]$Corpus,
    [string]$ForbiddenFile = "",
    [int64]$MemoryCacheBytes = 4294967296,
    [int64]$MinimumFreeVramMiB = 18000
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ($OutputName -notmatch '^[A-Za-z0-9._-]+$' -or
    $Epochs -lt 1 -or $BatchSize -lt 1 -or $GradientAccumulation -lt 1 -or
    $LearningRate -le 0 -or $ValidationLimit -lt 0 -or
    $EvaluationLimit -lt 1 -or $GateRank -lt 1 -or
    $GateAlpha -le 0 -or $KnowledgeDropout -lt 0 -or $KnowledgeDropout -ge 1 -or
    $MaximumMemoryTokens -lt 32 -or $MaximumNewTokens -lt 8 -or
    $MemoryCacheBytes -lt 268435456) {
    throw "Invalid Memory Expert capability limits"
}

$environment = Join-Path $script:RepoRoot "work\memory-expert-venv"
$python = Join-Path $environment "Scripts\python.exe"
$script = Join-Path $script:RepoRoot "experiments\memory_expert\capability.py"
$selfTest = Join-Path $script:RepoRoot "experiments\memory_expert\capability_self_test.py"
$output = Join-Path (Join-Path $script:RepoRoot "work") $OutputName
$corpusPath = if ([System.IO.Path]::IsPathRooted($Corpus)) {
    $Corpus
} else { Join-Path $script:RepoRoot $Corpus }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Memory Expert environment is missing"
}
if (-not (Test-Path -LiteralPath $corpusPath -PathType Leaf)) {
    throw "Natural capability corpus is missing: $corpusPath"
}

$freeVram = & nvidia-smi.exe --query-gpu=memory.free --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or -not $freeVram) { throw "Cannot query free GPU memory" }
$freeVramMiB = [int64]($freeVram | Select-Object -First 1)
if ($freeVramMiB -lt $MinimumFreeVramMiB) {
    throw "Capability run requires $MinimumFreeVramMiB MiB free VRAM; available=$freeVramMiB"
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"

& $python $selfTest
if ($LASTEXITCODE -ne 0) { throw "Memory Expert capability self-test failed" }

$arguments = @(
    $script, $Action, "--output", $output,
    "--corpus", $corpusPath,
    "--epochs", [string]$Epochs,
    "--batch-size", [string]$BatchSize,
    "--gradient-accumulation", [string]$GradientAccumulation,
    "--learning-rate", [string]$LearningRate,
    "--validation-limit", [string]$ValidationLimit,
    "--evaluation-limit", [string]$EvaluationLimit,
    "--gate-rank", [string]$GateRank,
    "--gate-alpha", [string]$GateAlpha,
    "--knowledge-dropout", [string]$KnowledgeDropout,
    "--maximum-memory-tokens", [string]$MaximumMemoryTokens,
    "--maximum-new-tokens", [string]$MaximumNewTokens,
    "--memory-cache-bytes", [string]$MemoryCacheBytes
)
if ($Resume) { $arguments += "--resume" }
if ($ForbiddenFile) {
    $forbiddenPath = if ([System.IO.Path]::IsPathRooted($ForbiddenFile)) {
        $ForbiddenFile
    } else { Join-Path $script:RepoRoot $ForbiddenFile }
    if (-not (Test-Path -LiteralPath $forbiddenPath -PathType Leaf)) {
        throw "Forbidden benchmark file is missing: $forbiddenPath"
    }
    $arguments += @("--forbidden-file", $forbiddenPath)
}

$started = [DateTime]::UtcNow
& $python @arguments
$exitCode = $LASTEXITCODE
$status = [PSCustomObject]@{
    schema_version = 2
    action = $Action
    epochs = $Epochs
    batch_size = $BatchSize
    gradient_accumulation = $GradientAccumulation
    validation_limit = $ValidationLimit
    resume = [bool]$Resume
    output = [System.IO.Path]::GetFullPath($output)
    free_vram_mib_before = $freeVramMiB
    started_utc = $started.ToString("o")
    finished_utc = [DateTime]::UtcNow.ToString("o")
    exit_code = $exitCode
}
Write-JsonArtifact -Value $status -Name "memory-capability-status.json" | Out-Null
$status | ConvertTo-Json
if ($exitCode -ne 0) { exit $exitCode }
