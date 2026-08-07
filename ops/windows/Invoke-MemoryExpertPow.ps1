param(
    [ValidateSet("train", "probe", "evaluate", "run")][string]$Action = "run",
    [string]$ModelId = "Qwen/Qwen3-4B",
    [string]$Revision = "1cfa9a7208912126459214e8b04321603b3df60c",
    [int]$Epochs = 3,
    [int]$BatchSize = 2,
    [int]$GradientAccumulation = 16,
    [double]$LearningRate = 0.0002,
    [ValidateSet("adamw", "lamb")][string]$Optimizer = "adamw",
    [int]$EvaluationLimit = 16,
    [int]$GateRank = 16,
    [double]$GateAlpha = 32.0,
    [int]$InjectionEvery = 1,
    [double]$KnowledgeDropout = 0.2,
    [int64]$MinimumFreeVramMiB = 18000,
    [string]$OutputDirectory = ""
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ($Epochs -lt 1 -or $BatchSize -lt 1 -or $GradientAccumulation -lt 1 -or
    $LearningRate -le 0 -or $EvaluationLimit -lt 1 -or
    $GateRank -lt 1 -or $GateAlpha -le 0 -or $InjectionEvery -lt 1 -or
    $KnowledgeDropout -lt 0 -or $KnowledgeDropout -ge 1 -or
    $MinimumFreeVramMiB -lt 1) {
    throw "Invalid Memory Expert PoW limits"
}

$snapshot = Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
$indexPath = Join-Path $snapshot "model.safetensors.index.json"
if (-not (Test-Path -LiteralPath $indexPath -PathType Leaf)) {
    throw "The pinned model snapshot is incomplete: missing $indexPath"
}
$modelIndex = Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json
$missingShards = @($modelIndex.weight_map.PSObject.Properties.Value |
    Sort-Object -Unique | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $snapshot ([string]$_)) -PathType Leaf)
    })
if ($missingShards.Count -gt 0) {
    throw "The pinned model snapshot is incomplete: $($missingShards.Count) model shards are missing"
}
$environment = Join-Path $script:RepoRoot "work\memory-expert-venv"
$pythonPath = Join-Path $environment "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Run Install-MemoryExpertEnvironment.ps1 first"
}
$python = Get-Item -LiteralPath $pythonPath
$script = Join-Path $script:RepoRoot "experiments\memory_expert\pow.py"
if (-not (Test-Path -LiteralPath $script -PathType Leaf)) {
    throw "Memory Expert PoW script is missing: $script"
}
$selfTest = Join-Path $script:RepoRoot "experiments\memory_expert\self_test.py"
if (-not (Test-Path -LiteralPath $selfTest -PathType Leaf)) {
    throw "Memory Expert self-test is missing: $selfTest"
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $script:RepoRoot "work\memory-expert-pow"
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$freeVram = & nvidia-smi.exe --query-gpu=memory.free --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or -not $freeVram) {
    throw "Cannot query free GPU memory"
}
$freeVramMiB = [int64]($freeVram | Select-Object -First 1)
if ($freeVramMiB -lt $MinimumFreeVramMiB) {
    throw "Memory Expert PoW requires $MinimumFreeVramMiB MiB free VRAM; available=$freeVramMiB"
}

& $python.FullName -c "import torch, transformers, numpy"
if ($LASTEXITCODE -ne 0) {
    throw "Python environment is missing torch, transformers, or numpy"
}
& $python.FullName $selfTest
if ($LASTEXITCODE -ne 0) {
    throw "Memory Expert structural self-test failed"
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"

$arguments = @(
    $script,
    $Action,
    "--model", $ModelId,
    "--revision", $Revision,
    "--output", $OutputDirectory,
    "--epochs", [string]$Epochs,
    "--batch-size", [string]$BatchSize,
    "--gradient-accumulation", [string]$GradientAccumulation,
    "--learning-rate", [string]$LearningRate,
    "--optimizer", $Optimizer,
    "--evaluation-limit", [string]$EvaluationLimit,
    "--gate-rank", [string]$GateRank,
    "--gate-alpha", [string]$GateAlpha,
    "--injection-every", [string]$InjectionEvery,
    "--knowledge-dropout", [string]$KnowledgeDropout
)

$started = [DateTime]::UtcNow
& $python.FullName @arguments
$exitCode = $LASTEXITCODE
$status = [PSCustomObject]@{
    schema_version = 2
    action = $Action
    model_id = $ModelId
    revision = $Revision
    snapshot = $snapshot
    output_directory = [System.IO.Path]::GetFullPath($OutputDirectory)
    epochs = $Epochs
    batch_size = $BatchSize
    gradient_accumulation = $GradientAccumulation
    learning_rate = $LearningRate
    optimizer = $Optimizer
    evaluation_limit = $EvaluationLimit
    gate_rank = $GateRank
    gate_alpha = $GateAlpha
    injection_every = $InjectionEvery
    knowledge_dropout = $KnowledgeDropout
    free_vram_mib_before = $freeVramMiB
    started_utc = $started.ToString("o")
    finished_utc = [DateTime]::UtcNow.ToString("o")
    exit_code = $exitCode
}
Write-JsonArtifact -Value $status -Name "memory-expert-pow-status.json" | Out-Null
$status | ConvertTo-Json -Depth 8
if ($exitCode -ne 0) { exit $exitCode }
