param(
    [Parameter(Mandatory = $true)][string]$Corpus,
    [string]$OutputName = "memory-kv-attach-lora-v1",
    [string]$ModelId = "Qwen/Qwen3-4B",
    [string]$Revision = "1cfa9a7208912126459214e8b04321603b3df60c",
    [int]$Rank = 16,
    [double]$Alpha = 32.0,
    [int]$Epochs = 3,
    [int]$Accumulation = 16,
    [double]$LearningRate = 0.0002,
    [int]$EvalLimit = 128,
    [int]$MaximumMemoryTokens = 768,
    [int64]$MinimumFreeVramMiB = 14000
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ($OutputName -notmatch '^[A-Za-z0-9._-]+$' -or
    $Rank -lt 1 -or $Epochs -lt 1 -or $Accumulation -lt 1 -or
    $EvalLimit -lt 1 -or $MaximumMemoryTokens -lt 1 -or
    $MinimumFreeVramMiB -lt 1) {
    throw "Invalid Memory KV-Attach Train parameters"
}

$environment = Join-Path $script:RepoRoot "work\memory-expert-venv"
$python = Join-Path $environment "Scripts\python.exe"
$corpusPath = if ([System.IO.Path]::IsPathRooted($Corpus)) {
    $Corpus
} else { Join-Path $script:RepoRoot $Corpus }
$outputRoot = Join-Path (Join-Path $script:RepoRoot "work") $OutputName

foreach ($required in @($python, $corpusPath, "${corpusPath}.manifest.json")) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Memory KV-Attach Train artifact is missing: $required"
    }
}

$freeVram = & nvidia-smi.exe --query-gpu=memory.free --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or -not $freeVram) { throw "Cannot query free GPU memory" }
$freeVramMiB = [int64]($freeVram | Select-Object -First 1)
if ($freeVramMiB -lt $MinimumFreeVramMiB) {
    throw "Memory KV-Attach Train requires $MinimumFreeVramMiB MiB free VRAM; available=$freeVramMiB"
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"

$script = Join-Path $script:RepoRoot "experiments\memory_expert\kv_attach_train.py"
& $python $script --model $ModelId --revision $Revision `
    --corpus $corpusPath --output $outputRoot --device cuda `
    --rank $Rank --alpha $Alpha --epochs $Epochs `
    --accumulation $Accumulation --learning-rate $LearningRate `
    --eval-limit $EvalLimit --maximum-memory-tokens $MaximumMemoryTokens
$exitCode = $LASTEXITCODE

[PSCustomObject]@{
    schema_version = 1
    action = "memory-kv-attach-train"
    corpus = $Corpus
    output = $outputRoot
    exit_code = $exitCode
} | ConvertTo-Json -Compress

if ($exitCode -ne 0) { exit $exitCode }
