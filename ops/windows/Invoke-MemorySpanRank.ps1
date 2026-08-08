param(
    [Parameter(Mandatory = $true)][string]$DatasetName,
    [Parameter(Mandatory = $true)][string]$Corpus,
    [string]$Checkpoint = "work/memory-expert-capability-conflictqa-v4/memory-expert.pt",
    [string]$OutputName = "memory-mechanism-span-rank-v1",
    [int]$OutOfDistributionCases = 8,
    [int64]$MinimumFreeVramMiB = 18000
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ($DatasetName -notmatch '^[A-Za-z0-9._-]+$' -or
    $OutputName -notmatch '^[A-Za-z0-9._-]+$' -or
    $OutOfDistributionCases -lt 1 -or $MinimumFreeVramMiB -lt 1) {
    throw "Invalid Memory Span Rank parameters"
}

$environment = Join-Path $script:RepoRoot "work\memory-expert-venv"
$python = Join-Path $environment "Scripts\python.exe"
$dataset = Join-Path (Join-Path $script:RepoRoot "work\memory-data") $DatasetName
$ingest = Join-Path $dataset "ingest"
$questions = Join-Path $dataset "questions.jsonl"
$corpusPath = if ([System.IO.Path]::IsPathRooted($Corpus)) {
    $Corpus
} else { Join-Path $script:RepoRoot $Corpus }
$checkpointPath = if ([System.IO.Path]::IsPathRooted($Checkpoint)) {
    $Checkpoint
} else { Join-Path $script:RepoRoot $Checkpoint }
$outputRoot = Join-Path (Join-Path $script:RepoRoot "work") $OutputName
$output = Join-Path $outputRoot "span-rank.json"

foreach ($required in @(
    $python, (Join-Path $ingest "manifest.json"), $questions,
    $corpusPath, "${corpusPath}.manifest.json", $checkpointPath
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Memory Span Rank artifact is missing: $required"
    }
}

$freeVram = & nvidia-smi.exe --query-gpu=memory.free --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or -not $freeVram) { throw "Cannot query free GPU memory" }
$freeVramMiB = [int64]($freeVram | Select-Object -First 1)
if ($freeVramMiB -lt $MinimumFreeVramMiB) {
    throw "Memory Span Rank requires $MinimumFreeVramMiB MiB free VRAM; available=$freeVramMiB"
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"

$script = Join-Path $script:RepoRoot "experiments\memory_expert\mechanism_span_rank.py"
& $python $script --checkpoint $checkpointPath --corpus $corpusPath `
    --ingest $ingest --questions $questions --output $output --device cuda `
    --ood-cases $OutOfDistributionCases
$exitCode = $LASTEXITCODE

[PSCustomObject]@{
    schema_version = 1
    action = "memory-span-rank"
    dataset = $DatasetName
    corpus = $Corpus
    checkpoint = $Checkpoint
    output = $output
    free_vram_mib_before = $freeVramMiB
    exit_code = $exitCode
} | ConvertTo-Json
if ($exitCode -ne 0) { exit $exitCode }
