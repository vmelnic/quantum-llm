param(
    [ValidateSet("index", "query", "status", "selftest")][string]$Action = "status",
    [Parameter(Mandatory = $true)][string]$DatasetName,
    [string]$EncoderModel = "BAAI/bge-m3",
    [string]$EncoderRevision = "5617a9f61b028005a4858fdac845db406aefb181",
    [int]$EncoderMaximumTokens = 1024,
    [int]$EncoderBatchSize = 16,
    [int]$TopK = 2,
    [int]$MaximumMemoryTokens = 768,
    [int]$MaximumNewTokens = 96,
    [string]$Language = "",
    [int64]$MinimumFreeVramMiB = 18000
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ($DatasetName -notmatch '^[A-Za-z0-9._-]+$' -or
    $EncoderRevision -notmatch '^[0-9a-f]{40}$' -or
    $EncoderMaximumTokens -lt 8 -or $EncoderBatchSize -lt 1 -or
    $TopK -lt 1 -or $MaximumMemoryTokens -lt 8 -or
    $MaximumNewTokens -lt 1 -or $MinimumFreeVramMiB -lt 1) {
    throw "Invalid Memory Data parameters"
}

$dataset = Join-Path (Join-Path $script:RepoRoot "work\memory-data") $DatasetName
$ingest = Join-Path $dataset "ingest"
$index = Join-Path $dataset "index"
$questions = Join-Path $dataset "questions.jsonl"
$report = Join-Path $dataset "query-report.json"
$checkpoint = Join-Path $script:RepoRoot "work\memory-expert-pow\memory-expert.pt"

if ($Action -eq "status") {
    $ingestManifest = Join-Path $ingest "manifest.json"
    $indexManifest = Join-Path $index "manifest.json"
    [PSCustomObject]@{
        schema_version = 1
        dataset = $DatasetName
        ingest_ready = Test-Path -LiteralPath $ingestManifest -PathType Leaf
        index_ready = Test-Path -LiteralPath $indexManifest -PathType Leaf
        questions_ready = Test-Path -LiteralPath $questions -PathType Leaf
        report_ready = Test-Path -LiteralPath $report -PathType Leaf
        ingest_manifest = if (Test-Path $ingestManifest) {
            Get-Content -LiteralPath $ingestManifest -Raw | ConvertFrom-Json
        } else { $null }
        index_manifest = if (Test-Path $indexManifest) {
            Get-Content -LiteralPath $indexManifest -Raw | ConvertFrom-Json
        } else { $null }
    } | ConvertTo-Json -Depth 8
    exit 0
}

$environment = Join-Path $script:RepoRoot "work\memory-expert-venv"
$python = Join-Path $environment "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Run Install-MemoryExpertEnvironment.ps1 first"
}
& $python -c "import numpy, torch, transformers; assert torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) { throw "Memory Expert environment is invalid" }
$selfTest = Join-Path $script:RepoRoot "experiments\memory_expert\data_self_test.py"
& $python $selfTest
if ($LASTEXITCODE -ne 0) { throw "Memory Data contract self-test failed" }
$runtimeSelfTest = Join-Path $script:RepoRoot `
    "experiments\memory_expert\data_runtime_self_test.py"
& $python $runtimeSelfTest
if ($LASTEXITCODE -ne 0) { throw "Memory Data runtime self-test failed" }
if ($Action -eq "selftest") {
    [PSCustomObject]@{
        schema_version = 1
        action = $Action
        contract = $true
        runtime = $true
    } | ConvertTo-Json
    exit 0
}

foreach ($required in @((Join-Path $ingest "manifest.json"), $checkpoint)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Memory Data artifact is missing: $required"
    }
}
if ($Action -eq "query" -and
    -not (Test-Path -LiteralPath $questions -PathType Leaf)) {
    throw "Question file is missing: $questions"
}

$freeVram = & nvidia-smi.exe --query-gpu=memory.free --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or -not $freeVram) { throw "Cannot query free GPU memory" }
$freeVramMiB = [int64]($freeVram | Select-Object -First 1)
if ($freeVramMiB -lt $MinimumFreeVramMiB) {
    throw "Memory Data requires $MinimumFreeVramMiB MiB free VRAM; available=$freeVramMiB"
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"

$started = [DateTime]::UtcNow
if ($Action -eq "index") {
    $script = Join-Path $script:RepoRoot "experiments\memory_expert\dense_index.py"
    & $python $script --ingest $ingest --output $index --model $EncoderModel `
        --revision $EncoderRevision --device cuda `
        --maximum-tokens $EncoderMaximumTokens --batch-size $EncoderBatchSize
} else {
    if (-not (Test-Path -LiteralPath (Join-Path $index "manifest.json") -PathType Leaf)) {
        throw "Build the dataset index before querying"
    }
    $script = Join-Path $script:RepoRoot "experiments\memory_expert\real_query.py"
    $arguments = @(
        $script, "--ingest", $ingest, "--index", $index,
        "--checkpoint", $checkpoint, "--questions", $questions,
        "--output", $report, "--encoder-model", $EncoderModel,
        "--encoder-revision", $EncoderRevision, "--device", "cuda",
        "--top-k", [string]$TopK,
        "--maximum-memory-tokens", [string]$MaximumMemoryTokens,
        "--maximum-new-tokens", [string]$MaximumNewTokens
    )
    if ($Language) { $arguments += @("--language", $Language) }
    & $python @arguments
}
$exitCode = $LASTEXITCODE
$status = [PSCustomObject]@{
    schema_version = 1
    action = $Action
    dataset = $DatasetName
    encoder_model = $EncoderModel
    encoder_revision = $EncoderRevision
    free_vram_mib_before = $freeVramMiB
    started_utc = $started.ToString("o")
    finished_utc = [DateTime]::UtcNow.ToString("o")
    exit_code = $exitCode
}
Write-JsonArtifact -Value $status -Name "memory-data-status.json" | Out-Null
$status | ConvertTo-Json
if ($exitCode -ne 0) { exit $exitCode }
