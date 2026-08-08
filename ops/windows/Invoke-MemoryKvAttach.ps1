param(
    [Parameter(Mandatory = $true)][string]$DatasetName,
    [string]$OutputName = "memory-kv-attach-v1",
    [string]$Selection = "",
    [string]$Lora = "",
    [string]$ModelId = "Qwen/Qwen3-4B",
    [string]$Revision = "1cfa9a7208912126459214e8b04321603b3df60c",
    [int]$MaximumMemoryTokens = 768,
    [int]$MaximumNewTokens = 128,
    [int64]$MinimumFreeVramMiB = 18000
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ($DatasetName -notmatch '^[A-Za-z0-9._-]+$' -or
    $OutputName -notmatch '^[A-Za-z0-9._-]+$' -or
    $MaximumMemoryTokens -lt 1 -or $MaximumNewTokens -lt 1 -or
    $MinimumFreeVramMiB -lt 1) {
    throw "Invalid Memory KV-Attach parameters"
}

$environment = Join-Path $script:RepoRoot "work\memory-expert-venv"
$python = Join-Path $environment "Scripts\python.exe"
$dataset = Join-Path (Join-Path $script:RepoRoot "work\memory-data") $DatasetName
$ingest = Join-Path $dataset "ingest"
$questions = Join-Path $dataset "questions.jsonl"
$outputRoot = Join-Path (Join-Path $script:RepoRoot "work") $OutputName
$output = Join-Path $outputRoot "kv-attach.json"

foreach ($required in @(
    $python, (Join-Path $ingest "manifest.json"), $questions
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Memory KV-Attach artifact is missing: $required"
    }
}

$freeVram = & nvidia-smi.exe --query-gpu=memory.free --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or -not $freeVram) { throw "Cannot query free GPU memory" }
$freeVramMiB = [int64]($freeVram | Select-Object -First 1)
if ($freeVramMiB -lt $MinimumFreeVramMiB) {
    throw "Memory KV-Attach requires $MinimumFreeVramMiB MiB free VRAM; available=$freeVramMiB"
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"

$script = Join-Path $script:RepoRoot "experiments\memory_expert\kv_attach.py"
$selectionArguments = @()
if ($Selection -ne "") {
    $selectionPath = Join-Path $dataset $Selection
    if (-not (Test-Path -LiteralPath $selectionPath -PathType Leaf)) {
        throw "Required Memory KV-Attach selection artifact is missing: $selectionPath"
    }
    $selectionArguments = @("--selection", $selectionPath)
}
if ($Lora -ne "") {
    $loraPath = if ([System.IO.Path]::IsPathRooted($Lora)) {
        $Lora
    } else { Join-Path $script:RepoRoot $Lora }
    if (-not (Test-Path -LiteralPath $loraPath -PathType Leaf)) {
        throw "Required Memory KV-Attach LoRA checkpoint is missing: $loraPath"
    }
    $selectionArguments += @("--lora", $loraPath)
}
& $python $script --model $ModelId --revision $Revision `
    --ingest $ingest --questions $questions --output $output --device cuda `
    --maximum-memory-tokens $MaximumMemoryTokens `
    --maximum-new-tokens $MaximumNewTokens @selectionArguments
$exitCode = $LASTEXITCODE

[PSCustomObject]@{
    schema_version = 1
    action = "memory-kv-attach-validate"
    dataset = $DatasetName
    output = $output
    exit_code = $exitCode
} | ConvertTo-Json -Compress

if ($exitCode -ne 0) { exit $exitCode }
