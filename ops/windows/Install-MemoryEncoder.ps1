param(
    [string]$ModelId = "BAAI/bge-m3",
    [string]$Revision = "5617a9f61b028005a4858fdac845db406aefb181",
    [int64]$ExpectedWeightBytes = 2271145830,
    [int]$MaxWorkers = 4
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ($ModelId -notmatch '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$' -or
    $Revision -notmatch '^[0-9a-f]{40}$' -or $ExpectedWeightBytes -lt 1 -or
    $MaxWorkers -lt 1) {
    throw "Invalid pinned encoder download parameters"
}

$hf = Join-Path $env:USERPROFILE ".hf-cli\venv\Scripts\hf.exe"
if (-not (Test-Path -LiteralPath $hf -PathType Leaf)) {
    $hf = (Get-Command hf.exe -ErrorAction Stop).Source
}
$snapshot = Join-Path (Join-Path (Join-Path (Join-Path $env:USERPROFILE `
    ".cache\huggingface\hub") ("models--" + ($ModelId -replace "/", "--"))) `
    "snapshots") $Revision
$weights = Join-Path $snapshot "pytorch_model.bin"
$weightLength = [int64]0
if (Test-Path -LiteralPath $weights -PathType Leaf) {
    $stream = [System.IO.File]::OpenRead($weights)
    try { $weightLength = [int64]$stream.Length } finally { $stream.Dispose() }
}
$requiredFiles = @(
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "pytorch_model.bin"
)
$missingFiles = @($requiredFiles | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $snapshot $_) -PathType Leaf)
})
$alreadyComplete = $weightLength -eq $ExpectedWeightBytes -and $missingFiles.Count -eq 0

if (-not $alreadyComplete) {
    & $hf download $ModelId --revision $Revision --max-workers $MaxWorkers `
        --include "*.json" --include "*.model" --include "pytorch_model.bin"
    if ($LASTEXITCODE -ne 0) { throw "Encoder download failed" }
}
if (-not (Test-Path -LiteralPath $weights -PathType Leaf)) {
    throw "Pinned encoder snapshot has no pytorch_model.bin"
}
$missingFiles = @($requiredFiles | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $snapshot $_) -PathType Leaf)
})
if ($missingFiles.Count -gt 0) {
    throw "Pinned encoder snapshot is incomplete: $($missingFiles -join ', ')"
}
$stream = [System.IO.File]::OpenRead($weights)
try { $actualBytes = [int64]$stream.Length } finally { $stream.Dispose() }
if ($actualBytes -ne $ExpectedWeightBytes) {
    throw "Encoder weight size mismatch: expected=$ExpectedWeightBytes actual=$actualBytes"
}

[PSCustomObject]@{
    schema_version = 1
    model_id = $ModelId
    revision = $Revision
    snapshot = [System.IO.Path]::GetFullPath($snapshot)
    weight_bytes = $actualBytes
    downloaded = -not $alreadyComplete
    validated = $true
} | ConvertTo-Json
