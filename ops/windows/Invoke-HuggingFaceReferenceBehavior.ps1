param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [string]$Corpus = "",
    [string]$Output = "",
    [string]$MaximumGpuMemory = "20GiB",
    [string]$MaximumCpuMemory = "42GiB"
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ([string]::IsNullOrWhiteSpace($Corpus)) {
    $Corpus = Join-Path $script:RepoRoot "ops\quality\behavior-corpus-v1.json"
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $script:RepoRoot "artifacts\official-reference-behavior-latest.json"
}
$snapshot = Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
$python = Join-Path $script:RepoRoot "work\venv\hf-reference\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Install the isolated reference environment first"
}
$runner = Join-Path $script:RepoRoot "ops\python\hf_reference_smoke.py"
$offload = Join-Path $script:RepoRoot "work\hf-reference-offload"

& $python $runner `
    --snapshot $snapshot `
    --corpus ([System.IO.Path]::GetFullPath($Corpus)) `
    --output ([System.IO.Path]::GetFullPath($Output)) `
    --max-gpu-memory $MaximumGpuMemory `
    --max-cpu-memory $MaximumCpuMemory `
    --offload-directory $offload
if ($LASTEXITCODE -ne 0) { throw "Official reference behavior gate failed" }
Get-Content -LiteralPath ([System.IO.Path]::GetFullPath($Output)) -Raw
