param(
    [Parameter(Mandatory = $true)][string]$ArtifactName,
    [Parameter(Mandatory = $true)]
    [ValidateSet('q4-f16-per-head', 'fp4-e2m1-ue8m0-block32-key-outlier1', 'fp16')]
    [string]$KvDtype,
    [Parameter(Mandatory = $true)][string]$RoundId
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($env:MODEL_ROOT)) { throw 'MODEL_ROOT is required' }
if ($RoundId -notmatch '^[a-z0-9][a-z0-9_-]*$') { throw 'RoundId is invalid' }
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = Join-Path $repoRoot 'work/venv/server/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Server Python is missing' }
$runner = Join-Path $repoRoot 'out/build/windows-msvc-release/runtime/Release/expert-moe-vm-runner.exe'
$prompt = Join-Path $repoRoot 'ops/benchmarks/qwen27b_english_startups_prompt.json'
$output = Join-Path $repoRoot ("out/benchmarks/qwen27b-english-speed/$RoundId")
& $python (Join-Path $repoRoot 'ops/python/qwen_speed_benchmark.py') `
    --model-root $env:MODEL_ROOT --repo-root $repoRoot --runner $runner `
    --prompt $prompt --artifact $ArtifactName --kv $KvDtype `
    --run-id $RoundId --output $output
if ($LASTEXITCODE -ne 0) { throw "Benchmark round failed with exit code $LASTEXITCODE" }
