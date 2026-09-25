$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($env:MODEL_ROOT)) { throw 'MODEL_ROOT is required' }
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
$modelRoot = [System.IO.Path]::GetFullPath($env:MODEL_ROOT)
$runner = Join-Path $repoRoot 'out/build/windows-msvc-release/runtime/Release/expert-moe-vm-runner.exe'
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = Join-Path $repoRoot 'work/venv/server/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) { throw 'CUDA runner missing' }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Server Python missing' }

$scripts = @(
    (Join-Path $repoRoot 'ops/windows/Run-Qwen27BSpeedRound.ps1'),
    (Join-Path $PSScriptRoot 'Launch.ps1'),
    (Join-Path $PSScriptRoot 'Run-All.ps1')
) + @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter 'Run.ps1' -Recurse |
    Select-Object -ExpandProperty FullName)
if ($scripts.Count -ne 15) { throw "Expected 15 PowerShell files, got $($scripts.Count)" }
foreach ($scriptPath in $scripts) {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $scriptPath, [ref]$tokens, [ref]$errors) | Out-Null
    if ($errors.Count -ne 0) {
        throw "PowerShell parse error in $scriptPath : $($errors[0].Message)"
    }
}

$names = @(
    'qwen3.8-27b-abliterated-fp4',
    'qwen3.8-27b-abliterated-fp4-mse-v2',
    'qwen3.8-27b-abliterated-fp4-activation-v3',
    'qwen3.8-27b-abliterated-fp4-activation-v4'
)
$artifacts = @()
foreach ($name in $names) {
    $container = Join-Path $modelRoot $name
    $markerPath = Join-Path $container 'COMPLETED'
    $manifestPath = Join-Path $container 'manifest.json'
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Incomplete artifact: $name"
    }
    $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
    $actualHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($marker.manifest_file_sha256.ToLowerInvariant() -ne $actualHash) {
        throw "Manifest hash mismatch: $name"
    }
    $artifacts += $name
}

$gpu = @(nvidia-smi --query-gpu=index,name --format=csv,noheader |
    Where-Object { $_ -match 'RTX 3090' })
if ($gpu.Count -ne 1) { throw "Expected one RTX 3090, got $($gpu.Count)" }
if (Get-ScheduledTask -TaskName 'QuantumLLM-Qwen27B-EnglishSpeed-Single-20260924' -ErrorAction SilentlyContinue) {
    throw 'Benchmark task already exists'
}
if (Test-Path -LiteralPath (Join-Path $repoRoot 'out/benchmarks/qwen27b-english-speed')) {
    throw 'Benchmark result directory already exists'
}
@{
    ready = $true
    scripts = $scripts.Count
    artifacts = $artifacts
    gpu = $gpu[0]
    runner = $runner
    python = $python
} | ConvertTo-Json -Compress
