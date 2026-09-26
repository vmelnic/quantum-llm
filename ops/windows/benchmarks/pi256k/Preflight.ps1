param(
    [Parameter(Mandatory = $true)][string]$ModelRoot,
    [Parameter(Mandatory = $true)][ValidateSet('quantum', 'llama')][string]$Backend
)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
$root = [System.IO.Path]::GetFullPath($ModelRoot)
if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw 'MODEL_ROOT missing' }
$pi = Join-Path $repo 'out/benchmarks/pi256k-tools/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js'
if (-not (Test-Path -LiteralPath $pi -PathType Leaf)) { throw 'Pinned Pi is not installed' }
if ((& node $pi --version).Trim() -ne '0.87.1') { throw 'Pinned Pi version mismatch' }
$gpus = @(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits |
    Where-Object { $_ -match 'RTX 3090' })
if ($gpus.Count -ne 1) { throw 'Expected one RTX 3090' }
$free = [int](($gpus[0] -split ',')[-1].Trim())
if ($free -lt 23000) { throw "RTX 3090 is not idle: $free MiB free" }
$port = if ($Backend -eq 'quantum') { 18080 } else { 18081 }
if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
    throw "Benchmark port $port is occupied"
}
if (Test-Path -LiteralPath (Join-Path $repo "out/benchmarks/pi256k/$Backend")) {
    throw 'Result directory already exists'
}
if (Get-ScheduledTask -TaskName "QuantumLLM-Pi256k-$Backend" -ErrorAction SilentlyContinue) {
    throw 'Benchmark task already exists'
}
$model = if ($Backend -eq 'quantum') {
    $artifact = Join-Path $root 'qwen3.8-27b-fp4'
    $marker = Join-Path $artifact 'COMPLETED'
    $manifest = Join-Path $artifact 'manifest.json'
    $runner = Join-Path $repo 'out/build/windows-msvc-release/runtime/Release/expert-moe-vm-runner.exe'
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf) -or
        -not (Test-Path -LiteralPath $manifest -PathType Leaf) -or
        -not (Test-Path -LiteralPath $runner -PathType Leaf)) {
        throw 'Qwen artifact or CUDA runner is incomplete'
    }
    $expected = (Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json).manifest_file_sha256
    $actual = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expected.ToLowerInvariant() -ne $actual) { throw 'Qwen manifest hash mismatch' }
    $artifact
} else {
    $artifact = Join-Path $root 'qwen3.8-27b-gguf-q4-k-m'
    $manifest = Join-Path $artifact 'manifest.json'
    $marker = Join-Path $artifact 'COMPLETED'
    $binary = Join-Path $repo 'out/llama-v0.5.0/build/bin/Release/llama-server.exe'
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf) -or
        -not (Test-Path -LiteralPath $marker -PathType Leaf) -or
        -not (Test-Path -LiteralPath $binary -PathType Leaf)) {
        throw 'Pinned GGUF artifact or llama.cpp build is missing'
    }
    $expected = (Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json).manifest_file_sha256
    $actual = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expected.ToLowerInvariant() -ne $actual) { throw 'GGUF manifest hash mismatch' }
    $memory = Get-CimInstance Win32_OperatingSystem
    $freeRamBytes = [int64]$memory.FreePhysicalMemory * 1024
    if ($freeRamBytes -lt (20654142304 + 2GB)) {
        throw "Insufficient free RAM for GGUF CPU placement: $freeRamBytes bytes"
    }
    $artifact
}
try {
    $search = Invoke-RestMethod -Uri 'https://api.github.com/search/repositories?per_page=1&q=icalendar' `
        -Headers @{ 'User-Agent' = 'quantum-llm-pi256k-benchmark/1' } -TimeoutSec 15
    if (@($search.items).Count -lt 1) { throw 'No live search results' }
} catch { throw "Live search preflight failed: $($_.Exception.Message)" }
@{
    ready = $true
    backend = $Backend
    model_artifact = $model
    model_root = $root
    gpu = $gpus[0]
    port = $port
    context = 256000
    thinking = 'xhigh'
    compaction = 'disabled in isolated Pi session'
    mtp = 'MTP-4'
    ram_offload = if ($Backend -eq 'llama') { 'llama.cpp auto-fit; exact placement captured at startup' } else { 'not requested' }
    live_search = 'GitHub API returned results'
} | ConvertTo-Json -Compress
