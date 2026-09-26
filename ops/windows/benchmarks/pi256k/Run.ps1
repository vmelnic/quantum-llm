param(
    [Parameter(Mandatory = $true)][string]$ModelRoot,
    [Parameter(Mandatory = $true)][ValidateSet('quantum', 'llama')][string]$Backend
)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
$root = [System.IO.Path]::GetFullPath($ModelRoot)
if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw 'MODEL_ROOT is missing' }
$output = Join-Path $repo "out/benchmarks/pi256k/$Backend"
$parent = Split-Path $output -Parent
[System.IO.Directory]::CreateDirectory($parent) | Out-Null
if (Test-Path -LiteralPath $output) { throw "Result directory already exists: $output" }
$python = Join-Path $repo '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = Join-Path $repo 'work/venv/server/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Server Python is missing' }
$arguments = @(
    (Join-Path $repo 'ops/python/pi256k_benchmark.py'),
    '--backend', $Backend, '--repo-root', $repo, '--model-root', $root,
    '--output', $output, '--port', $(if ($Backend -eq 'quantum') { '18080' } else { '18081' })
)
$runLog = Join-Path $parent "$Backend-run.log"
$errorLog = Join-Path $parent "$Backend-run.stderr.log"
try {
    $quotedArguments = @($arguments | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' })
    $process = Start-Process -FilePath $python -ArgumentList $quotedArguments -Wait -PassThru `
        -RedirectStandardOutput $runLog -RedirectStandardError $errorLog -WindowStyle Hidden
    if ($process.ExitCode -ne 0) { throw "Benchmark exited with code $($process.ExitCode)" }
} catch {
    $failurePath = Join-Path $parent "$Backend-FAILED.json"
    @{ schema = 'pi256k-launch-failure-v1'; backend = $Backend;
       error = $_.Exception.Message; log = $runLog; stderr_log = $errorLog;
       time_utc = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $failurePath -Encoding UTF8
    throw
}
