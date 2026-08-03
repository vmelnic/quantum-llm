param(
    [ValidateSet(0, 1)][int]$RamCache = 1,
    [int]$Cache = 8,
    [int]$Context = 4096,
    [int]$MaxNew = 512,
    [int]$Threads = 6,
    [ValidateSet(0, 1)][int]$Direct = 1
)

. (Join-Path $PSScriptRoot "Common.ps1")

Set-CpuOnlyEnvironment
$config = Get-ExperimentConfig
$checkout = Resolve-RepositoryPath -Path ([string]$config.colibri.checkout)
$model = Resolve-RepositoryPath -Path ([string]$config.colibri.converted_model)
$engine = Join-Path $checkout "c\olmoe.exe"
if (-not (Test-Path -LiteralPath $engine -PathType Leaf)) {
    throw "Colibri OLMoE engine is missing: $engine"
}
if (-not (Test-Path -LiteralPath $model -PathType Container)) {
    throw "Converted Colibri model is missing: $model"
}
if ($Context -lt 1 -or $Context -gt 4096) {
    throw "Context must be between 1 and 4096."
}
if ($MaxNew -lt 1) {
    throw "MaxNew must be positive."
}

$saved = @{}
foreach ($name in @(
    "SNAP", "CHAT", "RAMCACHE", "DIRECT", "PILOT", "HOT", "EXPERT_DROP",
    "CTX", "MAX_NEW", "OMP_NUM_THREADS"
)) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
try {
    $env:SNAP = $model
    $env:CHAT = "1"
    $env:RAMCACHE = [string]$RamCache
    $env:DIRECT = [string]$Direct
    $env:PILOT = "0"
    $env:HOT = "0"
    $env:EXPERT_DROP = "0"
    $env:CTX = [string]$Context
    $env:MAX_NEW = [string]$MaxNew
    $env:OMP_NUM_THREADS = [string]$Threads
    Write-Output "Starting OLMoE chat: RAMCACHE=$RamCache threads=$Threads context=$Context"
    & $engine ([string]$Cache) ([string]$config.colibri.quant_bits)
    if ($LASTEXITCODE -ne 0) {
        throw "Colibri chat exited with code $LASTEXITCODE"
    }
}
finally {
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], "Process")
    }
}
