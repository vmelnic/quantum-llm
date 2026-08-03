param(
    [int]$Cache = 4,
    [int]$Pilot = 0,
    [int]$Direct = 1,
    [ValidateSet(0, 1)][int]$RamCache = 0,
    [switch]$UseIndependentOracle
)

$benchmark = Join-Path $PSScriptRoot "Invoke-ColibriBenchmark.ps1"
& $benchmark `
    -CacheSizes ([string]$Cache) `
    -PilotModes ([string]$Pilot) `
    -DirectModes ([string]$Direct) `
    -RamCache $RamCache `
    -Repeats 1 `
    -UseIndependentOracle:$UseIndependentOracle
