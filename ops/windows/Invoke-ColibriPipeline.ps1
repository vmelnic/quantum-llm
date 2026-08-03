param(
    [switch]$InstallToolchain,
    [switch]$InstallPythonDependencies,
    [switch]$GenerateIndependentOracle,
    [switch]$RunSweep
)

$ErrorActionPreference = "Stop"

$prepare = Join-Path $PSScriptRoot "Invoke-PrepareColibri.ps1"
$convert = Join-Path $PSScriptRoot "Invoke-ConvertOlmoeForColibri.ps1"
$oracle = Join-Path $PSScriptRoot "Invoke-GenerateOlmoeOracle.ps1"
$smoke = Join-Path $PSScriptRoot "Invoke-ColibriSmoke.ps1"
$benchmark = Join-Path $PSScriptRoot "Invoke-ColibriBenchmark.ps1"

& $prepare `
    -InstallToolchain:$InstallToolchain `
    -InstallPythonDependencies:$InstallPythonDependencies
& $convert
if ($GenerateIndependentOracle) {
    & $oracle
}
& $smoke -UseIndependentOracle:$GenerateIndependentOracle
if ($RunSweep) {
    & $benchmark -UseIndependentOracle:$GenerateIndependentOracle
}
