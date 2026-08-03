param(
    [int]$Layer = 0,
    [int]$MicroExperts = 16,
    [int]$MicroWidth = 256,
    [int]$MaxActiveMicro = 8,
    [double]$CoverageThreshold = 0.75,
    [double]$RiskThreshold = 0.60,
    [int]$Epochs = 12,
    [int]$Threads = 6
)

. (Join-Path $PSScriptRoot "Common.ps1")

Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig
$modelId = [string]$config.models.recommended_lab_moe
$revision = [string]$config.models.recommended_lab_revision
$snapshot = Get-ModelSnapshot -ModelId $modelId -Revision $revision `
    -HubRoot $config.huggingface_hub -RequireComplete
$python = Get-ColibriPythonCommand
$compiler = Join-Path $script:OpsRoot "python\compile_hesr_amc_layer.py"
$prompts = Join-Path $script:OpsRoot "config\amc_calibration_prompts.json"
$outputDir = Join-Path $script:RepoRoot "work\models\hesr-amc\olmoe-layer-$Layer"
$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$report = Join-Path (Join-Path $script:RepoRoot "artifacts") "hesr-amc-compile-layer-$Layer-$stamp.json"
$latest = Join-Path (Join-Path $script:RepoRoot "artifacts") "hesr-amc-compile-layer-$Layer-latest.json"
$log = Join-Path (Join-Path $script:RepoRoot "logs") "hesr-amc-compile-layer-$Layer-$stamp.log"

$previousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $python.Source $compiler `
        --snapshot $snapshot `
        --prompts $prompts `
        --output-dir $outputDir `
        --report $report `
        --layer $Layer `
        --micro-experts $MicroExperts `
        --micro-width $MicroWidth `
        --max-active-micro $MaxActiveMicro `
        --coverage-threshold $CoverageThreshold `
        --risk-threshold $RiskThreshold `
        --epochs $Epochs `
        --threads $Threads 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
if ($exitCode -ne 0) {
    throw "HESR-AMC compilation failed with code $exitCode. Log: $log"
}
Copy-Item -LiteralPath $report -Destination $latest -Force
Write-Output "Compiled HESR-AMC layer: $outputDir"
Write-Output "Compilation report: $report"
