param(
    [int]$Layer = 0,
    [int]$RetainedWidth = 252,
    [int]$CorrectionRank = 32,
    [double]$RiskThreshold = 0.35,
    [int]$Threads = 6
)

. (Join-Path $PSScriptRoot "Common.ps1")
Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig
$snapshot = Get-ModelSnapshot -ModelId ([string]$config.models.recommended_lab_moe) `
    -Revision ([string]$config.models.recommended_lab_revision) `
    -HubRoot $config.huggingface_hub -RequireComplete
$python = Get-ColibriPythonCommand
$compiler = Join-Path $script:OpsRoot "python\compile_amc_neuron_bank.py"
$prompts = Join-Path $script:OpsRoot "config\amc_calibration_prompts.json"
$outputDir = Join-Path $script:RepoRoot "work\models\hesr-amc\neuron-bank-layer-$Layer"
$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$report = Join-Path (Join-Path $script:RepoRoot "artifacts") "amc-neuron-bank-compile-layer-$Layer-$stamp.json"
$latest = Join-Path (Join-Path $script:RepoRoot "artifacts") "amc-neuron-bank-compile-layer-$Layer-latest.json"
$log = Join-Path (Join-Path $script:RepoRoot "logs") "amc-neuron-bank-compile-layer-$Layer-$stamp.log"

$oldPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $python.Source $compiler --snapshot $snapshot --prompts $prompts `
        --output-dir $outputDir --report $report --layer $Layer `
        --retained-width $RetainedWidth --correction-rank $CorrectionRank `
        --risk-threshold $RiskThreshold `
        --threads $Threads 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
}
finally { $ErrorActionPreference = $oldPreference }
if ($exitCode -ne 0) { throw "AMC neuron-bank compilation failed. Log: $log" }
Copy-Item -LiteralPath $report -Destination $latest -Force
Write-Output "AMC neuron bank: $outputDir"
Write-Output "Compilation report: $report"
