param([int]$Layer = 0, [int]$Threads = 6)

. (Join-Path $PSScriptRoot "Common.ps1")
Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig
$snapshot = Get-ModelSnapshot -ModelId ([string]$config.models.recommended_lab_moe) `
    -Revision ([string]$config.models.recommended_lab_revision) `
    -HubRoot $config.huggingface_hub -RequireComplete
$python = Get-ColibriPythonCommand
$runner = Join-Path $script:OpsRoot "python\run_amc_neuron_bank.py"
$compiled = Join-Path $script:RepoRoot "work\models\hesr-amc\neuron-bank-layer-$Layer"
$prompts = Join-Path $script:OpsRoot "config\amc_evaluation_prompts.json"
$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$output = Join-Path (Join-Path $script:RepoRoot "artifacts") "amc-neuron-bank-run-layer-$Layer-$stamp.json"
$latest = Join-Path (Join-Path $script:RepoRoot "artifacts") "amc-neuron-bank-run-layer-$Layer-latest.json"
$log = Join-Path (Join-Path $script:RepoRoot "logs") "amc-neuron-bank-run-layer-$Layer-$stamp.log"

$oldPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $python.Source $runner --snapshot $snapshot --compiled $compiled `
        --prompts $prompts --output $output --threads $Threads 2>&1 |
        Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
}
finally { $ErrorActionPreference = $oldPreference }
if ($exitCode -ne 0) { throw "AMC neuron-bank runtime failed. Log: $log" }
Copy-Item -LiteralPath $output -Destination $latest -Force
Write-Output "AMC neuron-bank runtime report: $output"
