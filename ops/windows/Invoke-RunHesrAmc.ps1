param(
    [int]$Layer = 0,
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
$runner = Join-Path $script:OpsRoot "python\run_hesr_amc_layer.py"
$compiled = Join-Path $script:RepoRoot "work\models\hesr-amc\olmoe-layer-$Layer"
$prompts = Join-Path $script:OpsRoot "config\amc_evaluation_prompts.json"
$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$output = Join-Path (Join-Path $script:RepoRoot "artifacts") "hesr-amc-run-layer-$Layer-$stamp.json"
$latest = Join-Path (Join-Path $script:RepoRoot "artifacts") "hesr-amc-run-layer-$Layer-latest.json"
$log = Join-Path (Join-Path $script:RepoRoot "logs") "hesr-amc-run-layer-$Layer-$stamp.log"

$previousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $python.Source $runner `
        --snapshot $snapshot `
        --compiled $compiled `
        --prompts $prompts `
        --output $output `
        --threads $Threads 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
if ($exitCode -ne 0) {
    throw "HESR-AMC runtime failed with code $exitCode. Log: $log"
}
Copy-Item -LiteralPath $output -Destination $latest -Force
Write-Output "HESR-AMC runtime report: $output"
