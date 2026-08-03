param(
    [int]$Layer = 0,
    [int]$Clusters = 4,
    [int]$Rank = 16,
    [double]$SparseFraction = 0.01
)

. (Join-Path $PSScriptRoot "Common.ps1")

Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig

$modelId = [string]$config.models.recommended_lab_moe
$revision = [string]$config.models.recommended_lab_revision
$snapshot = Get-ModelSnapshot -ModelId $modelId -Revision $revision `
    -HubRoot $config.huggingface_hub -RequireComplete

$python = Get-PythonCommand
$experiment = Join-Path $script:OpsRoot "python\evaluate_layer_activations.py"
$prompts = Join-Path $script:OpsRoot "config\calibration_prompts.json"
$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$output = Join-Path (Join-Path $script:RepoRoot "artifacts") "activation-eval-layer-$Layer-$stamp.json"
$latest = Join-Path (Join-Path $script:RepoRoot "artifacts") "activation-eval-layer-$Layer-latest.json"

& $python.Source $experiment `
    --snapshot $snapshot `
    --prompts $prompts `
    --layer $Layer `
    --clusters $Clusters `
    --rank $Rank `
    --sparse-fraction $SparseFraction `
    --output $output
if ($LASTEXITCODE -ne 0) {
    throw "Activation evaluation exited with code $LASTEXITCODE"
}

Copy-Item -LiteralPath $output -Destination $latest -Force
Write-Output "Activation evaluation written: $output"
Write-Output "Latest activation evaluation: $latest"
Write-Output "CPU-only policy: CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES"
