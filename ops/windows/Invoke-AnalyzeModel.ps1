. (Join-Path $PSScriptRoot "Common.ps1")

Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig

$modelId = [string]$config.models.recommended_lab_moe
$revision = [string]$config.models.recommended_lab_revision
$snapshot = Get-ModelSnapshot -ModelId $modelId -Revision $revision `
    -HubRoot $config.huggingface_hub -RequireComplete

$python = Get-PythonCommand
$analyzer = Join-Path $script:OpsRoot "python\analyze_safetensors.py"
$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$output = Join-Path (Join-Path $script:RepoRoot "artifacts") "model-analysis-$stamp.json"
$latest = Join-Path (Join-Path $script:RepoRoot "artifacts") "model-analysis-latest.json"

& $python.Source $analyzer --snapshot $snapshot --output $output
if ($LASTEXITCODE -ne 0) {
    throw "SafeTensors analyzer exited with code $LASTEXITCODE"
}

Copy-Item -LiteralPath $output -Destination $latest -Force
Write-Output "Model analysis written: $output"
Write-Output "Latest model analysis: $latest"
Write-Output "CPU-only policy: CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES"
