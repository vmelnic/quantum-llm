param(
    [int]$Layer = 0,
    [int]$Clusters = 4,
    [int]$MaxRank = 16,
    [switch]$AlignNeurons
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
$experiment = Join-Path $script:OpsRoot "python\factorize_moe_layer.py"
$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$variant = if ($AlignNeurons) { "aligned" } else { "unaligned" }
$output = Join-Path (Join-Path $script:RepoRoot "artifacts") "factorization-$variant-layer-$Layer-$stamp.json"
$latest = Join-Path (Join-Path $script:RepoRoot "artifacts") "factorization-$variant-layer-$Layer-latest.json"

$arguments = @(
    $experiment,
    "--snapshot", $snapshot,
    "--layer", $Layer,
    "--clusters", $Clusters,
    "--max-rank", $MaxRank,
    "--output", $output
)
if ($AlignNeurons) {
    $arguments += "--align-neurons"
}
& $python.Source @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Layer factorization exited with code $LASTEXITCODE"
}

Copy-Item -LiteralPath $output -Destination $latest -Force
Write-Output "Factorization written: $output"
Write-Output "Latest factorization: $latest"
Write-Output "CPU-only policy: CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES"
