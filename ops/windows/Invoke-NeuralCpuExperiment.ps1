param(
    [Parameter(Mandatory = $true)][string]$TraceDirectory,
    [Parameter(Mandatory = $true)][string]$PromptManifest,
    [Parameter(Mandatory = $true)][string]$Output,
    [int]$Epochs = 12,
    [int]$BatchSize = 128,
    [int]$Width = 216,
    [int]$Atoms = 8,
    [int]$Rank = 16,
    [int]$MaximumCycles = 32,
    [string]$Checkpoints = "1,2,4,8,16,32"
)

. (Join-Path $PSScriptRoot "Common.ps1")

$python = Join-Path $script:RepoRoot "work\venv\neural-cpu\Scripts\python.exe"
$validator = Join-Path $script:RepoRoot `
    "experiments\neural_cpu\validate_trace.py"
$experiment = Join-Path $script:RepoRoot `
    "experiments\neural_cpu\experiment.py"
$probe = Join-Path $script:RepoRoot "experiments\neural_cpu\cuda_probe.py"
foreach ($path in @(
    $python, $probe, $validator, $experiment, $TraceDirectory, $PromptManifest
)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path is missing: $path" }
}
if ($Epochs -le 0 -or $BatchSize -le 0 -or $Width -le 0 -or
    $Atoms -le 1 -or $Rank -le 0 -or $MaximumCycles -le 0) {
    throw "Invalid Neural CPU experiment setting"
}

& $python $probe
if ($LASTEXITCODE -ne 0) { throw "Neural CPU CUDA environment is unavailable" }
& $python $validator $TraceDirectory
if ($LASTEXITCODE -ne 0) { throw "Neural CPU trace validation failed" }
& $python $experiment `
    --trace $TraceDirectory `
    --prompts-manifest $PromptManifest `
    --output $Output `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --width $Width `
    --atoms $Atoms `
    --rank $Rank `
    --maximum-cycles $MaximumCycles `
    --checkpoints $Checkpoints
if ($LASTEXITCODE -ne 0) { throw "Neural CPU experiment failed" }
