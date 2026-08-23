param(
    [string]$Python = "",
    [string]$TorchVersion = "2.11.0",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130",
    [switch]$CheckOnly
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$environmentRoot = Join-Path $script:RepoRoot "work\venv\hf-reference"
$environmentPython = Join-Path $environmentRoot "Scripts\python.exe"
$requirements = Join-Path $script:RepoRoot "requirements\reference.txt"
if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
    throw "Reference requirements are missing: $requirements"
}
if (-not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
    if ($CheckOnly) { throw "Hugging Face reference environment is missing" }
    $base = if ($Python) {
        (Get-Command $Python -ErrorAction Stop).Source
    } else {
        (Get-PythonCommand).Source
    }
    & $base -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to create reference environment" }
}

if (-not $CheckOnly) {
    & $environmentPython -m pip install --disable-pip-version-check --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to update reference pip" }
    & $environmentPython -m pip install --disable-pip-version-check `
        "torch==$TorchVersion" --index-url $TorchIndexUrl
    if ($LASTEXITCODE -ne 0) { throw "Failed to install reference CUDA PyTorch" }
    & $environmentPython -m pip install --disable-pip-version-check `
        -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Failed to install reference dependencies" }
}

$probe = @'
import torch, transformers, accelerate, safetensors
assert torch.cuda.is_available(), "reference PyTorch has no CUDA"
assert hasattr(transformers, "AutoModelForMultimodalLM")
print(torch.__version__)
print(transformers.__version__)
print(accelerate.__version__)
print(safetensors.__version__)
print(torch.cuda.get_device_name(0))
'@
& $environmentPython -c $probe
if ($LASTEXITCODE -ne 0) { throw "Reference environment probe failed" }

