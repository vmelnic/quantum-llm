param(
    [string]$Python = "",
    [string]$TorchVersion = "2.11.0",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130",
    [switch]$CheckOnly
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$environmentRoot = Join-Path $script:RepoRoot "work\venv\neural-cpu"
$environmentPython = Join-Path $environmentRoot "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $environmentPython)) {
    if ($CheckOnly) { throw "Neural CPU Python environment is missing" }
    $base = if ($Python) {
        (Get-Command $Python -ErrorAction Stop).Source
    } else {
        (Get-PythonCommand).Source
    }
    & $base -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to create Neural CPU environment" }
}

if (-not $CheckOnly) {
    & $environmentPython -m pip install --disable-pip-version-check --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to update pip" }
    & $environmentPython -m pip install --disable-pip-version-check numpy
    if ($LASTEXITCODE -ne 0) { throw "Failed to install NumPy" }
    & $environmentPython -m pip install --disable-pip-version-check `
        "torch==$TorchVersion" --index-url $TorchIndexUrl
    if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA PyTorch" }
}

$probe = Join-Path $script:RepoRoot "experiments\neural_cpu\cuda_probe.py"
if (-not (Test-Path -LiteralPath $probe)) { throw "CUDA probe script is missing" }
& $environmentPython $probe
if ($LASTEXITCODE -ne 0) { throw "Neural CPU CUDA probe failed" }
