param([switch]$CheckOnly)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$basePython = Get-PythonCommand
$environment = Join-Path $script:RepoRoot "work\memory-expert-venv"
$python = Join-Path $environment "Scripts\python.exe"
$cudaPackages = Join-Path $script:RepoRoot "work\venv\neural-cpu\Lib\site-packages"
if (-not (Test-Path -LiteralPath (Join-Path $cudaPackages "torch") -PathType Container)) {
    throw "The validated CUDA Torch packages are missing: $cudaPackages"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    if ($CheckOnly) { throw "Memory Expert environment is missing: $environment" }
    & $basePython.Source -m venv --system-site-packages $environment
    if ($LASTEXITCODE -ne 0) { throw "Failed to create Memory Expert environment" }
}

$environmentPackages = Join-Path $environment "Lib\site-packages"
$cudaBridge = Join-Path $environmentPackages "memory_expert_cuda.pth"
$bridgeStatement = "import sys; sys.path.insert(0, r'$cudaPackages')"
Set-Content -LiteralPath $cudaBridge -Value $bridgeStatement -Encoding ASCII

$validatedBeforeInstall = $false
& $python -c "import packaging, numpy, torch, transformers; assert torch.cuda.is_available()" `
    2>$null
if ($LASTEXITCODE -eq 0) { $validatedBeforeInstall = $true }

if ($CheckOnly -and -not $validatedBeforeInstall) {
    throw "Memory Expert Python environment validation failed"
}
if (-not $CheckOnly -and -not $validatedBeforeInstall) {
    # Reinstall the Python-only serving stack inside the PoW environment. The
    # host installation intentionally supplies CUDA Torch, but its optional
    # Transformers dependencies are not complete enough for model loading.
    & $python -m pip install --disable-pip-version-check --upgrade `
        "transformers==5.14.1" "accelerate==1.14.0" "packaging==25.0"
    if ($LASTEXITCODE -ne 0) { throw "Failed to install Memory Expert dependencies" }
}

& $python -c "import packaging, numpy, torch, transformers; print(torch.__version__); print(torch.version.cuda); print(transformers.__version__); assert torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) { throw "Memory Expert Python environment validation failed" }

[PSCustomObject]@{
    schema_version = 1
    environment = [System.IO.Path]::GetFullPath($environment)
    python = [System.IO.Path]::GetFullPath($python)
    system_site_packages = $true
    validated = $true
} | ConvertTo-Json
