param(
    [switch]$InstallToolchain,
    [switch]$InstallPythonDependencies,
    [switch]$SkipBuild
)

. (Join-Path $PSScriptRoot "Common.ps1")

Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig
$colibri = $config.colibri
$checkout = Resolve-RepositoryPath -Path ([string]$colibri.checkout)
$patchPath = Join-Path $script:OpsRoot "patches\colibri-olmoe-load-metrics.patch"
$expectedCommit = [string]$colibri.commit

function Get-RequiredCommand {
    param([Parameter(Mandatory = $true)][string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Required command is missing: $Name"
    }
    return $command
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Description
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

$git = Get-Command "git.exe" -ErrorAction SilentlyContinue
if ($null -eq $git -and $InstallToolchain) {
    $winget = Get-RequiredCommand -Name "winget.exe"
    Invoke-Checked -FilePath $winget.Source -Description "Git installation" -Arguments @(
        "install", "--exact", "--id", "Git.Git", "--silent",
        "--accept-package-agreements", "--accept-source-agreements"
    )
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:PATH = "$machinePath;$userPath"
    $git = Get-Command "git.exe" -ErrorAction SilentlyContinue
}
if ($null -eq $git) {
    throw "Git is missing. Re-run with -InstallToolchain or install Git first."
}

$checkoutParent = Split-Path $checkout -Parent
New-Item -ItemType Directory -Path $checkoutParent -Force | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $checkout ".git") -PathType Container)) {
    if (Test-Path -LiteralPath $checkout) {
        throw "Colibri checkout path exists but is not a Git repository: $checkout"
    }
    Invoke-Checked -FilePath $git.Source -Description "Colibri clone" -Arguments @(
        "clone", "--depth", "1", [string]$colibri.repository, $checkout
    )
}

$statusBefore = @(& $git.Source -C $checkout status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0) {
    throw "Cannot inspect Colibri checkout: $checkout"
}
$patchAlreadyApplied = $false
if ($statusBefore.Count -gt 0) {
    $changedFiles = @($statusBefore | ForEach-Object {
        if ($_.Length -ge 4) { $_.Substring(3).Trim() } else { $_.Trim() }
    } | Sort-Object -Unique)
    if ($changedFiles.Count -eq 1 -and $changedFiles[0] -eq "c/olmoe.c") {
        & $git.Source -C $checkout apply --reverse --check $patchPath 2>$null
        $patchAlreadyApplied = ($LASTEXITCODE -eq 0)
    }
    if (-not $patchAlreadyApplied) {
        throw "Colibri checkout contains unexpected local changes: $($statusBefore -join '; ')"
    }
}

$currentCommit = (& $git.Source -C $checkout rev-parse HEAD 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($currentCommit)) {
    $currentCommit = $null
}
if ($currentCommit -ne $expectedCommit) {
    if ($patchAlreadyApplied) {
        throw "Patched Colibri checkout is at $currentCommit, expected $expectedCommit."
    }
    Invoke-Checked -FilePath $git.Source -Description "Colibri pinned commit fetch" -Arguments @(
        "-C", $checkout, "fetch", "--depth", "1", "origin", $expectedCommit
    )
    Invoke-Checked -FilePath $git.Source -Description "Colibri pinned checkout" -Arguments @(
        "-C", $checkout, "checkout", "--detach", $expectedCommit
    )
}

if (-not $patchAlreadyApplied) {
    Invoke-Checked -FilePath $git.Source -Description "Colibri metrics patch check" -Arguments @(
        "-C", $checkout, "apply", "--check", $patchPath
    )
    Invoke-Checked -FilePath $git.Source -Description "Colibri metrics patch" -Arguments @(
        "-C", $checkout, "apply", $patchPath
    )
}

$systemPython = Get-PythonCommand
$venv = Resolve-RepositoryPath -Path ([string]$colibri.python_venv)
$venvPython = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    if (-not $InstallPythonDependencies) {
        throw "Colibri Python environment is missing: $venvPython. Re-run with -InstallPythonDependencies."
    }
    New-Item -ItemType Directory -Path (Split-Path $venv -Parent) -Force | Out-Null
    Invoke-Checked -FilePath $systemPython.Source -Description "Colibri virtual environment creation" -Arguments @(
        "-m", "venv", $venv
    )
}
$python = Get-Command $venvPython -ErrorAction Stop
function Test-PythonDependencies {
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5 can promote a native program's stderr to a
        # terminating NativeCommandError when the script preference is Stop.
        # Import failures are expected here and must be handled by exit code so
        # the installation branch below can run.
        $ErrorActionPreference = "Continue"
        & $python.Source -c "import torch, safetensors, huggingface_hub, transformers, accelerate, numpy" 2>&1 | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

$pythonDependenciesReady = Test-PythonDependencies
if (-not $pythonDependenciesReady -and $InstallPythonDependencies) {
    Invoke-Checked -FilePath $python.Source -Description "Python dependency installation" -Arguments @(
        "-m", "pip", "install", "torch", "safetensors",
        "huggingface_hub", "transformers", "accelerate", "numpy"
    )
    $pythonDependenciesReady = Test-PythonDependencies
}
if (-not $pythonDependenciesReady) {
    throw "Colibri Python dependencies are missing. Re-run with -InstallPythonDependencies."
}

$msysRoot = "C:\msys64"
$bashPath = Join-Path $msysRoot "usr\bin\bash.exe"
$makePath = Join-Path $msysRoot "usr\bin\make.exe"
$gccPath = Join-Path $msysRoot "ucrt64\bin\gcc.exe"
if ((-not (Test-Path -LiteralPath $bashPath)) -and $InstallToolchain) {
    $winget = Get-RequiredCommand -Name "winget.exe"
    Invoke-Checked -FilePath $winget.Source -Description "MSYS2 installation" -Arguments @(
        "install", "--exact", "--id", "MSYS2.MSYS2", "--silent",
        "--accept-package-agreements", "--accept-source-agreements"
    )
}
if ((-not (Test-Path -LiteralPath $bashPath)) -and -not $SkipBuild) {
    throw "MSYS2 is missing at $msysRoot. Re-run with -InstallToolchain."
}
if ((Test-Path -LiteralPath $bashPath) -and
    ((-not (Test-Path -LiteralPath $makePath)) -or (-not (Test-Path -LiteralPath $gccPath))) -and
    $InstallToolchain) {
    Invoke-Checked -FilePath $bashPath -Description "MSYS2 compiler installation" -Arguments @(
        "-lc", "pacman -Sy --needed --noconfirm mingw-w64-ucrt-x86_64-gcc make"
    )
}

$enginePath = Join-Path $checkout "c\olmoe.exe"
if (-not $SkipBuild) {
    if (-not (Test-Path -LiteralPath $makePath) -or -not (Test-Path -LiteralPath $gccPath)) {
        throw "MSYS2 UCRT64 gcc/make are missing. Re-run with -InstallToolchain."
    }
    $oldPath = $env:PATH
    try {
        $env:PATH = "$(Join-Path $msysRoot 'ucrt64\bin');$(Join-Path $msysRoot 'usr\bin');$oldPath"
        Push-Location (Join-Path $checkout "c")
        try {
            Invoke-Checked -FilePath $makePath -Description "Colibri OLMoE build" -Arguments @(
                "olmoe.exe", "ARCH=native"
            )
        }
        finally {
            Pop-Location
        }
    }
    finally {
        $env:PATH = $oldPath
    }
    if (-not (Test-Path -LiteralPath $enginePath -PathType Leaf)) {
        throw "Colibri build completed without producing $enginePath"
    }
}

$actualCommit = (& $git.Source -C $checkout rev-parse HEAD | Out-String).Trim()
$engineBuilt = Test-Path -LiteralPath $enginePath -PathType Leaf
$resolvedBash = if (Test-Path -LiteralPath $bashPath) { $bashPath } else { $null }
$resolvedMake = if (Test-Path -LiteralPath $makePath) { $makePath } else { $null }
$resolvedGcc = if (Test-Path -LiteralPath $gccPath) { $gccPath } else { $null }
$resolvedEngine = if ($engineBuilt) { $enginePath } else { $null }
$report = [PSCustomObject]@{
    schema_version = 1
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    repository = [string]$colibri.repository
    expected_commit = $expectedCommit
    actual_commit = $actualCommit
    checkout = $checkout
    metrics_patch = $patchPath
    metrics_patch_sha256 = Get-Sha256 -Path $patchPath
    python = $python.Source
    python_dependencies_ready = $pythonDependenciesReady
    toolchain = [PSCustomObject]@{
        msys_root = $msysRoot
        bash = $resolvedBash
        make = $resolvedMake
        gcc = $resolvedGcc
    }
    engine = [PSCustomObject]@{
        path = $resolvedEngine
        sha256 = Get-Sha256 -Path $enginePath
        built = $engineBuilt
    }
    cpu_only = $true
}
$artifact = Write-JsonArtifact -Value $report -Name "colibri-prepare-latest.json"
Write-Output "Colibri prepared at pinned commit $actualCommit"
Write-Output "Preparation artifact: $artifact"
