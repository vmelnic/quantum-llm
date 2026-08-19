param(
    [string]$Python = "",
    [switch]$CheckOnly
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$requirements = Join-Path $script:RepoRoot "requirements\server.txt"
if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
    throw "Server requirements are missing: $requirements"
}

$pythonPath = ""
if ($Python) {
    $command = Get-Command $Python -ErrorAction Stop
    $pythonPath = $command.Source
} else {
    foreach ($candidate in @(
        (Join-Path $script:RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $script:RepoRoot "work\venv\server\Scripts\python.exe")
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $pythonPath = [System.IO.Path]::GetFullPath($candidate)
            break
        }
    }
}

if (-not $pythonPath) {
    if ($CheckOnly) {
        throw "Server Python environment is missing; run model.sh install"
    }
    $pythonPath = Join-Path $script:RepoRoot "work\venv\server\Scripts\python.exe"
    $basePython = Get-PythonCommand
    & $basePython.Source -m venv (Split-Path (Split-Path $pythonPath -Parent) -Parent)
    if ($LASTEXITCODE -ne 0) { throw "Failed to create server Python environment" }
}

if (-not $CheckOnly) {
    & $pythonPath -m pip install --disable-pip-version-check -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Failed to install server requirements" }
}

$pins = foreach ($line in Get-Content -LiteralPath $requirements) {
    if ($line -match '^\s*([A-Za-z0-9_.-]+)==([^\s#]+)') {
        [PSCustomObject]@{ Name = $Matches[1]; Version = $Matches[2] }
    }
}
foreach ($pin in $pins) {
    $probeVersion = "import importlib.metadata as m; print(m.version('$($pin.Name)'))"
    $actual = [string](& $pythonPath -c $probeVersion)
    if ($LASTEXITCODE -ne 0 -or $actual.Trim() -ne $pin.Version) {
        throw "Server dependency $($pin.Name) must be $($pin.Version); installed=$($actual.Trim())"
    }
}

$probe = "import jinja2, transformers; print(jinja2.__version__); print(transformers.__version__)"
$versionLines = @(& $pythonPath -c $probe)
if ($LASTEXITCODE -ne 0) {
    throw "Server Python dependencies are incomplete; run model.sh install"
}
if ($versionLines.Count -ne 2) {
    throw "Server dependency version probe returned unexpected output"
}

[PSCustomObject]@{
    status = if ($CheckOnly) { "ready" } else { "installed" }
    python = $pythonPath
    versions = [PSCustomObject]@{
        jinja2 = [string]$versionLines[0]
        transformers = [string]$versionLines[1]
    }
} | ConvertTo-Json -Depth 4
