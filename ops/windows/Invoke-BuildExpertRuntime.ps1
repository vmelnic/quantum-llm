param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [bool]$EnableCudaPinned = $true,
    [bool]$EnableCudaCompute = $true,
    [ValidateRange(1, 256)]
    [int]$ParallelJobs = [Environment]::ProcessorCount
)

. (Join-Path $PSScriptRoot "Common.ps1")

Initialize-ExperimentDirectories

$cmake = Get-Command "cmake.exe" -ErrorAction Stop
$ctest = Get-Command "ctest.exe" -ErrorAction Stop
$python = Get-PythonCommand
$preset = if ($Configuration -eq "Release") {
    "windows-msvc-release"
} else {
    "windows-msvc-debug"
}
$cudaPinned = if ($EnableCudaPinned) { "ON" } else { "OFF" }
$cudaCompute = if ($EnableCudaCompute) { "ON" } else { "OFF" }
$buildParent = [System.IO.Path]::GetFullPath(
    (Join-Path $script:RepoRoot "out\build"))
$buildRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $buildParent $preset))
if (-not $buildRoot.StartsWith(
    $buildParent + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Build root escapes the generated-output directory"
}

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Command.Source @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$($Command.Name) failed with exit code $LASTEXITCODE"
    }
}

Push-Location $script:RepoRoot
try {
    if (Test-Path -LiteralPath $buildRoot) {
        $buildEntry = Get-Item -LiteralPath $buildRoot -Force
        if (-not $buildEntry.PSIsContainer -or
            ($buildEntry.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
            throw "Build root is not an ordinary generated directory"
        }
        Remove-Item -LiteralPath $buildRoot -Recurse -Force
    }
    $configureArguments = @(
        "--fresh", "--preset", $preset,
        "-DEXPERT_RUNTIME_ENABLE_CUDA_PINNED=$cudaPinned",
        "-DEXPERT_RUNTIME_ENABLE_CUDA_COMPUTE=$cudaCompute"
    )
    if ($EnableCudaCompute) {
        $nvcc = Get-Command "nvcc.exe" -ErrorAction Stop
        $cudaRoot = Split-Path (Split-Path $nvcc.Source -Parent) -Parent
        $configureArguments += @("-T", "cuda=$cudaRoot")
    }
    Invoke-CheckedNative -Command $cmake -Arguments $configureArguments
    Invoke-CheckedNative -Command $cmake -Arguments @(
        "--build", "--preset", $preset, "--clean-first",
        "--parallel", $ParallelJobs.ToString()
    )
    Invoke-CheckedNative -Command $ctest -Arguments @(
        "--preset", $preset
    )
    Invoke-CheckedNative -Command $python -Arguments @(
        "-m", "unittest", "-v", "tests.compiler.test_expert_pack",
        "tests.server.test_expert_server"
    )
}
finally {
    Pop-Location
}

$result = [PSCustomObject]@{
    schema_version = 1
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    preset = $preset
    configuration = $Configuration
    cuda_pinned = $EnableCudaPinned
    cuda_compute = $EnableCudaCompute
    parallel_jobs = $ParallelJobs
    build_root = $buildRoot
    status = "pass"
}

$path = Write-JsonArtifact -Value $result -Name "expert-runtime-build-latest.json"
Write-Output "Expert runtime build/tests passed: $path"
