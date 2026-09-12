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
    $configureArguments = @(
        "--fresh", "--preset", $preset,
        "-DEXPERT_RUNTIME_ENABLE_CUDA_PINNED=$cudaPinned",
        "-DEXPERT_RUNTIME_ENABLE_CUDA_COMPUTE=$cudaCompute"
    )
    if ($EnableCudaCompute) {
        $nvcc = Get-Command "nvcc.exe" -ErrorAction Stop
        $cudaRoot = Split-Path (Split-Path $nvcc.Source -Parent) -Parent
        $env:NVCC_PREPEND_FLAGS = "--allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
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
        "-m", "unittest", "-v", "tests.compiler.test_deepseek_quant",
        "tests.compiler.test_expert_pack",
        "tests.server.test_expert_server",
        "tests.server.test_deepseek_route_oracle"
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
    build_root = Join-Path $script:RepoRoot "out\build\$preset"
    status = "pass"
}

$path = Write-JsonArtifact -Value $result -Name "expert-runtime-build-latest.json"
Write-Output "Expert runtime build/tests passed: $path"
