param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [bool]$EnableCudaPinned = $true
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
    Invoke-CheckedNative -Command $cmake -Arguments @(
        "--preset", $preset,
        "-DEXPERT_RUNTIME_ENABLE_CUDA_PINNED=$cudaPinned"
    )
    Invoke-CheckedNative -Command $cmake -Arguments @(
        "--build", "--preset", $preset
    )
    Invoke-CheckedNative -Command $ctest -Arguments @(
        "--preset", $preset
    )
    Invoke-CheckedNative -Command $python -Arguments @(
        "-m", "unittest", "-v", "tests.compiler.test_expert_pack"
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
    build_root = Join-Path $script:RepoRoot "out\build\$preset"
    status = "pass"
}

$path = Write-JsonArtifact -Value $result -Name "expert-runtime-build-latest.json"
Write-Output "Expert runtime build/tests passed: $path"
