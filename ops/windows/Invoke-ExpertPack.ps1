param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Compile", "Validate", "RefreshModelProgram", "RefreshSamplingProfiles")]
    [string]$Action,
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [string]$Source = "",
    [string]$Output,
    [string]$SourceId = "local-checkpoint",
    [string]$SourceRevision = "local",
    [string]$Adapter = "olmoe",
    [string]$SamplingProfiles = "",
    [string]$ConfigFile = "config.json",
    [string]$IndexFile = "model.safetensors.index.json",
    [ValidateSet("int8-symmetric-per-row-v1", "fp4-e2m1-ue8m0-block32-v1", "nvfp4-e2m1-e4m3fn-block16-w4a4-v1")]
    [string]$QuantProfile = "int8-symmetric-per-row-v1",
    [int64]$MaxExpertPackBytes = 4GB,
    [switch]$Resume,
    [switch]$ReclaimSourceShards
)

. (Join-Path $PSScriptRoot "Common.ps1")

Initialize-ExperimentDirectories
$python = Get-PythonCommand
$resolvedPath = [System.IO.Path]::GetFullPath($Path)

Push-Location $script:RepoRoot
try {
    if ($Action -eq "Compile") {
        if ([string]::IsNullOrWhiteSpace($Output)) {
            throw "-Output is required for Compile"
        }
        $resolvedOutput = [System.IO.Path]::GetFullPath($Output)
        $arguments = @(
            "-m", "compiler", "compile",
            "--source", $resolvedPath,
            "--output", $resolvedOutput,
            "--source-id", $SourceId,
            "--source-revision", $SourceRevision,
            "--adapter", $Adapter,
            "--config-file", $ConfigFile,
            "--index-file", $IndexFile,
            "--quant-profile", $QuantProfile,
            "--max-expert-pack-bytes", [string]$MaxExpertPackBytes
        )
        if ($Resume) { $arguments += "--resume" }
        if ($ReclaimSourceShards) { $arguments += "--reclaim-source-shards" }
        if (-not [string]::IsNullOrWhiteSpace($SamplingProfiles)) {
            $arguments += @(
                "--sampling-profiles",
                [System.IO.Path]::GetFullPath($SamplingProfiles)
            )
        }
    }
    elseif ($Action -eq "RefreshModelProgram") {
        if ([string]::IsNullOrWhiteSpace($Output) -or
            [string]::IsNullOrWhiteSpace($Source)) {
            throw "-Output and -Source are required for RefreshModelProgram"
        }
        $resolvedOutput = [System.IO.Path]::GetFullPath($Output)
        $resolvedSource = [System.IO.Path]::GetFullPath($Source)
        $arguments = @(
            "-m", "compiler", "refresh-model-program",
            "--container", $resolvedPath,
            "--output", $resolvedOutput,
            "--source", $resolvedSource,
            "--adapter", $Adapter,
            "--config-file", $ConfigFile,
            "--index-file", $IndexFile
        )
    }
    elseif ($Action -eq "RefreshSamplingProfiles") {
        if ([string]::IsNullOrWhiteSpace($Output) -or
            [string]::IsNullOrWhiteSpace($SamplingProfiles)) {
            throw "-Output and -SamplingProfiles are required for RefreshSamplingProfiles"
        }
        $resolvedOutput = [System.IO.Path]::GetFullPath($Output)
        $resolvedProfiles = [System.IO.Path]::GetFullPath($SamplingProfiles)
        $arguments = @(
            "-m", "compiler", "refresh-sampling-profiles",
            "--container", $resolvedPath,
            "--output", $resolvedOutput,
            "--profiles", $resolvedProfiles
        )
    }
    else {
        $arguments = @("-m", "compiler", "validate", $resolvedPath)
    }

    & $python.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Expert Pack $Action failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
