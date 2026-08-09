param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Compile", "Validate")]
    [string]$Action,
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [string]$Output,
    [string]$SourceId = "local-checkpoint",
    [string]$SourceRevision = "local",
    [string]$Adapter = "olmoe",
    [ValidateSet("int8-symmetric-per-row-v1", "fp4-e2m1-ue8m0-block32-v1")]
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
            "--quant-profile", $QuantProfile,
            "--max-expert-pack-bytes", [string]$MaxExpertPackBytes
        )
        if ($Resume) { $arguments += "--resume" }
        if ($ReclaimSourceShards) { $arguments += "--reclaim-source-shards" }
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
