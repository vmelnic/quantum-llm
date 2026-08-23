param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [Parameter(Mandatory = $true)][string]$ArtifactName,
    [int]$SamplesPerTensor = 8,
    [double]$MaximumRelativeL2 = 0.20,
    [double]$MinimumCosine = 0.98,
    [string]$Output = ""
)

. (Join-Path $PSScriptRoot "Common.ps1")

Initialize-ExperimentDirectories
if ([string]::IsNullOrWhiteSpace($env:MODEL_ROOT)) {
    throw "MODEL_ROOT is required"
}
if ($ArtifactName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "ArtifactName must be a stable name under MODEL_ROOT"
}
if ($SamplesPerTensor -lt 3 -or $MaximumRelativeL2 -le 0 -or
    $MinimumCosine -lt -1 -or $MinimumCosine -gt 1) {
    throw "Invalid numerical quality parameters"
}

$source = Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
$container = [System.IO.Path]::GetFullPath(
    (Join-Path $env:MODEL_ROOT $ArtifactName)
)
if (-not (Test-Path -LiteralPath $container -PathType Container)) {
    throw "Published artifact is absent: $container"
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path (Join-Path $script:RepoRoot "artifacts") `
        "fp4-source-quality-latest.json"
}
$resolvedOutput = [System.IO.Path]::GetFullPath($Output)
$partial = $resolvedOutput + ".partial"
$python = Get-PythonCommand
$arguments = @(
    "-m", "compiler", "qualify-fp4-container",
    "--source", $source,
    "--container", $container,
    "--samples-per-tensor", [string]$SamplesPerTensor,
    "--maximum-relative-l2", [string]$MaximumRelativeL2,
    "--minimum-cosine", [string]$MinimumCosine
)

Push-Location $script:RepoRoot
try {
    $json = (& $python.Source @arguments | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "FP4 source quality gate failed with exit code $LASTEXITCODE"
    }
    $document = $json | ConvertFrom-Json
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    [System.IO.File]::WriteAllText(
        $partial, $json + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $partial -Destination $resolvedOutput -Force
    if ($document.ok -ne $true -or $null -eq $document.result -or
        $document.result.valid -ne $true) {
        throw "FP4 source quality gate rejected the published artifact; see $resolvedOutput"
    }
    Get-Content -LiteralPath $resolvedOutput -Raw
}
finally {
    Pop-Location
}
