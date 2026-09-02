param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [int[]]$Layers = @(0, 21, 42),
    [int]$RowChunk = 128,
    [double]$MaximumWeightRelativeL2 = 0.02,
    [double]$MaximumProjectionRelativeL2 = 0.02,
    [double]$MinimumCosine = 0.999,
    [string]$Output = ""
)

. (Join-Path $PSScriptRoot "Common.ps1")

Initialize-ExperimentDirectories
if ($Layers.Count -eq 0 -or $RowChunk -lt 1 -or
    $MaximumWeightRelativeL2 -le 0 -or
    $MaximumProjectionRelativeL2 -le 0 -or
    $MinimumCosine -lt -1 -or $MinimumCosine -gt 1) {
    throw "Invalid DeepSeek INT8 organ quality gate"
}
$source = Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path (Join-Path $script:RepoRoot "artifacts") `
        "deepseek-int8-organ-quality-latest.json"
}
$resolvedOutput = [System.IO.Path]::GetFullPath($Output)
$partial = $resolvedOutput + ".partial"
$python = Get-PythonCommand
$arguments = @(
    "-m", "compiler", "qualify-deepseek-int8-organs",
    "--source", $source,
    "--layers"
) + @($Layers | ForEach-Object { [string]$_ }) + @(
    "--row-chunk", [string]$RowChunk,
    "--maximum-weight-relative-l2", [string]$MaximumWeightRelativeL2,
    "--maximum-projection-relative-l2", [string]$MaximumProjectionRelativeL2,
    "--minimum-cosine", [string]$MinimumCosine
)

Push-Location $script:RepoRoot
try {
    $json = (& $python.Source @arguments | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "DeepSeek INT8 organ quality command failed with exit code $LASTEXITCODE"
    }
    $document = $json | ConvertFrom-Json
    if ($document.ok -ne $true -or $null -eq $document.result -or
        $document.result.format -ne "deepseek-int8-organ-quality-v1") {
        throw "DeepSeek INT8 organ quality command returned an invalid document"
    }
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    [System.IO.File]::WriteAllText(
        $partial, $json + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $partial -Destination $resolvedOutput -Force
    Get-Content -LiteralPath $resolvedOutput -Raw
}
finally {
    Pop-Location
}
