param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [string]$Snapshot,
    [string[]]$Names = @(
        "layers.0.attn.wkv",
        "layers.0.attn.wq_a",
        "layers.0.attn.wq_b",
        "layers.0.attn.wo_a",
        "layers.0.attn.wo_b"
    ),
    [int[]]$Bits = @(4, 5, 6),
    [int[]]$BlockSizes = @(32, 64, 128),
    [int]$RowChunk = 128
)

. (Join-Path $PSScriptRoot "Common.ps1")

Initialize-ExperimentDirectories
$source = if ($Snapshot) {
    [System.IO.Path]::GetFullPath($Snapshot)
} else {
    Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
}
$python = Get-PythonCommand
$screens = @()

Push-Location $script:RepoRoot
try {
    foreach ($name in $Names) {
        $arguments = @(
            "-m", "compiler", "qualify-deepseek-compact-matrix",
            "--source", $source,
            "--name", $name,
            "--row-chunk", [string]$RowChunk,
            "--bits"
        ) + ($Bits | ForEach-Object { [string]$_ }) + @(
            "--block-sizes"
        ) + ($BlockSizes | ForEach-Object { [string]$_ })
        $raw = & $python.Source @arguments | Out-String
        if ($LASTEXITCODE -ne 0) {
            Write-Output $raw
            throw "Compact dense screen failed for $name"
        }
        $decoded = $raw | ConvertFrom-Json
        $screens += $decoded.result
    }
}
finally {
    Pop-Location
}

$result = [PSCustomObject]@{
    schema_version = 1
    model_id = $ModelId
    revision = $Revision
    screens = $screens
}
$artifact = Write-JsonArtifact -Value $result `
    -Name "deepseek-compact-dense-screen-latest.json"
Write-Output ($result | ConvertTo-Json -Depth 8)
Write-Output "Compact dense screen artifact: $artifact"
