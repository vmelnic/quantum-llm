[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z]:[\\/].+')]
    [string]$ModelRoot,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$')]
    [string]$ArtifactName
)

$ErrorActionPreference = 'Stop'

$normalizedRoot = [IO.Path]::GetFullPath($ModelRoot)
$cacheRoot = [IO.Path]::GetFullPath((Join-Path $normalizedRoot '.session-cache'))
$target = [IO.Path]::GetFullPath((Join-Path $cacheRoot $ArtifactName))
if (-not $target.StartsWith($cacheRoot + [IO.Path]::DirectorySeparatorChar,
                            [StringComparison]::OrdinalIgnoreCase)) {
    throw 'refusing to clear a path outside MODEL_ROOT/.session-cache'
}

$removedBytes = 0L
if (Test-Path -LiteralPath $target) {
    $removedBytes = (Get-ChildItem -LiteralPath $target -File -Recurse -Force |
        Measure-Object -Property Length -Sum).Sum
    if ($null -eq $removedBytes) { $removedBytes = 0L }
    Remove-Item -LiteralPath $target -Recurse -Force
}

[pscustomobject]@{
    status = 'cleared'
    cache_root = $target
    removed_bytes = [Int64]$removedBytes
} | ConvertTo-Json -Depth 3
