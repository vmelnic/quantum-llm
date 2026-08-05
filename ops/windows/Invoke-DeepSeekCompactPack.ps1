param(
    [string]$ModelId,
    [string]$Revision,
    [string]$Snapshot,
    [string]$RoutedCatalog,
    [string]$Output,
    [switch]$Resume
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories
$source = if ($Snapshot) {
    [System.IO.Path]::GetFullPath($Snapshot)
} else {
    if (-not $ModelId -or -not $Revision) {
        throw "Provide Snapshot, or both ModelId and Revision"
    }
    Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
}
$catalog = if ($RoutedCatalog) {
    [System.IO.Path]::GetFullPath($RoutedCatalog)
} else {
    Join-Path (Join-Path $script:RepoRoot "work") "deepseek-routed-catalog"
}
$destination = if ($Output) {
    [System.IO.Path]::GetFullPath($Output)
} else {
    Join-Path (Join-Path $script:RepoRoot "work") "deepseek-compact-pack"
}
if (-not (Test-Path (Join-Path $catalog "catalog.tsv") -PathType Leaf)) {
    throw "Generate the complete routed catalog before compact packing"
}
$python = Get-PythonCommand
$arguments = @(
    "-m", "compiler", "pack-deepseek-routed",
    "--catalog", $catalog,
    "--source", $source,
    "--output", $destination
)
if ($Resume) { $arguments += "--resume" }

Push-Location $script:RepoRoot
try {
    & $python.Source @arguments
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek compact packing failed" }
}
finally {
    Pop-Location
}
