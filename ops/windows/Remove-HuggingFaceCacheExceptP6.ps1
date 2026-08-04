param(
    [string]$HubRoot = "C:\Users\vladi\.cache\huggingface\hub",
    [switch]$ConfirmDeletion
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if (-not $ConfirmDeletion) {
    throw "This cleanup is destructive; pass -ConfirmDeletion after operator approval"
}
$root = [System.IO.Path]::GetFullPath($HubRoot).TrimEnd('\')
if (-not (Test-Path -LiteralPath $root -PathType Container)) {
    throw "Hugging Face hub root is missing: $root"
}
$keep = @(
    "models--allenai--OLMoE-1B-7B-0125-Instruct",
    "models--Qwen--Qwen3-Next-80B-A3B-Instruct"
)
foreach ($name in $keep) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $name) -PathType Container)) {
        throw "Required retained cache is missing: $name"
    }
}

$targets = @(Get-ChildItem -LiteralPath $root -Directory | Where-Object {
    ($_.Name -like "models--*" -or $_.Name -like "datasets--*") -and
    $_.Name -notin $keep
} | Sort-Object Name)
$deleted = @()
foreach ($target in $targets) {
    $full = [System.IO.Path]::GetFullPath($target.FullName).TrimEnd('\')
    if ([System.IO.Path]::GetDirectoryName($full) -ne $root -or
        ($target.Name -notlike "models--*" -and $target.Name -notlike "datasets--*")) {
        throw "Refusing unsafe cache target: $full"
    }
    $bytes = Get-DirectorySizeBytes -Path $full
    Remove-Item -LiteralPath $full -Recurse -Force -ErrorAction Stop
    $lockPath = Join-Path (Join-Path $root ".locks") $target.Name
    if (Test-Path -LiteralPath $lockPath) {
        Remove-Item -LiteralPath $lockPath -Recurse -Force -ErrorAction Stop
    }
    $deleted += [PSCustomObject]@{
        name = $target.Name
        path = $full
        bytes = $bytes
    }
}

$remainingCaches = @(Get-ChildItem -LiteralPath $root -Directory | Where-Object {
    $_.Name -like "models--*" -or $_.Name -like "datasets--*"
} | Select-Object -ExpandProperty Name | Sort-Object)
if (@($remainingCaches | Where-Object { $_ -notin $keep }).Count -ne 0 -or
    @($keep | Where-Object { $_ -notin $remainingCaches }).Count -ne 0) {
    throw "Post-cleanup cache set does not match the retention contract"
}
$result = [PSCustomObject]@{
    schema_version = 1
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    hub_root = $root
    retained = $keep
    deleted = $deleted
    deleted_bytes = [int64](($deleted | Measure-Object bytes -Sum).Sum)
    remaining_caches = $remainingCaches
}
$path = Write-JsonArtifact -Value $result -Name "hf-cache-cleanup-latest.json"
$result | ConvertTo-Json -Depth 6
Write-Output "Hugging Face cache cleanup artifact: $path"
