param(
    [string]$ModelId,
    [string]$Revision,
    [switch]$Download
)

. (Join-Path $PSScriptRoot "Common.ps1")

Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig

if ([string]::IsNullOrWhiteSpace($ModelId)) {
    $ModelId = $config.models.recommended_lab_moe
}
if ([string]::IsNullOrWhiteSpace($Revision)) {
    $Revision = $config.models.recommended_lab_revision
}

$hf = Get-Command "hf.exe" -ErrorAction SilentlyContinue
if ($null -eq $hf) {
    $hf = Get-Command "hf" -ErrorAction SilentlyContinue
}
if ($null -eq $hf) {
    throw "Hugging Face CLI (hf) is not installed on 3090box."
}

$arguments = @(
    "download", $ModelId,
    "--revision", $Revision,
    "--cache-dir", $config.huggingface_hub,
    "--max-workers", "4",
    "--format", "json"
)
if (-not $Download) {
    $arguments += "--dry-run"
}

$mode = if ($Download) { "download" } else { "dry-run" }
Write-Output "Hugging Face $mode on 3090box: $ModelId@$Revision"
$output = @(& $hf.Source @arguments 2>&1)
$exitCode = $LASTEXITCODE

$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$result = [PSCustomObject]@{
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    mode = $mode
    model_id = $ModelId
    revision = $Revision
    cache_dir = $config.huggingface_hub
    max_workers = 4
    exit_code = $exitCode
    output = @($output | ForEach-Object { [string]$_ })
}
$artifact = Write-JsonArtifact -Value $result -Name "hf-$mode-$stamp.json"

$output | ForEach-Object { Write-Output $_ }
Write-Output "Download artifact: $artifact"
if ($exitCode -ne 0) {
    throw "hf download exited with code $exitCode"
}
