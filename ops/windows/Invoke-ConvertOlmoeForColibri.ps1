param(
    [int]$FlushEvery = 0
)

. (Join-Path $PSScriptRoot "Common.ps1")

Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig
$modelId = [string]$config.models.recommended_lab_moe
$revision = [string]$config.models.recommended_lab_revision
$snapshot = Get-ModelSnapshot -ModelId $modelId -Revision $revision `
    -HubRoot $config.huggingface_hub -RequireComplete
$checkout = Resolve-RepositoryPath -Path ([string]$config.colibri.checkout)
$outputModel = Resolve-RepositoryPath -Path ([string]$config.colibri.converted_model)
$converter = Join-Path $checkout "c\tools\convert_olmoe_merged.py"
if (-not (Test-Path -LiteralPath $converter -PathType Leaf)) {
    throw "Colibri converter is missing. Run Invoke-PrepareColibri.ps1 first: $converter"
}

$git = Get-Command "git.exe" -ErrorAction Stop
$actualCommit = (& $git.Source -C $checkout rev-parse HEAD | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne [string]$config.colibri.commit) {
    throw "Colibri checkout is not at the pinned commit $($config.colibri.commit): $actualCommit"
}

$python = Get-ColibriPythonCommand
& $python.Source -c "import torch, safetensors, numpy" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Python requires torch and safetensors for the Colibri conversion."
}

if ($FlushEvery -le 0) {
    $FlushEvery = [int]$config.colibri.converter_flush_every
}
if ($FlushEvery -lt 2) {
    throw "FlushEvery must be at least 2."
}

New-Item -ItemType Directory -Path $outputModel -Force | Out-Null
$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$logPath = Join-Path (Join-Path $script:RepoRoot "logs") "colibri-convert-$stamp.log"
$started = [DateTime]::UtcNow
$output = @(& $python.Source $converter `
    --model $snapshot `
    --out $outputModel `
    --flush-every $FlushEvery 2>&1 | Tee-Object -FilePath $logPath)
$exitCode = $LASTEXITCODE
$finished = [DateTime]::UtcNow
if ($exitCode -ne 0) {
    throw "Colibri OLMoE conversion failed with code $exitCode. Log: $logPath"
}

$inspectionPath = Join-Path (Join-Path $script:RepoRoot "artifacts") "colibri-container-$stamp.json"
$inspectionLatest = Join-Path (Join-Path $script:RepoRoot "artifacts") "colibri-container-latest.json"
$inspector = Join-Path $script:OpsRoot "python\inspect_colibri_olmoe.py"
& $python.Source $inspector `
    --model $outputModel `
    --output $inspectionPath `
    --source-model $modelId `
    --source-revision $revision `
    --colibri-commit $actualCommit
if ($LASTEXITCODE -ne 0) {
    throw "Converted Colibri container failed validation: $inspectionPath"
}
Copy-Item -LiteralPath $inspectionPath -Destination $inspectionLatest -Force

$summary = [PSCustomObject]@{
    schema_version = 1
    timestamp_utc = $finished.ToString("o")
    source_model = $modelId
    source_revision = $revision
    source_snapshot = $snapshot
    output_model = $outputModel
    output_size_bytes = Get-DirectorySizeBytes -Path $outputModel
    colibri_commit = $actualCommit
    flush_every = $FlushEvery
    started_utc = $started.ToString("o")
    duration_seconds = ($finished - $started).TotalSeconds
    exit_code = $exitCode
    log = $logPath
    inspection = $inspectionPath
    cpu_only = $true
}
$artifact = Write-JsonArtifact -Value $summary -Name "colibri-conversion-latest.json"
Write-Output "Colibri model ready: $outputModel"
Write-Output "Container validation: $inspectionPath"
Write-Output "Conversion artifact: $artifact"
