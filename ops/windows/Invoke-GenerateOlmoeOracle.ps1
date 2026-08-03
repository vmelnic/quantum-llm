param(
    [string]$Prompt = "The capital of France is",
    [int]$MaxNewTokens = 12,
    [int]$Threads = 6
)

. (Join-Path $PSScriptRoot "Common.ps1")

Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig
$modelId = [string]$config.models.recommended_lab_moe
$revision = [string]$config.models.recommended_lab_revision
$snapshot = Get-ModelSnapshot -ModelId $modelId -Revision $revision `
    -HubRoot $config.huggingface_hub -RequireComplete
$python = Get-ColibriPythonCommand
$generator = Join-Path $script:OpsRoot "python\generate_olmoe_oracle.py"
$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$outputPath = Join-Path (Join-Path $script:RepoRoot "artifacts") "olmoe-oracle-$stamp.json"
$latestPath = Join-Path (Join-Path $script:RepoRoot "artifacts") "olmoe-oracle-latest.json"
$logPath = Join-Path (Join-Path $script:RepoRoot "logs") "olmoe-oracle-$stamp.log"

$started = [DateTime]::UtcNow
$previousErrorActionPreference = $ErrorActionPreference
try {
    # Model loaders emit progress and non-fatal warnings on stderr. Preserve
    # those in the log and use Python's exit code as the success criterion.
    $ErrorActionPreference = "Continue"
    & $python.Source $generator `
        --snapshot $snapshot `
        --output $outputPath `
        --prompt $Prompt `
        --max-new-tokens $MaxNewTokens `
        --threads $Threads 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
$finished = [DateTime]::UtcNow
if ($exitCode -ne 0) {
    throw "OLMoE teacher oracle failed with code $exitCode. Log: $logPath"
}
Copy-Item -LiteralPath $outputPath -Destination $latestPath -Force

$summary = [PSCustomObject]@{
    schema_version = 1
    timestamp_utc = $finished.ToString("o")
    model_id = $modelId
    revision = $revision
    snapshot = $snapshot
    prompt = $Prompt
    max_new_tokens = $MaxNewTokens
    threads = $Threads
    duration_seconds = ($finished - $started).TotalSeconds
    oracle = $outputPath
    log = $logPath
    cpu_only = $true
}
$artifact = Write-JsonArtifact -Value $summary -Name "olmoe-oracle-run-latest.json"
Write-Output "Independent BF16 oracle written: $outputPath"
Write-Output "Oracle run artifact: $artifact"
