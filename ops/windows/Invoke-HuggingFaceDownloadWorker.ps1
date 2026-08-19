param(
    [Parameter(Mandatory = $true)][string]$HfPath,
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [Parameter(Mandatory = $true)][int]$MaxWorkers,
    [Parameter(Mandatory = $true)][string]$StdoutPath,
    [Parameter(Mandatory = $true)][string]$StderrPath,
    [Parameter(Mandatory = $true)][string]$StatusPath
)

$ErrorActionPreference = "Stop"
$started = [DateTime]::UtcNow
$exitCode = 1
$failure = $null
try {
    # huggingface_hub uses hf_xet automatically when the package is installed.
    # High-performance mode increases concurrency without changing the Hub cache
    # layout, resumability, or Xet content verification.
    $env:HF_XET_HIGH_PERFORMANCE = "1"
    # Publish the immutable model contract before starting large shard
    # transfers. This keeps architecture work and source validation from
    # depending on the Hub client's internal file ordering while retaining a
    # single pinned, resumable Xet workflow.
    $metadataArguments = @(
        "download", $ModelId,
        "config.json", "model.safetensors.index.json",
        "--revision", $Revision,
        "--max-workers", "1",
        "--no-truncate"
    )
    $metadataStdout = "$StdoutPath.metadata"
    $metadataStderr = "$StderrPath.metadata"
    $metadataProcess = Start-Process -FilePath $HfPath `
        -ArgumentList $metadataArguments -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $metadataStdout `
        -RedirectStandardError $metadataStderr
    if ([int]$metadataProcess.ExitCode -ne 0) {
        throw "hf metadata download exited with code $($metadataProcess.ExitCode)"
    }
    $arguments = @(
        "download", $ModelId,
        "--revision", $Revision,
        "--max-workers", [string]$MaxWorkers,
        "--no-truncate"
    )
    $process = Start-Process -FilePath $HfPath -ArgumentList $arguments `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath
    $exitCode = [int]$process.ExitCode
    if ($exitCode -ne 0) { $failure = "hf download exited with code $exitCode" }
}
catch {
    $failure = $_.Exception.Message
}
finally {
    $temporary = "$StatusPath.tmp"
    [PSCustomObject]@{
        schema_version = 1
        model_id = $ModelId
        revision = $Revision
        xet_high_performance = $true
        started_utc = $started.ToString("o")
        finished_utc = [DateTime]::UtcNow.ToString("o")
        exit_code = $exitCode
        error = $failure
    } | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $StatusPath -Force
}
exit $exitCode
