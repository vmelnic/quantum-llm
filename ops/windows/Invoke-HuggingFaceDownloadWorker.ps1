param(
    [Parameter(Mandatory = $true)][string]$HfPath,
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [string]$Files = "",
    [Parameter(Mandatory = $true)][int]$MaxWorkers,
    [Parameter(Mandatory = $true)][string]$StdoutPath,
    [Parameter(Mandatory = $true)][string]$StderrPath,
    [Parameter(Mandatory = $true)][string]$StatusPath
)

$ErrorActionPreference = "Stop"
$started = [DateTime]::UtcNow
$exitCode = 1
$failure = $null
$integrityChecked = $false
try {
    # huggingface_hub uses hf_xet automatically when the package is installed.
    # High-performance mode increases concurrency without changing the Hub cache
    # layout, resumability, or Xet content verification.
    $env:HF_XET_HIGH_PERFORMANCE = "1"
    $selectedFiles = @($Files -split ',' | Where-Object { $_ } | ForEach-Object {
        $_.Trim().Replace('\', '/')
    })
    if ($selectedFiles.Count -eq 0) {
        # Safetensors downloads publish the immutable model contract before
        # large shard transfers. Exact-file downloads already carry their
        # complete immutable contract in the selected path list.
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
    }
    $arguments = @("download", $ModelId)
    if ($selectedFiles.Count -gt 0) {
        $arguments += $selectedFiles
    }
    $arguments += @(
        "--revision", $Revision,
        "--max-workers", [string]$MaxWorkers,
        "--no-truncate"
    )
    $process = Start-Process -FilePath $HfPath -ArgumentList $arguments `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath
    $exitCode = [int]$process.ExitCode
    if ($exitCode -ne 0) {
        $failure = "hf download exited with code $exitCode"
    } elseif ($selectedFiles.Count -eq 0) {
        # Xet can leave zero-byte transfer markers after publishing a fully
        # verified snapshot. Verify the pinned revision before removing only
        # those empty markers; non-empty partials remain resumable.
        $verifyStdout = "$StdoutPath.verify"
        $verifyStderr = "$StderrPath.verify"
        $verifyArguments = @(
            "cache", "verify", $ModelId,
            "--revision", $Revision,
            "--fail-on-missing-files",
            "--format", "quiet"
        )
        $verifyProcess = Start-Process -FilePath $HfPath `
            -ArgumentList $verifyArguments -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $verifyStdout `
            -RedirectStandardError $verifyStderr
        if ([int]$verifyProcess.ExitCode -ne 0) {
            $exitCode = [int]$verifyProcess.ExitCode
            $failure = "hf cache verify exited with code $exitCode"
        } else {
            $integrityChecked = $true
            $cacheName = "models--" + ($ModelId -replace "/", "--")
            $cacheRoot = Join-Path `
                (Join-Path $env:USERPROFILE ".cache\huggingface\hub") `
                $cacheName
            Get-ChildItem (Join-Path $cacheRoot "blobs") `
                    -Filter "*.incomplete" -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Length -eq 0 } |
                Remove-Item -Force
        }
    }
}
catch {
    $failure = $_.Exception.Message
}
finally {
    $temporary = "$StatusPath.tmp"
    [PSCustomObject]@{
        schema_version = 3
        model_id = $ModelId
        revision = $Revision
        files = $selectedFiles
        xet_high_performance = $true
        integrity_checked = $integrityChecked
        started_utc = $started.ToString("o")
        finished_utc = [DateTime]::UtcNow.ToString("o")
        exit_code = $exitCode
        error = $failure
    } | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $StatusPath -Force
}
exit $exitCode
