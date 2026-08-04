param(
    [Parameter(Mandatory = $true)][string]$HfPath,
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][int]$MaxWorkers,
    [Parameter(Mandatory = $true)][string]$StatusPath
)

$ErrorActionPreference = "Stop"
$started = [DateTime]::UtcNow
$exitCode = 1
$failure = $null
try {
    & $HfPath download $ModelId --max-workers $MaxWorkers
    $exitCode = [int]$LASTEXITCODE
    if ($exitCode -ne 0) { $failure = "hf download exited with code $exitCode" }
}
catch {
    $failure = $_.Exception.Message
    if ($LASTEXITCODE -is [int] -and $LASTEXITCODE -ne 0) {
        $exitCode = [int]$LASTEXITCODE
    }
}
finally {
    $temporary = "$StatusPath.tmp"
    [PSCustomObject]@{
        schema_version = 1
        model_id = $ModelId
        started_utc = $started.ToString("o")
        finished_utc = [DateTime]::UtcNow.ToString("o")
        exit_code = $exitCode
        error = $failure
    } | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $StatusPath -Force
}
exit $exitCode
