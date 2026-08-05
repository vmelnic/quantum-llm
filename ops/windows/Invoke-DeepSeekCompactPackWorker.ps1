param(
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [Parameter(Mandatory = $true)][string]$RoutedCatalog,
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string]$StdoutPath,
    [Parameter(Mandatory = $true)][string]$StderrPath,
    [Parameter(Mandatory = $true)][string]$StatusPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$started = [DateTime]::UtcNow
$exitCode = 0
$failure = $null
try {
    & (Join-Path $PSScriptRoot "Invoke-DeepSeekCompactPack.ps1") `
        -Snapshot $Snapshot -RoutedCatalog $RoutedCatalog -Output $Output `
        -Resume *>> $StdoutPath
}
catch {
    $exitCode = 1
    $failure = $_.Exception.Message
    ($_ | Out-String) | Add-Content -LiteralPath $StderrPath -Encoding UTF8
}
finally {
    [PSCustomObject]@{
        schema_version = 1
        exit_code = $exitCode
        error = $failure
        started_utc = $started.ToString("o")
        finished_utc = [DateTime]::UtcNow.ToString("o")
        output = $Output
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}
exit $exitCode
