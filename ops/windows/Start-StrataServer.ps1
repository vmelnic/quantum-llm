param(
    [Parameter(Mandatory=$true)][string]$RuntimeHome,
    [Parameter(Mandatory=$true)][string]$ProfileName,
    [Parameter(Mandatory=$true)][string]$StartupLog
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:STRATA_DATA_DIR = $RuntimeHome
$env:STRATA_COMPONENTS_DIR = Join-Path $RuntimeHome 'components'
$env:PYTHONUTF8 = '1'
$cli = Join-Path $RuntimeHome 'terminal/current/bin/strata.cmd'
& $cli runtime start --profile $ProfileName --wait 600 --output json > $StartupLog
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# Keep the Task Scheduler owner alive for the complete managed process lifetime.
while ($true) {
    Start-Sleep -Seconds 5
    $lines = & $cli runtime status --profile $ProfileName --output json
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $state = ($lines -join "`n") | ConvertFrom-Json
    if ($state.state -ne 'ready') { break }
}
