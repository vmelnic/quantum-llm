param([Parameter(Mandatory = $true)][string]$LaunchFile)

. (Join-Path $PSScriptRoot 'Common.ps1')
$launch = Get-Content -LiteralPath $LaunchFile -Raw | ConvertFrom-Json
if ($launch.schema -ne 'llama-launch-v1') { throw 'Invalid llama launch record' }
$env:CUDA_VISIBLE_DEVICES = '0'
$arguments = @($launch.arguments | ForEach-Object { '"' + ([string]$_).Replace('"', '\"') + '"' })
$process = Start-Process -FilePath $launch.binary -ArgumentList $arguments -PassThru `
    -RedirectStandardOutput $launch.stdout -RedirectStandardError $launch.stderr
@{schema = 'llama-process-v1'; pid = $process.Id; model = $launch.model;
    binary = $launch.binary; port = $launch.port; started_utc = [DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json | Set-Content (Join-Path $script:RepoRoot 'artifacts/llama-server-process.json') -Encoding UTF8
$process.WaitForExit()
exit $process.ExitCode
