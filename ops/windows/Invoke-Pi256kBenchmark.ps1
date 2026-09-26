param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('prepare-pi', 'start-download', 'download-status', 'publish',
                 'build-llama', 'preflight', 'launch', 'status')]
    [string]$Action,
    [ValidateSet('quantum', 'llama')][string]$Backend = 'quantum'
)

$ErrorActionPreference = 'Stop'
$tools = Join-Path $PSScriptRoot 'benchmarks/pi256k'
if ($Action -in @('start-download', 'publish', 'preflight', 'launch') -and
    [string]::IsNullOrWhiteSpace($env:MODEL_ROOT)) {
    throw 'MODEL_ROOT must be configured for this action'
}
switch ($Action) {
    'prepare-pi' { & (Join-Path $tools 'Prepare-Pi.ps1') }
    'start-download' { & (Join-Path $tools 'Start-GgufDownload.ps1') -ModelRoot $env:MODEL_ROOT }
    'download-status' { & (Join-Path $PSScriptRoot 'Get-HuggingFaceModelDownload.ps1') }
    'publish' { & (Join-Path $tools 'Publish-Gguf.ps1') -ModelRoot $env:MODEL_ROOT }
    'build-llama' { & (Join-Path $tools 'Build-Llama.ps1') }
    'preflight' { & (Join-Path $tools 'Preflight.ps1') -ModelRoot $env:MODEL_ROOT -Backend $Backend }
    'launch' { & (Join-Path $tools 'Launch.ps1') -ModelRoot $env:MODEL_ROOT -Backend $Backend }
    'status' { & (Join-Path $tools 'Status.ps1') -Backend $Backend }
}
