# Offline contract: integrated MTP arguments and prepare must never invoke
# a serving lifecycle operation, health request, or process inspection.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
. (Join-Path $repo 'ops/windows/LlamaSpec.ps1')
$registry = Get-Content (Join-Path $repo 'ops/llama-models.json') -Raw | ConvertFrom-Json
$spec = $registry.models[0] | ConvertTo-Json -Depth 8 | ConvertFrom-Json
$spec | Add-Member -Force -NotePropertyName speculative -NotePropertyValue (
    '{"type":"draft-mtp","max_tokens":10,"cache_type_k":"q8_0","cache_type_v":"q8_0"}' | ConvertFrom-Json)

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Assert-Argument($Arguments, [string]$Flag, [string]$Expected) {
    $index = [Array]::IndexOf($Arguments, $Flag)
    Assert-True ($index -ge 0 -and $Arguments[$index + 1] -eq $Expected) "Invalid argument: $Flag"
}
$arguments = @(Get-LlamaServerArguments -Spec $spec -Weights 'candidate.gguf')
Assert-Argument $arguments '--spec-type' 'draft-mtp'
Assert-Argument $arguments '--spec-draft-n-max' '10'
Assert-Argument $arguments '--spec-draft-type-k' 'q8_0'
Assert-Argument $arguments '--spec-draft-type-v' 'q8_0'
Assert-Argument $arguments '--cache-type-k' 'q4_0'
Assert-Argument $arguments '--ctx-size' '262144'
Assert-True ('--spec-draft-model' -notin $arguments) 'Integrated MTP must share target weights'
Assert-True ('--spec-type' -notin @(Get-LlamaServerArguments -Spec $registry.models[1] -Weights 'other.gguf')) 'Scalar policy changed'
# Selection must depend on artifact policy rather than model/family names.
$spec.model_id = 'arbitrary-artifact-id'
Assert-Argument @(Get-LlamaServerArguments -Spec $spec -Weights 'candidate.gguf') '--spec-type' 'draft-mtp'
foreach ($invalid in @(0, -1, 128, 1.5, $true, '10')) {
    $spec.speculative.max_tokens = $invalid
    $rejected = $false
    try { $null = Get-LlamaServerArguments -Spec $spec -Weights 'candidate.gguf' } catch { $rejected = $true }
    Assert-True $rejected 'Invalid draft depth was accepted'
}
$spec.speculative.max_tokens = 10
$spec.speculative | Add-Member -NotePropertyName unknown -NotePropertyValue 'typo'
$rejected = $false
try { $null = Get-LlamaServerArguments -Spec $spec -Weights 'candidate.gguf' } catch { $rejected = $true }
Assert-True $rejected 'Unknown draft policy field was accepted'
$spec.speculative.PSObject.Properties.Remove('unknown')

function Stop-ScheduledTask { throw 'prepare attempted to stop a task' }
function Start-ScheduledTask { throw 'prepare attempted to start a task' }
function Register-ScheduledTask { throw 'prepare attempted to register a task' }
function Get-CimInstance { throw 'prepare attempted to inspect an active process' }
function Get-NetTCPConnection { throw 'prepare attempted to inspect an active port' }
function Invoke-RestMethod { throw 'prepare attempted an API request' }

$scratch = Join-Path ([IO.Path]::GetTempPath()) ('llama-prepare-' + [Guid]::NewGuid().ToString('N'))
$previousRoot = $env:MODEL_ROOT
try {
    $windows = Join-Path $scratch 'ops/windows'
    $artifact = Join-Path $scratch ('models/' + $spec.artifact)
    New-Item -ItemType Directory -Path $windows, $artifact, (Join-Path $scratch 'artifacts') -Force | Out-Null
    foreach ($file in @('Common.ps1', 'LlamaSpec.ps1', 'Invoke-LlamaServer.ps1')) {
        Copy-Item (Join-Path $repo "ops/windows/$file") $windows
    }
    $spec.bytes = 1
    [IO.File]::WriteAllBytes((Join-Path $artifact $spec.file), [byte[]]@(0))
    @{schema = 'llama-models-v1'; models = @($spec)} | ConvertTo-Json -Depth 8 |
        Set-Content (Join-Path $scratch 'ops/llama-models.json') -Encoding UTF8
    $manifest = Join-Path $artifact 'manifest.json'
    @{schema = 'llama-artifact-v1'; model = $spec} | ConvertTo-Json -Depth 8 |
        Set-Content $manifest -Encoding UTF8
    @{schema = 'llama-completed-v1'; manifest_sha256 = (Get-FileHash $manifest).Hash.ToLowerInvariant()} |
        ConvertTo-Json | Set-Content (Join-Path $artifact 'COMPLETED') -Encoding UTF8
    $binary = Join-Path $scratch 'fake-server.exe'
    [IO.File]::WriteAllBytes($binary, [byte[]]@(0))
    $record = Join-Path $scratch 'artifacts/llama-server-process.json'
    '{"sentinel":"active process record must remain byte-identical"}' | Set-Content $record
    $recordHash = (Get-FileHash $record).Hash
    $launch = Join-Path $scratch 'artifacts/llama-server-launch.json'
    '{"sentinel":"old launch"}' | Set-Content $launch
    $env:MODEL_ROOT = Join-Path $scratch 'models'
    $null = & (Join-Path $windows 'Invoke-LlamaServer.ps1') -Action prepare -Model $spec.aliases[0] -Binary $binary
    Assert-True ($LASTEXITCODE -eq 0) 'Preparation failed'
    $prepared = Get-Content $launch -Raw | ConvertFrom-Json
    Assert-True ($prepared.schema -eq 'llama-launch-v1') 'Launch was not replaced'
    Assert-Argument $prepared.arguments '--spec-draft-n-max' '10'
    Assert-True ((Get-FileHash $record).Hash -eq $recordHash) 'Active process record changed'
    Assert-True (@(Get-ChildItem (Split-Path $launch) -Filter '*.tmp').Count -eq 0) 'Temporary launch file leaked'
} finally {
    $env:MODEL_ROOT = $previousRoot
    if (Test-Path $scratch) { Remove-Item $scratch -Recurse -Force }
}
@{status = 'passed'; gate = 'offline-mtp-arguments-and-no-lifecycle-prepare'} | ConvertTo-Json -Compress
