param(
    [Parameter(Mandatory = $true)][ValidateSet('start', 'prepare', 'stop', 'status')][string]$Action,
    [string]$Model = '',
    [string]$HostAddress = '0.0.0.0',
    [int]$Port = 8081,
    [string]$ApiKey = '',
    [int]$WaitSeconds = 600,
    [int]$TimeoutSeconds = 14400,
    [string]$Binary = ''
)

. (Join-Path $PSScriptRoot 'Common.ps1')
. (Join-Path $PSScriptRoot 'LlamaSpec.ps1')
Initialize-ExperimentDirectories
if ($Port -lt 1 -or $Port -gt 65535 -or $WaitSeconds -lt 1 -or $TimeoutSeconds -lt 1) {
    throw 'Invalid llama server limits'
}
$taskName = 'QuantumLLM-LlamaCpp'
$processPath = Join-Path $script:RepoRoot 'artifacts/llama-server-process.json'
$launchPath = Join-Path $script:RepoRoot 'artifacts/llama-server-launch.json'
$logPath = Join-Path $script:RepoRoot 'logs/llama-server.stderr.log'
$registry = Get-Content (Join-Path $script:OpsRoot 'llama-models.json') -Raw | ConvertFrom-Json
if ($registry.schema -ne 'llama-models-v1') { throw 'Invalid llama model registry' }
$record = if (Test-Path $processPath) { Get-Content $processPath -Raw | ConvertFrom-Json } else { $null }

function Get-OwnedProcess {
    if ($null -eq $record) { return $null }
    if ($record.schema -ne 'llama-process-v1' -or $record.model -notin @($registry.models.model_id)) {
        throw 'Unknown llama process record; refusing process operation'
    }
    $running = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$record.pid)" -ErrorAction SilentlyContinue
    if ($null -ne $running -and ($running.ExecutablePath -ne $record.binary -or
        $running.CommandLine -notmatch [regex]::Escape([string]$record.model))) {
        throw 'Llama process identity changed; refusing process operation'
    }
    return $running
}

function Stop-OwnedServer {
    $owned = Get-OwnedProcess
    $tracked = if ($null -ne $owned) { Get-Process -Id $owned.ProcessId -ErrorAction SilentlyContinue } else { $null }
    if ($null -ne $tracked) { $null = $tracked.Handle }
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -ne $tracked) {
        if (-not $tracked.HasExited) { $tracked.Kill() }
        if (-not $tracked.WaitForExit(5000)) { throw 'Llama server process did not stop' }
    }
    # Only inspect a newly recorded child. The captured process is already
    # validated and awaited; its CIM fields are unreliable during termination.
    $after = if (Test-Path $processPath) { Get-Content $processPath -Raw | ConvertFrom-Json } else { $null }
    if ($null -ne $after -and ($null -eq $owned -or $after.pid -ne $owned.ProcessId)) {
        $script:record = $after
        $newChild = Get-OwnedProcess
        if ($null -ne $newChild) {
            $newTracked = Get-Process -Id $newChild.ProcessId -ErrorAction SilentlyContinue
            if ($null -ne $newTracked) {
                $null = $newTracked.Handle
                if (-not $newTracked.HasExited) { $newTracked.Kill() }
                if (-not $newTracked.WaitForExit(5000)) { throw 'Newly recorded llama child did not stop' }
            }
        }
    }
    Remove-Item $processPath -ErrorAction SilentlyContinue
    $script:record = $null
}

if ($Action -eq 'stop') {
    Stop-OwnedServer
    @{status = 'stopped'; task = $taskName} | ConvertTo-Json -Compress
    exit
}
$headers = if ($ApiKey) { @{Authorization = "Bearer $ApiKey"} } else { @{} }
function Get-ServerStatus {
    $health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -Headers $headers -TimeoutSec 5
    $models = Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" -Headers $headers -TimeoutSec 5
    $props = Invoke-RestMethod "http://127.0.0.1:$Port/props" -Headers $headers -TimeoutSec 5
    if ($health.status -ne 'ok') { throw 'Llama server is not ready' }
    return @{status = 'ready'; task = $taskName; model = $models.data[0].id;
        context = $props.default_generation_settings.n_ctx; slots = $props.total_slots;
        port = $Port; log = $logPath}
}
if ($Action -eq 'status') {
    $owned = Get-OwnedProcess
    if ($null -eq $owned) { @{status = 'stopped'; task = $taskName} | ConvertTo-Json -Compress; exit }
    Get-ServerStatus | ConvertTo-Json -Compress
    exit
}
if (-not $env:MODEL_ROOT) { throw 'MODEL_ROOT is required' }
$matches = @($registry.models | Where-Object { $Model -in $_.aliases })
if ($matches.Count -ne 1) { throw "Unknown or ambiguous llama model: $Model" }
$spec = $matches[0]
$slots = Get-LlamaParallelSlots -Spec $spec
$contextPerSlot = [long]$spec.context / $slots
$container = Join-Path $env:MODEL_ROOT $spec.artifact
$manifestPath = Join-Path $container 'manifest.json'
$completion = Get-Content (Join-Path $container 'COMPLETED') -Raw | ConvertFrom-Json
if ($completion.schema -ne 'llama-completed-v1' -or
    $completion.manifest_sha256 -ne (Get-FileHash $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()) {
    throw 'Llama artifact completion failed validation'
}
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
if ($manifest.schema -ne 'llama-artifact-v1' -or
    ($manifest.model | ConvertTo-Json -Depth 8 -Compress) -ne
    ($spec | ConvertTo-Json -Depth 8 -Compress)) { throw 'Llama artifact launch policy differs from registry' }
$weights = Join-Path $container $spec.file
if ((Get-Item $weights).Length -ne [int64]$spec.bytes) { throw 'GGUF size changed after publication' }
$projector = Get-LlamaProjector -Spec $spec
if ($null -ne $projector) {
    $projectorPath = Join-Path $container $projector.file
    if (-not (Test-Path -LiteralPath $projectorPath -PathType Leaf) -or
        (Get-Item -LiteralPath $projectorPath).Length -ne [int64]$projector.bytes -or
        (Get-FileHash -LiteralPath $projectorPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
        $projector.sha256) { throw 'Projector failed publication integrity validation' }
}
if (-not $Binary) {
    $installPath = Join-Path $script:RepoRoot 'artifacts/llama-install.json'
    if (-not (Test-Path -LiteralPath $installPath -PathType Leaf)) {
        throw 'llama.cpp is not installed; run ./ops/llama.sh update'
    }
    $install = Get-Content -LiteralPath $installPath -Raw | ConvertFrom-Json
    if ($install.schema -ne 'llama-install-v1') { throw 'Invalid llama.cpp installation record' }
    $Binary = [string]$install.binary
    if ((Get-FileHash -LiteralPath $Binary -Algorithm SHA256).Hash.ToLowerInvariant() -ne
        $install.binary_sha256) { throw 'Installed llama.cpp binary integrity mismatch' }
}
if (-not (Test-Path $Binary -PathType Leaf)) { throw "Missing llama.cpp CUDA binary: $Binary" }
$arguments = @(Get-LlamaServerArguments -Spec $spec -Weights $weights -HostAddress $HostAddress `
    -Port $Port -TimeoutSeconds $TimeoutSeconds)
# Preparation validates everything before touching the active service.
if ($Action -eq 'start') {
    Stop-OwnedServer
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port belongs to another service; it was not stopped"
    }
}
if ($ApiKey) {
    $keyPath = Join-Path $script:RepoRoot 'artifacts/llama-server-api-key.txt'
    [IO.File]::WriteAllText($keyPath, $ApiKey + "`n", [Text.UTF8Encoding]::new($false))
    $acl = New-Object Security.AccessControl.FileSecurity
    $acl.SetAccessRuleProtection($true, $false)
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($identity, 'FullControl', 'Allow'))
    Set-Acl -LiteralPath $keyPath -AclObject $acl
    $arguments += @('--api-key-file', $keyPath)
}
$temporaryLaunch = $launchPath + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
try {
    @{schema = 'llama-launch-v1'; binary = $Binary; arguments = $arguments;
        model = $spec.model_id; port = $Port; stderr = $logPath;
        stdout = (Join-Path $script:RepoRoot 'logs/llama-server.stdout.log')} |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporaryLaunch -Encoding UTF8
    if (Test-Path -LiteralPath $launchPath) {
        [IO.File]::Replace($temporaryLaunch, $launchPath, [NullString]::Value)
    } else {
        [IO.File]::Move($temporaryLaunch, $launchPath)
    }
} finally {
    if (Test-Path -LiteralPath $temporaryLaunch) { Remove-Item -LiteralPath $temporaryLaunch }
}
if ($Action -eq 'prepare') {
    @{status = 'prepared'; model = $spec.model_id; launch = $launchPath; restart_required = $true;
        slots = $slots; context_total = $spec.context; context_per_slot = $contextPerSlot} |
        ConvertTo-Json -Compress
    exit
}
Remove-Item $processPath -ErrorAction SilentlyContinue
$worker = Join-Path $PSScriptRoot 'Start-LlamaServer.ps1'
$taskAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument (
    '-NoProfile -ExecutionPolicy Bypass -File "' + $worker + '" -LaunchFile "' + $launchPath + '"')
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $taskAction -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
$deadline = [DateTime]::UtcNow.AddSeconds($WaitSeconds)
try {
    while ([DateTime]::UtcNow -lt $deadline) {
        $ready = $null
        try { $ready = Get-ServerStatus } catch { }
        if ($null -ne $ready) {
            if ($ready.model -ne $spec.model_id -or $ready.context -ne $contextPerSlot -or $ready.slots -ne $slots) {
                throw 'Llama server advertised a different model/context/slot contract'
            }
            $ready | ConvertTo-Json -Compress
            exit
        }
        $task = Get-ScheduledTask -TaskName $taskName
        if ($task.State -ne 'Running') { throw "Llama server exited; inspect $logPath" }
        Start-Sleep -Seconds 1
    }
    throw "Llama readiness failed; inspect $logPath"
} catch {
    $script:record = if (Test-Path $processPath) { Get-Content $processPath -Raw | ConvertFrom-Json } else { $null }
    Stop-OwnedServer
    throw
}
