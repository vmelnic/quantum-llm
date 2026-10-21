param(
    [ValidateSet('install','publish','deploy','start','restart','stop','status','config')]
    [string]$Action = 'status',
    [string]$Model = 'swift',
    [ValidateRange(1,65535)][int]$Port = 8082,
    [string]$ApiKeyBase64 = ''
)

. (Join-Path $PSScriptRoot 'Common.ps1')
$ProgressPreference = 'SilentlyContinue'
Initialize-ExperimentDirectories
if (-not $env:MODEL_ROOT) { throw 'MODEL_ROOT is required' }
$runtimeHome = Join-Path $script:RepoRoot 'work/strata'
$terminal = Join-Path $runtimeHome 'terminal/current'
$cli = Join-Path $terminal 'bin/strata.cmd'
$python = Join-Path $terminal 'python/python.exe'
$env:STRATA_DATA_DIR = $runtimeHome
$env:STRATA_COMPONENTS_DIR = Join-Path $runtimeHome 'components'
$env:PYTHONUTF8 = '1'
$installPath = Join-Path $script:RepoRoot 'artifacts/strata-install.json'
$registry = Get-Content (Join-Path $script:OpsRoot 'strata-models.json') -Raw | ConvertFrom-Json
if ($registry.schema -ne 'strata-models-v1') { throw 'Invalid Strata model registry' }
$matchingModels = @($registry.models | Where-Object { $Model -in $_.aliases })
if ($matchingModels.Count -ne 1) { throw "Unknown Strata model: $Model" }
$selected = $matchingModels[0]
$data = Join-Path $env:MODEL_ROOT 'strata'
$modelDirectory = Join-Path $data ([string]$selected.directory)
$runDirectory = Join-Path $runtimeHome 'run'
$configPath = Join-Path $runDirectory ($selected.model_id + '.json')
$instanceProfile = 'quantum-' + $selected.model_id
$taskName = 'QuantumLLM-Strata'

function Invoke-StrataCli {
    param([string[]]$Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $lines = & $cli @Arguments; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $previous }
    if ($code -ne 0) { throw "Strata CLI failed (exit $code)" }
    return $lines
}

function Write-AtomicJson {
    param([string]$Path, $Value)
    New-Item -ItemType Directory -Path (Split-Path $Path -Parent) -Force | Out-Null
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath ($Path + '.tmp') -Encoding UTF8
    Move-Item -LiteralPath ($Path + '.tmp') -Destination $Path -Force
}

if ($Action -eq 'install') {
    $release = [string]$registry.release
    if (-not (Test-Path -LiteralPath $cli -PathType Leaf)) {
        $installer = Join-Path $script:RepoRoot 'out/strata-install.ps1'
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/Niko1221/Strata/releases/download/$release/install.ps1" -OutFile $installer
        $expected = '89f26d8884be1e05f026faec2e0bc80e8e370d9e51a16a1ec1fe92de17273c2c'
        if ((Get-FileHash $installer -Algorithm SHA256).Hash.ToLower() -ne $expected) {
            throw 'The upstream installer differs from the reviewed release script'
        }
        $env:STRATA_FEED = "https://github.com/Niko1221/Strata/releases/download/$release"
        $env:STRATA_NO_PATH = '1'
        & $installer
    }
    $result = (Invoke-StrataCli -Arguments @('update','--no-control','--feed',
        "https://github.com/Niko1221/Strata/releases/download/$release",'--output','json')) -join "`n" | ConvertFrom-Json
    $engineResult = @($result.results | Where-Object { $_.component -eq 'engine' })[0]
    $installed = if ($engineResult.PSObject.Properties['installed_after']) {
        $engineResult.installed_after
    } elseif ($engineResult.status -eq 'current') {
        $engineResult.installed_before
    } else { $null }
    if ($null -eq $installed -or -not $installed.signed) { throw 'No verified Strata engine installed' }
    $binary = Join-Path (Join-Path $env:STRATA_COMPONENTS_DIR 'engine') ($installed.folder + '/strata.exe')
    if (-not (Test-Path $binary -PathType Leaf)) { throw 'Installed engine executable is absent' }
    $record = [ordered]@{schema='strata-install-v1';release=$release;version=$installed.version;
        binary=$binary;sha256=(Get-FileHash $binary -Algorithm SHA256).Hash.ToLower();signed=$true;
        installed_utc=[DateTime]::UtcNow.ToString('o')}
    Write-AtomicJson -Path $installPath -Value $record
    $record | ConvertTo-Json
    exit
}
if (-not (Test-Path $cli -PathType Leaf) -or -not (Test-Path $installPath -PathType Leaf)) {
    throw 'Install the separate runtime first: ./ops/strata.sh install'
}
$installation = Get-Content $installPath -Raw | ConvertFrom-Json
if ($installation.schema -ne 'strata-install-v1' -or
    (Get-FileHash $installation.binary -Algorithm SHA256).Hash.ToLower() -ne $installation.sha256) {
    throw 'The Strata engine does not match the installed executable record'
}

function Test-SourceFile {
    param([string]$Path, $Metadata)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or
        (Get-Item -LiteralPath $Path).Length -ne [int64]$Metadata.size) {
        throw "Missing or incomplete Xet download: $($Metadata.path)"
    }
    if ($Metadata.PSObject.Properties['sha256']) {
        if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower() -ne $Metadata.sha256) {
            throw "Source SHA256 mismatch: $($Metadata.path)"
        }
    } else {
        # Small Hub text assets are Git blobs rather than LFS objects.
        $bytes = [IO.File]::ReadAllBytes($Path)
        $header = [Text.Encoding]::ASCII.GetBytes("blob $($bytes.Length)`0")
        $sha = [Security.Cryptography.SHA1]::Create()
        try { $digest = $sha.ComputeHash([byte[]]($header + $bytes)) }
        finally { $sha.Dispose() }
        $actual = ([BitConverter]::ToString($digest)).Replace('-','').ToLower()
        if ($actual -ne $Metadata.git_sha1) { throw "Source Git blob mismatch: $($Metadata.path)" }
    }
}

function Publish-Assets {
    $marker = Join-Path $modelDirectory 'COMPLETED'
    if (Test-Path -LiteralPath $marker -PathType Leaf) {
        $published = Get-Content $marker -Raw | ConvertFrom-Json
        if ($published.model_id -eq $selected.model_id -and
            $published.target_revision -eq $selected.sources[0].revision) {
            foreach ($source in $selected.sources) {
                foreach ($file in $source.files) {
                    $destination = if ($source.role -eq 'target') { Join-Path $modelDirectory $file.path }
                        else { Join-Path (Join-Path $data 'mtp/rt') (Split-Path $file.path -Leaf) }
                    if (-not (Test-Path $destination -PathType Leaf) -or
                        (Get-Item $destination).Length -ne [int64]$file.size) { throw 'Published asset is missing or truncated' }
                }
            }
            return
        }
        throw 'Existing model publication differs; preserve it and review the candidate'
    }
    $candidate = $modelDirectory + '.candidate'
    New-Item -ItemType Directory -Path $candidate -Force | Out-Null
    foreach ($source in $selected.sources) {
        $cache = 'models--' + ([string]$source.repository).Replace('/','--')
        $snapshot = Join-Path (Join-Path (Join-Path $env:MODEL_ROOT '.hf-cache/hub') $cache) ('snapshots/' + $source.revision)
        foreach ($file in $source.files) {
            $path = Join-Path $snapshot ([string]$file.path)
            Test-SourceFile -Path $path -Metadata $file
            $destination = if ($source.role -eq 'target') { Join-Path $candidate $file.path }
                else { Join-Path (Join-Path $data 'mtp/rt') (Split-Path $file.path -Leaf) }
            New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
            if (-not (Test-Path -LiteralPath $destination)) {
                try { New-Item -ItemType HardLink -Path $destination -Target $path -ErrorAction Stop | Out-Null }
                catch { Copy-Item -LiteralPath $path -Destination $destination }
            } else { Test-SourceFile -Path $destination -Metadata $file }
            if ($source.role -eq 'target') {
                $stream = [IO.File]::OpenRead($destination)
                try { $header=New-Object byte[] 8; [void]$stream.Read($header,0,8) } finally { $stream.Dispose() }
                if ([Text.Encoding]::ASCII.GetString($header,0,4) -ne 'GGUF' -or
                    [BitConverter]::ToUInt32($header,4) -ne 3) { throw 'Expected a complete GGUF v3 shard' }
            }
        }
    }
    Write-AtomicJson -Path (Join-Path $candidate 'COMPLETED') -Value @{
        schema='strata-assets-v1';model_id=$selected.model_id;target_revision=$selected.sources[0].revision;
        sources=$selected.sources;validated_utc=[DateTime]::UtcNow.ToString('o')}
    if (Test-Path -LiteralPath $modelDirectory) { throw 'Refusing to replace an existing incomplete publication' }
    Move-Item -LiteralPath $candidate -Destination $modelDirectory
}

function Prepare-Service {
    Publish-Assets
    if (-not $ApiKeyBase64) { throw 'EXPERT_API_KEY is required for the remote API' }
    $apiKey = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($ApiKeyBase64))
    $payload = @{api_key=$apiKey} | ConvertTo-Json -Compress
    $prepare = Join-Path $script:OpsRoot 'python/prepare_strata.py'
    $env:STRATA_NO_ACQUIRE = '1'
    $env:HF_HUB_OFFLINE = '1'
    $payload | & $python $prepare --terminal $terminal --data $data --model-dir $modelDirectory `
        --engine $installation.binary --config $configPath --model-id $selected.model_id `
        --context $selected.context_size --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Offline Strata preparation failed' }
    $previous = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $null = & $cli profile show $instanceProfile --output json 2>$null; $exists = $LASTEXITCODE -eq 0 }
    finally { $ErrorActionPreference = $previous }
    if (-not $exists) {
        $null = Invoke-StrataCli -Arguments @('profile','create',$instanceProfile,'--from-config',$configPath,'--output','json')
    }
}

function Stop-Service {
    Invoke-StrataCli -Arguments @('runtime','stop','--profile',$instanceProfile,'--output','json')
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -ne $task -and $task.State -eq 'Running') {
        Stop-ScheduledTask -TaskName $taskName
    }
}

function Start-Service {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -ne $task -and $task.State -eq 'Running') {
        Invoke-StrataCli -Arguments @('runtime','status','--profile',$instanceProfile,'--output','json')
        return
    }
    # The API is reachable only from the VPN subnet, using the existing API key.
    $addresses = @(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like '10.10.88.*' })
    if ($addresses.Count -ne 1) { throw 'Expected one WireGuard IPv4 address on 10.10.88.0/24' }
    $ruleName = 'QuantumLLM-Strata-WireGuard'
    $rule = Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
    $program = (& $python -c 'import os,sys; print(os.path.realpath(sys.executable))').Trim()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $program -PathType Leaf)) {
        throw 'Unable to resolve the installed Strata Python executable'
    }
    # Windows' first-run prompt can create an explicit block for the versioned
    # executable. Such a block overrides even the VPN-only allow rule below.
    Get-NetFirewallRule -Direction Inbound -Enabled True -Action Block | ForEach-Object {
        $blocked = $_
        $application = $blocked | Get-NetFirewallApplicationFilter
        $ports = $blocked | Get-NetFirewallPortFilter
        if ($application.Program -eq $program -and $ports.Protocol -in @('TCP','6')) {
            $blocked | Disable-NetFirewallRule | Out-Null
        }
    }
    if ($null -eq $rule) {
        New-NetFirewallRule -Name $ruleName -DisplayName 'QuantumLLM Strata API over WireGuard' `
            -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port `
            -LocalAddress $addresses[0].IPAddress -RemoteAddress '10.10.88.0/24' `
            -Profile Any -Program $program | Out-Null
    } else {
        $rule | Set-NetFirewallRule -Enabled True -Action Allow -Protocol TCP -LocalPort $Port `
            -LocalAddress $addresses[0].IPAddress -RemoteAddress '10.10.88.0/24' -Program $program | Out-Null
    }
    $worker = Join-Path $PSScriptRoot 'Start-StrataServer.ps1'
    $startupLog = Join-Path $runDirectory ($selected.model_id + '.startup.json')
    Remove-Item -LiteralPath $startupLog -ErrorAction SilentlyContinue
    $arguments = '-NoProfile -ExecutionPolicy Bypass -File "' + $worker +
        '" -RuntimeHome "' + $runtimeHome + '" -ProfileName "' + $instanceProfile +
        '" -StartupLog "' + $startupLog + '"'
    $taskAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments
    $principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
        -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $taskAction -Principal $principal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $taskName
    $deadline = [DateTime]::UtcNow.AddSeconds(630)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-Path -LiteralPath $startupLog) {
            $result = $null
            try { $result = Get-Content -LiteralPath $startupLog -Raw | ConvertFrom-Json } catch { }
            if ($null -ne $result) {
                if (-not $result.ok -or $result.state -ne 'ready') { throw "Strata startup failed; inspect $startupLog" }
                Invoke-StrataCli -Arguments @('runtime','status','--profile',$instanceProfile,'--output','json')
                return
            }
        }
        if ((Get-ScheduledTask -TaskName $taskName).State -ne 'Running') {
            throw "Strata task exited; inspect $startupLog"
        }
        Start-Sleep -Seconds 1
    }
    throw "Strata readiness timed out; inspect $startupLog"
}

switch ($Action) {
    'publish' { Publish-Assets; @{status='published';model=$selected.model_id;path=$modelDirectory} | ConvertTo-Json }
    'deploy' { Prepare-Service }
    'config' {
        if (-not (Test-Path $configPath)) { throw 'No prepared Strata configuration' }
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        $configured = [bool]$config.api_key
        $config.PSObject.Properties.Remove('api_key')
        $config | Add-Member -NotePropertyName api_key_configured -NotePropertyValue $configured
        $config | ConvertTo-Json -Depth 8
    }
    'status' { Invoke-StrataCli -Arguments @('runtime','status','--profile',$instanceProfile,'--output','json') }
    'stop' { Stop-Service }
    { $_ -in @('start','restart') } {
        if ($Action -eq 'restart') { $null = Stop-Service }
        if (-not (Test-Path $configPath)) { Prepare-Service }
        $competing = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -in @('llama-server.exe','expert-moe-vm-runner.exe') })
        if ($competing.Count) { throw 'Another GPU model is running; stop it through its own ops script first' }
        Start-Service
    }
}
