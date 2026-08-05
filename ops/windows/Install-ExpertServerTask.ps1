param(
    [ValidateSet("P6", "DeepSeekV4Flash")][string]$Profile = "P6",
    [string]$TaskName = "",
    [string]$Container = "",
    [string]$Bundle = "",
    [string]$Tokenizer = "",
    [string]$Runner = "",
    [string]$Python = "",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [int]$MaximumQueue = 8,
    [int]$MaximumContext = 4096,
    [int]$MaximumNewTokens = 512,
    [int]$WorkerCapacity = 4,
    [int]$WorkerRamCacheGiB = 48,
    [int]$WorkerVramCacheGiB = 18,
    [ValidateSet("latency", "balanced", "capacity")]
    [string]$PlacementProfile = "balanced",
    [int]$WorkerKvCacheMiB = 2048,
    [int]$WorkerKvPageTokens = 256,
    [double]$MicrobatchWindowMs = 2.0,
    [int]$LatencyWindow = 4096,
    [double]$QueueTimeoutSeconds = 1.0,
    [double]$GenerationTimeoutSeconds = 120.0,
    [int]$StartupTimeoutSeconds = 600,
    [int]$DrainTimeoutSeconds = 30,
    [string]$BuildId = "development",
    [switch]$Start
)

. (Join-Path $PSScriptRoot "Common.ps1")

if ($WorkerCapacity -lt 1 -or $StartupTimeoutSeconds -lt 1 -or
    $MaximumContext -lt 2 -or $MaximumNewTokens -lt 1 -or
    $WorkerKvCacheMiB -lt 1 -or $WorkerKvPageTokens -lt 1) {
    throw "Invalid service limits"
}
if (-not $TaskName) {
    $TaskName = if ($Profile -eq "P6") {
        "QuantumLLM-P6ExpertServer"
    } else {
        "QuantumLLM-DeepSeekV4Flash"
    }
}
if (($Profile -eq "P6" -and $Bundle) -or
    ($Profile -eq "DeepSeekV4Flash" -and (-not $Bundle -or $Container))) {
    throw "Profile model input is invalid"
}
$startScript = Join-Path $PSScriptRoot $(if ($Profile -eq "P6") {
    "Start-P6ExpertServer.ps1"
} else {
    "Start-DeepSeekExpertServer.ps1"
})
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue }
[void](Stop-ExpertServerProcessTree -Port $Port)

function Quote-TaskArgument {
    param([string]$Value)
    return '"' + ($Value -replace '"', '""') + '"'
}

$taskArguments = [Collections.Generic.List[string]]::new()
$taskArguments.AddRange([string[]]@(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
    (Quote-TaskArgument $startScript),
    "-HostAddress", (Quote-TaskArgument $HostAddress),
    "-Port", [string]$Port,
    "-MaximumQueue", [string]$MaximumQueue,
    "-MaximumContext", [string]$MaximumContext,
    "-MaximumNewTokens", [string]$MaximumNewTokens,
    "-WorkerCapacity", [string]$WorkerCapacity,
    "-WorkerRamCacheGiB", [string]$WorkerRamCacheGiB,
    "-WorkerVramCacheGiB", [string]$WorkerVramCacheGiB,
    "-PlacementProfile", (Quote-TaskArgument $PlacementProfile),
    "-WorkerKvCacheMiB", [string]$WorkerKvCacheMiB,
    "-WorkerKvPageTokens", [string]$WorkerKvPageTokens,
    "-MicrobatchWindowMs", $MicrobatchWindowMs.ToString([Globalization.CultureInfo]::InvariantCulture),
    "-LatencyWindow", [string]$LatencyWindow,
    "-QueueTimeoutSeconds", $QueueTimeoutSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
    "-GenerationTimeoutSeconds", $GenerationTimeoutSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
    "-StartupTimeoutSeconds", [string]$StartupTimeoutSeconds,
    "-DrainTimeoutSeconds", [string]$DrainTimeoutSeconds,
    "-BuildId", (Quote-TaskArgument $BuildId)
))
foreach ($entry in @(
    @{ Name = if ($Profile -eq "P6") { "Container" } else { "Bundle" };
       Value = if ($Profile -eq "P6") { $Container } else { $Bundle } },
    @{ Name = "Tokenizer"; Value = $Tokenizer },
    @{ Name = "Runner"; Value = $Runner },
    @{ Name = "Python"; Value = $Python }
)) {
    if ($entry.Value) {
        $taskArguments.Add("-$($entry.Name)")
        $taskArguments.Add((Quote-TaskArgument ([string]$entry.Value)))
    }
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument ($taskArguments -join " ")
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $identity `
    -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Out-Null
if ($Start) { Start-ScheduledTask -TaskName $TaskName }

[PSCustomObject]@{
    status = "installed"
    task = $TaskName
    profile = $Profile
    endpoint = "http://${HostAddress}:$Port"
    maximum_context = $MaximumContext
    maximum_new_tokens = $MaximumNewTokens
    model_input = if ($Profile -eq "P6") { $Container } else { $Bundle }
    placement_profile = $PlacementProfile
    started = [bool]$Start
} | ConvertTo-Json
