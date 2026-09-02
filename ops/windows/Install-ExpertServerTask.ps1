param(
    [string]$TaskName = "",
    [string]$Container = "",
    [string]$Tokenizer = "",
    [string]$Runner = "",
    [string]$Python = "",
    [string]$HostAddress = "127.0.0.1",
    [string]$ApiKey = "",
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
    [ValidateSet("artifact", "fp8-e4m3-per-head", "fp16")]
    [string]$WorkerKvCacheDtype = "artifact",
    [switch]$ProfileGpuPhases,
    [switch]$DisableRetainedRoute,
    [switch]$EnableCpuHybrid,
    [string]$WorkerRouteTraceFile = "",
    [ValidateRange(1, 65536)][int]$WorkerRouteTraceMaxSteps = 4096,
    [double]$MicrobatchWindowMs = 2.0,
    [int]$LatencyWindow = 4096,
    [double]$QueueTimeoutSeconds = 1.0,
    [double]$GenerationTimeoutSeconds = 120.0,
    [int]$MaximumBodyMiB = 16,
    [ValidateRange(65536, 16777216)][int]$MaximumImagePixels = 2097152,
    [ValidateRange(256, 32768)][int]$MaximumImagePatchTokens = 4096,
    [int]$StartupTimeoutSeconds = 600,
    [int]$DrainTimeoutSeconds = 30,
    [string]$BuildId = "development",
    [string]$ModelId = "",
    [string]$RawResponseTraceFile = "",
    [switch]$Start
)

. (Join-Path $PSScriptRoot "Common.ps1")

if ($WorkerCapacity -lt 1 -or $StartupTimeoutSeconds -lt 1 -or
    $MaximumContext -lt 2 -or $MaximumNewTokens -lt 1 -or
    $WorkerKvCacheMiB -lt 1 -or $WorkerKvPageTokens -lt 1 -or
    $MaximumBodyMiB -lt 1 -or $MaximumImagePixels -lt 65536 -or
    $MaximumImagePatchTokens -lt 256) {
    throw "Invalid service limits"
}
if (-not $TaskName) { $TaskName = "QuantumLLM-ExpertVm" }
if (-not $Container -or -not $Runner) {
    throw "Artifact container and VM runner are required"
}
$startScript = Join-Path $PSScriptRoot "Start-ExpertServer.ps1"
foreach ($retiredTask in @(
    "QuantumLLM-DeepSeekV4Flash",
    "QuantumLLM-P6ExpertServer",
    "QuantumLLM-MoeVm"
)) {
    $retired = Get-ScheduledTask -TaskName $retiredTask -ErrorAction SilentlyContinue
    if ($null -ne $retired) {
        Stop-ScheduledTask -TaskName $retiredTask -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $retiredTask -Confirm:$false
    }
}
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
    "-WorkerKvCacheDtype", (Quote-TaskArgument $WorkerKvCacheDtype),
    "-MicrobatchWindowMs", $MicrobatchWindowMs.ToString([Globalization.CultureInfo]::InvariantCulture),
    "-LatencyWindow", [string]$LatencyWindow,
    "-QueueTimeoutSeconds", $QueueTimeoutSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
    "-GenerationTimeoutSeconds", $GenerationTimeoutSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
    "-MaximumBodyMiB", [string]$MaximumBodyMiB,
    "-MaximumImagePixels", [string]$MaximumImagePixels,
    "-MaximumImagePatchTokens", [string]$MaximumImagePatchTokens,
    "-StartupTimeoutSeconds", [string]$StartupTimeoutSeconds,
    "-DrainTimeoutSeconds", [string]$DrainTimeoutSeconds,
    "-BuildId", (Quote-TaskArgument $BuildId)
))
foreach ($entry in @(
    @{ Name = "Container"; Value = $Container },
    @{ Name = "Tokenizer"; Value = $Tokenizer },
    @{ Name = "Runner"; Value = $Runner },
    @{ Name = "Python"; Value = $Python },
    @{ Name = "ModelId"; Value = $ModelId },
    @{ Name = "ApiKey"; Value = $ApiKey },
    @{ Name = "WorkerRouteTraceFile"; Value = $WorkerRouteTraceFile },
    @{ Name = "RawResponseTraceFile"; Value = $RawResponseTraceFile }
)) {
    if ($entry.Value) {
        $taskArguments.Add("-$($entry.Name)")
        $taskArguments.Add((Quote-TaskArgument ([string]$entry.Value)))
    }
}
if ($ProfileGpuPhases) {
    $taskArguments.Add("-ProfileGpuPhases")
}
if ($DisableRetainedRoute) {
    $taskArguments.Add("-DisableRetainedRoute")
}
if ($EnableCpuHybrid) {
    $taskArguments.Add("-EnableCpuHybrid")
}
if ($WorkerRouteTraceFile) {
    $taskArguments.Add("-WorkerRouteTraceMaxSteps")
    $taskArguments.Add([string]$WorkerRouteTraceMaxSteps)
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
    contract = "artifact-vm"
    endpoint = "http://${HostAddress}:$Port"
    maximum_context = $MaximumContext
    maximum_new_tokens = $MaximumNewTokens
    maximum_body_mib = $MaximumBodyMiB
    maximum_image_pixels = $MaximumImagePixels
    maximum_image_patch_tokens = $MaximumImagePatchTokens
    model_input = $Container
    placement_profile = $PlacementProfile
    worker_kv_cache_dtype = $WorkerKvCacheDtype
    profile_gpu_phases = [bool]$ProfileGpuPhases
    retained_route_policy = if ($DisableRetainedRoute) {
        "disabled"
    } else {
        "provider"
    }
    cpu_hybrid_requested = [bool]$EnableCpuHybrid
    started = [bool]$Start
} | ConvertTo-Json
