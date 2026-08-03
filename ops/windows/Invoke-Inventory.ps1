. (Join-Path $PSScriptRoot "Common.ps1")

Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor
$memoryModules = @(Get-CimInstance Win32_PhysicalMemory)
$physicalDisks = @(Get-PhysicalDisk)
$videoControllers = @(Get-CimInstance Win32_VideoController)

$hfModels = @()
if (Test-Path -LiteralPath $config.huggingface_hub) {
    $hfModels = @(Get-ChildItem -LiteralPath $config.huggingface_hub -Directory |
        Where-Object Name -Like "models--*" |
        ForEach-Object {
            $summaries = @()
            $snapshotRoot = Join-Path $_.FullName "snapshots"
            if (Test-Path -LiteralPath $snapshotRoot) {
                $summaries = @(Get-ChildItem -LiteralPath $snapshotRoot -Filter "config.json" -File -Recurse -ErrorAction SilentlyContinue |
                    ForEach-Object { Read-JsonSummary -Path $_.FullName })
            }
            [PSCustomObject]@{
                name = $_.Name
                path = $_.FullName
                size_bytes = Get-DirectorySizeBytes -Path $_.FullName
                configs = $summaries
            }
        })
}

$dockerStatus = [PSCustomObject]@{
    available = $false
    server_version = $null
    volume = $config.docker_volume
    volume_size_bytes = $null
    models = @()
    error = $null
}

try {
    $dockerStatus.server_version = (& docker version --format "{{.Server.Version}}" 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw $dockerStatus.server_version }
    $dockerStatus.available = $true

    $probeCommand = "du -sb /models/*"
    $probeArgs = @(
        "run", "--rm", "--network", "none", "--entrypoint", "/bin/sh",
        "-v", ($config.docker_volume + ":/models:ro"),
        $config.docker_probe_image, "-c", $probeCommand
    )
    $probeOutput = @(& docker @probeArgs 2>&1)
    if ($LASTEXITCODE -ne 0) { throw ($probeOutput -join [Environment]::NewLine) }

    $dockerModels = @()
    foreach ($line in $probeOutput) {
        if ($line -match '^(\d+)\s+/models/(.+)$') {
            $dockerModels += [PSCustomObject]@{
                name = $Matches[2]
                size_bytes = [int64]$Matches[1]
            }
        }
    }
    $dockerStatus.models = $dockerModels
    $dockerStatus.volume_size_bytes = [int64](($dockerModels | Measure-Object -Property size_bytes -Sum).Sum)
}
catch {
    $dockerStatus.error = $_.Exception.Message
}

$targetConfig = $null
if ($dockerStatus.available) {
    try {
        $targetPath = "/models/$($config.models.runtime_moe)/config.json"
        $targetArgs = @(
            "run", "--rm", "--network", "none", "--entrypoint", "/bin/cat",
            "-v", ($config.docker_volume + ":/models:ro"),
            $config.docker_probe_image, $targetPath
        )
        $rawConfig = (& docker @targetArgs 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0) { throw $rawConfig }
        $targetConfig = Convert-ModelConfigToSummary -Config ($rawConfig | ConvertFrom-Json) -Source ("docker://" + $config.docker_volume + $targetPath)
    }
    catch {
        $targetConfig = [PSCustomObject]@{ error = $_.Exception.Message }
    }
}

$inventory = [PSCustomObject]@{
    schema_version = 1
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    machine = [PSCustomObject]@{
        logical_name = $config.machine_name
        computer_name = $env:COMPUTERNAME
        os = $os.Caption
        os_version = $os.Version
        cpu = @($cpu | ForEach-Object {
            [PSCustomObject]@{
                name = $_.Name.Trim()
                cores = $_.NumberOfCores
                logical_processors = $_.NumberOfLogicalProcessors
                max_clock_mhz = $_.MaxClockSpeed
            }
        })
        total_memory_bytes = [int64]$os.TotalVisibleMemorySize * 1KB
        memory_modules = @($memoryModules | ForEach-Object {
            [PSCustomObject]@{
                manufacturer = $_.Manufacturer
                capacity_bytes = [int64]$_.Capacity
                configured_clock_mhz = $_.ConfiguredClockSpeed
            }
        })
        physical_disks = @($physicalDisks | ForEach-Object {
            [PSCustomObject]@{
                name = $_.FriendlyName
                media_type = [string]$_.MediaType
                bus_type = [string]$_.BusType
                size_bytes = [int64]$_.Size
                health_status = [string]$_.HealthStatus
            }
        })
        gpus_inventoried_not_used = @($videoControllers | ForEach-Object { $_.Name })
    }
    execution_policy = [PSCustomObject]@{
        cpu_only = $true
        cuda_visible_devices = $env:CUDA_VISIBLE_DEVICES
        nvidia_visible_devices = $env:NVIDIA_VISIBLE_DEVICES
        docker_gpu_flag_used = $false
    }
    models = [PSCustomObject]@{
        huggingface = $hfModels
        docker = $dockerStatus
        runtime_moe_config = $targetConfig
    }
}

$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$versionedPath = Write-JsonArtifact -Value $inventory -Name "inventory-$stamp.json"
$latestPath = Write-JsonArtifact -Value $inventory -Name "inventory-latest.json"

Write-Output "Inventory written: $versionedPath"
Write-Output "Latest inventory: $latestPath"
Write-Output "CPU-only policy: CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES; NVIDIA_VISIBLE_DEVICES=$env:NVIDIA_VISIBLE_DEVICES"
