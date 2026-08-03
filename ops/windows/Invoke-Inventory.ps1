param(
    [switch]$CpuOnly,
    [int64]$SustainedStorageReadBytesPerSecond = 0,
    [int64]$SustainedH2DBytesPerSecond = 0,
    [switch]$StorageBandwidthMeasured,
    [switch]$H2DBandwidthMeasured
)

. (Join-Path $PSScriptRoot "Common.ps1")

if ($CpuOnly) { Set-CpuOnlyEnvironment }
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor
$memoryModules = @(Get-CimInstance Win32_PhysicalMemory)
$physicalDisks = @(Get-PhysicalDisk)
$videoControllers = @(Get-CimInstance Win32_VideoController)
$logicalDisks = @(Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3")

$cuda = [ordered]@{
    nvidia_smi_available = $false
    driver_version = $null
    nvcc_available = $false
    toolkit_version = $null
    devices = @()
    error = $null
}

try {
    $nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction Stop
    $query = "index,name,driver_version,memory.total,memory.free,compute_cap,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
    $rows = @(& $nvidiaSmi.Source "--query-gpu=$query" "--format=csv,noheader,nounits" 2>&1)
    if ($LASTEXITCODE -ne 0) { throw ($rows -join [Environment]::NewLine) }
    $devices = @()
    foreach ($row in $rows) {
        $fields = @($row -split "," | ForEach-Object { $_.Trim() })
        if ($fields.Count -ne 11) { throw "Unexpected nvidia-smi row: $row" }
        $devices += [PSCustomObject]@{
            index = [int]$fields[0]
            name = $fields[1]
            vram_total_bytes = [int64]$fields[3] * 1MB
            vram_free_bytes = [int64]$fields[4] * 1MB
            compute_capability = $fields[5]
            pci_bus_id = $fields[6]
            pcie = [PSCustomObject]@{
                current_generation = [int]$fields[7]
                maximum_generation = [int]$fields[8]
                current_width = [int]$fields[9]
                maximum_width = [int]$fields[10]
            }
        }
    }
    $cuda.nvidia_smi_available = $true
    $cuda.devices = $devices
    if ($devices.Count -gt 0) { $cuda.driver_version = (@($rows[0] -split ","))[2].Trim() }
}
catch {
    $cuda.error = $_.Exception.Message
}

try {
    $nvcc = Get-Command "nvcc.exe" -ErrorAction Stop
    $nvccOutput = (& $nvcc.Source --version 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { throw $nvccOutput }
    $cuda.nvcc_available = $true
    if ($nvccOutput -match "release\s+([0-9]+\.[0-9]+)") {
        $cuda.toolkit_version = $Matches[1]
    }
}
catch {
    if ($null -eq $cuda.error) { $cuda.error = $_.Exception.Message }
}

$cmake = [ordered]@{ available = $false; version = $null; path = $null }
try {
    $cmakeCommand = Get-Command "cmake.exe" -ErrorAction Stop
    $cmakeOutput = (& $cmakeCommand.Source --version 2>&1 | Select-Object -First 1)
    $cmake.available = $true
    $cmake.path = $cmakeCommand.Source
    if ($cmakeOutput -match "cmake version\s+([^\s]+)") { $cmake.version = $Matches[1] }
}
catch {}

$primaryGpu = @($cuda.devices | Select-Object -First 1)
$plannerVramTotal = if ($primaryGpu.Count) { [int64]$primaryGpu[0].vram_total_bytes } else { [int64]0 }
$plannerVramAvailable = if ($primaryGpu.Count) { [int64]$primaryGpu[0].vram_free_bytes } else { [int64]0 }
$systemDrive = @($logicalDisks | Where-Object DeviceID -EQ $env:SystemDrive | Select-Object -First 1)
$plannerDiskFree = if ($systemDrive.Count) { [int64]$systemDrive[0].FreeSpace } else { [int64]0 }

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
    schema_version = 2
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
        available_memory_bytes = [int64]$os.FreePhysicalMemory * 1KB
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
                logical_sector_bytes = [int64]$_.LogicalSectorSize
                physical_sector_bytes = [int64]$_.PhysicalSectorSize
            }
        })
        volumes = @($logicalDisks | ForEach-Object {
            [PSCustomObject]@{
                drive = $_.DeviceID
                filesystem = $_.FileSystem
                size_bytes = [int64]$_.Size
                free_bytes = [int64]$_.FreeSpace
            }
        })
        display_adapters = @($videoControllers | ForEach-Object { $_.Name })
    }
    execution_policy = [PSCustomObject]@{
        cpu_only = [bool]$CpuOnly
        target = if ($CpuOnly) { "cpu-experiment" } else { "windows-cuda" }
        cuda_visible_devices = $env:CUDA_VISIBLE_DEVICES
        nvidia_visible_devices = $env:NVIDIA_VISIBLE_DEVICES
        docker_gpu_flag_used = $false
    }
    cuda = [PSCustomObject]$cuda
    toolchain = [PSCustomObject]@{
        cmake = [PSCustomObject]$cmake
        expected_generator = "Visual Studio 17 2022"
    }
    planner = [PSCustomObject]@{
        total_ram_bytes = [int64]$os.TotalVisibleMemorySize * 1KB
        available_ram_bytes = [int64]$os.FreePhysicalMemory * 1KB
        vram_total_bytes = $plannerVramTotal
        vram_available_bytes = $plannerVramAvailable
        disk_free_bytes = $plannerDiskFree
        sustained_storage_read_bytes_per_second = [int64]$SustainedStorageReadBytesPerSecond
        sustained_h2d_bytes_per_second = [int64]$SustainedH2DBytesPerSecond
        storage_bandwidth_measured = [bool]$StorageBandwidthMeasured
        h2d_bandwidth_measured = [bool]$H2DBandwidthMeasured
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
Write-Output "Execution target: $($inventory.execution_policy.target)"
Write-Output "CUDA devices: $(@($cuda.devices).Count); nvcc=$($cuda.toolkit_version); driver=$($cuda.driver_version)"
if ($SustainedStorageReadBytesPerSecond -eq 0 -or $SustainedH2DBytesPerSecond -eq 0) {
    Write-Warning "Planner bandwidth is zero until qualified values are supplied; SLO admission will fail closed."
}
