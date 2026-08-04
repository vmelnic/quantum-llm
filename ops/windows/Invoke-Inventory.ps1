param(
    [string]$HuggingFaceHub = (Join-Path $env:USERPROFILE ".cache\huggingface\hub"),
    [int64]$SustainedStorageReadBytesPerSecond = 0,
    [int64]$SustainedH2DBytesPerSecond = 0,
    [switch]$StorageBandwidthMeasured,
    [switch]$H2DBandwidthMeasured
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$os = Get-CimInstance Win32_OperatingSystem
$cpu = @(Get-CimInstance Win32_Processor)
$memoryModules = @(Get-CimInstance Win32_PhysicalMemory)
$physicalDisks = @(Get-PhysicalDisk)
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
    $query = "index,name,driver_version,memory.total,memory.free,compute_cap,pci.bus_id"
    $rows = @(& $nvidiaSmi.Source "--query-gpu=$query" "--format=csv,noheader,nounits" 2>&1)
    if ($LASTEXITCODE -ne 0) { throw ($rows -join [Environment]::NewLine) }
    $devices = @()
    foreach ($row in $rows) {
        $fields = @($row -split "," | ForEach-Object { $_.Trim() })
        if ($fields.Count -ne 7) { throw "Unexpected nvidia-smi row: $row" }
        $devices += [PSCustomObject]@{
            index = [int]$fields[0]
            name = $fields[1]
            vram_total_bytes = [int64]$fields[3] * 1MB
            vram_free_bytes = [int64]$fields[4] * 1MB
            compute_capability = $fields[5]
            pci_bus_id = $fields[6]
        }
    }
    $cuda.nvidia_smi_available = $true
    $cuda.devices = $devices
    if ($rows.Count -gt 0) { $cuda.driver_version = (@($rows[0] -split ","))[2].Trim() }
}
catch { $cuda.error = $_.Exception.Message }

try {
    $nvcc = Get-Command "nvcc.exe" -ErrorAction Stop
    $nvccOutput = (& $nvcc.Source --version 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { throw $nvccOutput }
    $cuda.nvcc_available = $true
    if ($nvccOutput -match "release\s+([0-9]+\.[0-9]+)") {
        $cuda.toolkit_version = $Matches[1]
    }
}
catch { if ($null -eq $cuda.error) { $cuda.error = $_.Exception.Message } }

$cmake = [ordered]@{ available = $false; version = $null; path = $null }
try {
    $command = Get-Command "cmake.exe" -ErrorAction Stop
    $line = (& $command.Source --version 2>&1 | Select-Object -First 1)
    $cmake.available = $true
    $cmake.path = $command.Source
    if ($line -match "cmake version\s+([^\s]+)") { $cmake.version = $Matches[1] }
}
catch {}

$primaryGpu = @($cuda.devices | Select-Object -First 1)
$systemDrive = @($logicalDisks | Where-Object DeviceID -EQ $env:SystemDrive |
    Select-Object -First 1)
$hfModels = @()
if (Test-Path -LiteralPath $HuggingFaceHub -PathType Container) {
    $hfModels = @(Get-ChildItem -LiteralPath $HuggingFaceHub -Directory |
        Where-Object Name -Like "models--*" | ForEach-Object {
            [PSCustomObject]@{
                name = $_.Name
                path = $_.FullName
                size_bytes = Get-DirectorySizeBytes -Path $_.FullName
            }
        })
}

$inventory = [PSCustomObject]@{
    schema_version = 3
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    machine = [PSCustomObject]@{
        computer_name = $env:COMPUTERNAME
        os = $os.Caption
        os_version = $os.Version
        cpu = @($cpu | ForEach-Object { [PSCustomObject]@{
            name = $_.Name.Trim()
            cores = $_.NumberOfCores
            logical_processors = $_.NumberOfLogicalProcessors
        } })
        total_memory_bytes = [int64]$os.TotalVisibleMemorySize * 1KB
        available_memory_bytes = [int64]$os.FreePhysicalMemory * 1KB
        memory_modules = @($memoryModules | ForEach-Object { [PSCustomObject]@{
            manufacturer = $_.Manufacturer
            capacity_bytes = [int64]$_.Capacity
            configured_clock_mhz = $_.ConfiguredClockSpeed
        } })
        physical_disks = @($physicalDisks | ForEach-Object { [PSCustomObject]@{
            name = $_.FriendlyName
            media_type = [string]$_.MediaType
            bus_type = [string]$_.BusType
            size_bytes = [int64]$_.Size
            health_status = [string]$_.HealthStatus
        } })
        volumes = @($logicalDisks | ForEach-Object { [PSCustomObject]@{
            drive = $_.DeviceID
            filesystem = $_.FileSystem
            size_bytes = [int64]$_.Size
            free_bytes = [int64]$_.FreeSpace
        } })
    }
    cuda = [PSCustomObject]$cuda
    toolchain = [PSCustomObject]@{
        cmake = [PSCustomObject]$cmake
        expected_generator = "Visual Studio 17 2022"
    }
    planner = [PSCustomObject]@{
        total_ram_bytes = [int64]$os.TotalVisibleMemorySize * 1KB
        available_ram_bytes = [int64]$os.FreePhysicalMemory * 1KB
        vram_total_bytes = if ($primaryGpu.Count) { [int64]$primaryGpu[0].vram_total_bytes } else { [int64]0 }
        vram_available_bytes = if ($primaryGpu.Count) { [int64]$primaryGpu[0].vram_free_bytes } else { [int64]0 }
        disk_free_bytes = if ($systemDrive.Count) { [int64]$systemDrive[0].FreeSpace } else { [int64]0 }
        sustained_storage_read_bytes_per_second = $SustainedStorageReadBytesPerSecond
        sustained_h2d_bytes_per_second = $SustainedH2DBytesPerSecond
        storage_bandwidth_measured = [bool]$StorageBandwidthMeasured
        h2d_bandwidth_measured = [bool]$H2DBandwidthMeasured
    }
    models = [PSCustomObject]@{ huggingface = $hfModels }
}

$path = Write-JsonArtifact -Value $inventory -Name "inventory-latest.json"
$inventory | ConvertTo-Json -Depth 8
Write-Output "Inventory written: $path"
