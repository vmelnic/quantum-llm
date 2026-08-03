. (Join-Path $PSScriptRoot "Common.ps1")

Set-CpuOnlyEnvironment
Initialize-ExperimentDirectories
$config = Get-ExperimentConfig

$state = [PSCustomObject]@{
    machine_name = $config.machine_name
    computer_name = $env:COMPUTERNAME
    repo_root = $script:RepoRoot
    cpu_only = $true
    cuda_visible_devices = $env:CUDA_VISIBLE_DEVICES
    nvidia_visible_devices = $env:NVIDIA_VISIBLE_DEVICES
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
}

$path = Write-JsonArtifact -Value $state -Name "bootstrap-latest.json"
Write-Output "Bootstrap complete: $path"
