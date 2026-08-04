. (Join-Path $PSScriptRoot "Common.ps1")

Initialize-ExperimentDirectories

$state = [PSCustomObject]@{
    schema_version = 1
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    computer_name = $env:COMPUTERNAME
    user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    repo_root = $script:RepoRoot
    powershell_version = $PSVersionTable.PSVersion.ToString()
}

$path = Write-JsonArtifact -Value $state -Name "bootstrap-latest.json"
Write-Output "Bootstrap complete: $path"
