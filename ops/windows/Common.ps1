Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$script:OpsRoot = Join-Path $script:RepoRoot "ops"

function Get-PythonCommand {
    $python = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($null -eq $python) {
        $python = Get-Command "python" -ErrorAction SilentlyContinue
    }
    if ($null -eq $python) {
        throw "Python is not installed or is not available on PATH."
    }
    return $python
}

function Initialize-ExperimentDirectories {
    foreach ($name in @("artifacts", "logs", "work")) {
        $path = Join-Path $script:RepoRoot $name
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

function Resolve-HuggingFaceSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$ModelId,
        [string]$Revision = "main",
        [string]$HubRoot = (Join-Path $env:USERPROFILE ".cache\huggingface\hub")
    )

    if ($ModelId -notmatch '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$') {
        throw "Invalid Hugging Face model id: $ModelId"
    }
    if ($Revision -notmatch '^[A-Za-z0-9._/-]+$' -or $Revision.Contains("..")) {
        throw "Invalid Hugging Face revision: $Revision"
    }

    $cacheName = "models--" + ($ModelId -replace "/", "--")
    $repository = Join-Path $HubRoot $cacheName
    $snapshots = Join-Path $repository "snapshots"
    $commit = $Revision
    $reference = Join-Path (Join-Path $repository "refs") $Revision
    if (Test-Path -LiteralPath $reference -PathType Leaf) {
        $commit = (Get-Content -LiteralPath $reference -Raw).Trim()
    }
    if ($commit -match '^[0-9a-fA-F]{7,64}$') {
        $candidate = Join-Path $snapshots $commit
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    $complete = @(Get-ChildItem -LiteralPath $snapshots -Directory `
        -ErrorAction SilentlyContinue | Where-Object {
            Test-Path -LiteralPath (Join-Path $_.FullName "model.safetensors.index.json") `
                -PathType Leaf
        })
    if ($complete.Count -eq 1) { return $complete[0].FullName }
    if ($complete.Count -eq 0) {
        throw "No complete cached snapshot for ${ModelId}@${Revision} under $repository"
    }
    throw "Multiple cached snapshots exist for $ModelId; pass -Snapshot explicitly"
}

function Stop-ExpertServerProcessTree {
    param(
        [string]$ServerScript = (Join-Path $script:RepoRoot "ops\python\expert_server.py"),
        [int]$Port = 8080
    )

    $serverPattern = [regex]::Escape([System.IO.Path]::GetFullPath($ServerScript))
    $portPattern = "(?:^|\s)--port\s+$Port(?:\s|$)"
    $processes = @(Get-CimInstance Win32_Process)
    $targetIds = [Collections.Generic.HashSet[int]]::new()
    foreach ($process in $processes) {
        if ($process.CommandLine -and
            $process.CommandLine -match $serverPattern -and
            $process.CommandLine -match $portPattern) {
            [void]$targetIds.Add([int]$process.ProcessId)
        }
    }
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($process in $processes) {
            if ($targetIds.Contains([int]$process.ParentProcessId) -and
                $targetIds.Add([int]$process.ProcessId)) {
                $changed = $true
            }
        }
    }
    if ($targetIds.Count -gt 0) {
        Stop-Process -Id @($targetIds) -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500
    }
    return @($targetIds)
}

function Get-DirectorySizeBytes {
    param([Parameter(Mandatory = $true)][string]$Path)

    try {
        $sum = (Get-ChildItem -LiteralPath $Path -File -Recurse -ErrorAction Stop |
            Measure-Object -Property Length -Sum).Sum
        if ($null -eq $sum) { return [int64]0 }
        return [int64]$sum
    }
    catch {
        return $null
    }
}

function Write-JsonArtifact {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $artifactPath = Join-Path (Join-Path $script:RepoRoot "artifacts") $Name
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $artifactPath -Encoding UTF8
    return $artifactPath
}
