Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$script:OpsRoot = Join-Path $script:RepoRoot "ops"
$script:ConfigPath = Join-Path $script:OpsRoot "config\experiment.json"

function Set-CpuOnlyEnvironment {
    $env:CUDA_VISIBLE_DEVICES = "-1"
    $env:NVIDIA_VISIBLE_DEVICES = "void"
    $env:ORT_DISABLE_CUDA = "1"
    $env:GGML_CUDA = "0"
}

function Get-ExperimentConfig {
    if (-not (Test-Path -LiteralPath $script:ConfigPath)) {
        throw "Missing experiment config: $script:ConfigPath"
    }
    return Get-Content -LiteralPath $script:ConfigPath -Raw | ConvertFrom-Json
}

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

function Get-ColibriPythonCommand {
    $config = Get-ExperimentConfig
    $venv = Resolve-RepositoryPath -Path ([string]$config.colibri.python_venv)
    $pythonPath = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw "Colibri Python environment is missing: $pythonPath. Run Invoke-PrepareColibri.ps1 -InstallPythonDependencies."
    }
    return Get-Command $pythonPath -ErrorAction Stop
}

function Get-ModelSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$ModelId,
        [Parameter(Mandatory = $true)][string]$Revision,
        [Parameter(Mandatory = $true)][string]$HubRoot,
        [switch]$RequireComplete
    )

    $cacheName = "models--" + ($ModelId -replace "/", "--")
    $snapshot = Join-Path (Join-Path (Join-Path $HubRoot $cacheName) "snapshots") $Revision
    if (-not (Test-Path -LiteralPath $snapshot -PathType Container)) {
        throw "Model snapshot is missing: $snapshot"
    }

    if ($RequireComplete) {
        $configPath = Join-Path $snapshot "config.json"
        if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
            throw "Incomplete model snapshot (config.json is missing): $snapshot"
        }

        $indexPath = Join-Path $snapshot "model.safetensors.index.json"
        if (Test-Path -LiteralPath $indexPath -PathType Leaf) {
            $index = Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json
            $shards = @($index.weight_map.PSObject.Properties.Value | Sort-Object -Unique)
            $missing = @($shards | Where-Object {
                -not (Test-Path -LiteralPath (Join-Path $snapshot $_) -PathType Leaf)
            })
            if ($missing.Count -gt 0) {
                throw "Incomplete model snapshot; missing SafeTensors shards: $($missing -join ', ')"
            }
        }
        else {
            $shards = @(Get-ChildItem -LiteralPath $snapshot -Filter "*.safetensors" -File)
            if ($shards.Count -eq 0) {
                throw "Incomplete model snapshot (no SafeTensors files): $snapshot"
            }
        }
    }

    return $snapshot
}

function Resolve-RepositoryPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $script:RepoRoot $Path))
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Initialize-ExperimentDirectories {
    foreach ($name in @("artifacts", "logs", "work")) {
        $path = Join-Path $script:RepoRoot $name
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
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

function Convert-ModelConfigToSummary {
    param(
        [Parameter(Mandatory = $true)]$Config,
        [Parameter(Mandatory = $true)][string]$Source
    )

    $text = $Config
    if ($null -ne $Config.PSObject.Properties["text_config"]) {
        $text = $Config.text_config
    }

    $quantMethod = $null
    if ($null -ne $Config.PSObject.Properties["quantization_config"] -and
        $null -ne $Config.quantization_config) {
        $quantMethod = $Config.quantization_config.quant_method
    }

    function Get-OptionalProperty {
        param($Object, [string]$Name)
        $property = $Object.PSObject.Properties[$Name]
        if ($null -eq $property) { return $null }
        return $property.Value
    }

    $dtype = Get-OptionalProperty -Object $text -Name "dtype"
    if ($null -eq $dtype) {
        $dtype = Get-OptionalProperty -Object $text -Name "torch_dtype"
    }

    [PSCustomObject]@{
        source = $Source
        architectures = @(Get-OptionalProperty -Object $Config -Name "architectures")
        model_type = Get-OptionalProperty -Object $Config -Name "model_type"
        text_model_type = Get-OptionalProperty -Object $text -Name "model_type"
        dtype = $dtype
        hidden_size = Get-OptionalProperty -Object $text -Name "hidden_size"
        num_hidden_layers = Get-OptionalProperty -Object $text -Name "num_hidden_layers"
        num_experts = if ($null -ne (Get-OptionalProperty -Object $text -Name "num_experts")) {
            Get-OptionalProperty -Object $text -Name "num_experts"
        } else {
            Get-OptionalProperty -Object $text -Name "num_local_experts"
        }
        num_experts_per_tok = Get-OptionalProperty -Object $text -Name "num_experts_per_tok"
        moe_intermediate_size = if ($null -ne (Get-OptionalProperty -Object $text -Name "moe_intermediate_size")) {
            Get-OptionalProperty -Object $text -Name "moe_intermediate_size"
        } else {
            Get-OptionalProperty -Object $text -Name "intermediate_size"
        }
        quant_method = $quantMethod
    }
}

function Read-JsonSummary {
    param([Parameter(Mandatory = $true)][string]$Path)

    try {
        $config = Get-Content -LiteralPath $Path -Raw -ErrorAction Stop | ConvertFrom-Json
        return Convert-ModelConfigToSummary -Config $config -Source $Path
    }
    catch {
        return [PSCustomObject]@{
            source = $Path
            error = $_.Exception.Message
        }
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
