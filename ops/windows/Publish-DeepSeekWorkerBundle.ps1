param(
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [Parameter(Mandatory = $true)][string]$DescriptorBundle,
    [Parameter(Mandatory = $true)][string]$RoutedCatalog,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$StateDirectory = "",
    [ValidateRange(0.001, 1000.0)][double]$CpuMillisecondsPerSelection = 9.342,
    [ValidateRange(0.001, 1000.0)][double]$GpuMillisecondsPerSelection = 0.543,
    [ValidateRange(0.001, 1000.0)][double]$H2DGigabytesPerSecond = 3.393
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Value
    )
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Value, $encoding)
}

$source = [System.IO.Path]::GetFullPath($Snapshot)
$descriptors = [System.IO.Path]::GetFullPath($DescriptorBundle)
$routed = [System.IO.Path]::GetFullPath($RoutedCatalog)
$destination = [System.IO.Path]::GetFullPath($Output)
$state = if ($StateDirectory) {
    [System.IO.Path]::GetFullPath($StateDirectory)
} else {
    $destination + ".state"
}
$partial = $destination + ".partial"

foreach ($path in @(
    (Join-Path $source "model.safetensors.index.json"),
    (Join-Path $descriptors "dense\dense-set.tsv"),
    (Join-Path $descriptors "typed\typed-set.tsv"),
    (Join-Path $descriptors "shared\shared-set.tsv"),
    (Join-Path $routed "catalog.tsv"),
    (Join-Path $routed "extents.tsv")
)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "DeepSeek bundle dependency missing: $path"
    }
}
if ((Test-Path -LiteralPath $destination) -or
    (Test-Path -LiteralPath $partial)) {
    throw "DeepSeek worker bundle output already exists: $destination"
}

$catalogHeader = Get-Content -LiteralPath (Join-Path $routed "catalog.tsv") `
    -TotalCount 1
$packed = $catalogHeader -eq "deepseek-routed-pack-catalog-v1"
if (-not $packed -and $catalogHeader -ne "deepseek-routed-catalog-v1") {
    throw "Unsupported DeepSeek routed catalog"
}
if ($packed) {
    if (-not (Test-Path -LiteralPath (Join-Path $routed "manifest.json") `
            -PathType Leaf) -or
        @(Get-ChildItem -LiteralPath $routed -Filter "experts-*.dsc" `
            -File).Count -ne 43 -or
        @(Get-ChildItem -LiteralPath $routed `
            -Filter "experts-*.dsc.commit.json" -File).Count -ne 43) {
        throw "DeepSeek compact pack is not atomically complete"
    }
}

$modelHash = (Get-FileHash -LiteralPath `
    (Join-Path $source "model.safetensors.index.json") `
    -Algorithm SHA256).Hash.ToLowerInvariant()
$denseHash = (Get-FileHash -LiteralPath `
    (Join-Path $descriptors "dense\dense-set.tsv") `
    -Algorithm SHA256).Hash.ToLowerInvariant()
$expertsHash = (Get-FileHash -LiteralPath `
    (Join-Path $routed "catalog.tsv") `
    -Algorithm SHA256).Hash.ToLowerInvariant()

New-Item -ItemType Directory -Path $partial | Out-Null
try {
    foreach ($name in @("dense", "typed", "shared")) {
        Copy-Item -LiteralPath (Join-Path $descriptors $name) `
            -Destination (Join-Path $partial $name) -Recurse
    }
    New-Item -ItemType Directory -Path $state -Force | Out-Null
    $runtimeLines = @(
        "deepseek-worker-bundle-v1",
        "model_id`t17",
        "model_sha256`t$modelHash",
        "checkpoint`t$source",
        "dense`tdense",
        "typed`ttyped",
        "shared`tshared",
        "routed`t$routed",
        "census`t$(Join-Path $state 'route-census')",
        "cpu_ns_per_selection`t$([long]($CpuMillisecondsPerSelection * 1000000.0))",
        "gpu_ns_per_selection`t$([long]($GpuMillisecondsPerSelection * 1000000.0))",
        "h2d_bytes_per_second`t$([long]($H2DGigabytesPerSecond * 1000000000.0))"
    )
    $runtimeText = ($runtimeLines -join "`n") + "`n"
    Write-Utf8NoBom -Path (Join-Path $partial "runtime.tsv") `
        -Value $runtimeText
    $runtimeHash = (Get-FileHash -LiteralPath `
        (Join-Path $partial "runtime.tsv") -Algorithm SHA256).Hash.ToLowerInvariant()
    $manifest = [ordered]@{
        schema_version = 1
        source = [ordered]@{
            model_id = "deepseek-ai/DeepSeek-V4-Flash"
            revision = Split-Path $source -Leaf
            checkpoint_index_sha256 = $modelHash
        }
        format = [ordered]@{
            name = "deepseek-worker-bundle"
            version = 1
            routed_storage = if ($packed) { "compact-pack" } else { "source-extents" }
        }
        quantization = [ordered]@{
            dense = "fp8-e4m3-ue8m0-to-sm86-int8-per-row"
            shared = "fp8-e4m3-ue8m0-to-sm86-int8-per-row"
            routed = "fp4-e2m1-ue8m0-block32-direct"
        }
        architecture = [ordered]@{
            model_type = "deepseek_v4_flash"
            layers = 43
            hidden_size = 4096
            experts_per_layer = 256
            active_experts = 6
        }
        masses = [ordered]@{
            routed_payload_bytes = 147169738752
            routed_shards = if ($packed) { 43 } else { 0 }
        }
        indexes = [ordered]@{
            dense_sha256 = $denseHash
            experts_sha256 = $expertsHash
        }
        integrity = [ordered]@{
            content_sha256 = $runtimeHash
        }
        placement_seed = [ordered]@{
            cpu_ns_per_selection = [long]($CpuMillisecondsPerSelection * 1000000.0)
            gpu_ns_per_selection = [long]($GpuMillisecondsPerSelection * 1000000.0)
            h2d_bytes_per_second = [long]($H2DGigabytesPerSecond * 1000000000.0)
        }
    }
    Write-Utf8NoBom -Path (Join-Path $partial "manifest.json") `
        -Value (($manifest | ConvertTo-Json -Depth 8) + "`n")
    Move-Item -LiteralPath $partial -Destination $destination
}
catch {
    Remove-Item -LiteralPath $partial -Recurse -Force -ErrorAction SilentlyContinue
    throw
}

$result = [PSCustomObject]@{
    schema_version = 1
    output = $destination
    state_directory = $state
    checkpoint = $source
    routed_catalog = $routed
    durable_routed_pack = $packed
    model_sha256 = $modelHash
    status = "published"
}
$artifact = Write-JsonArtifact -Value $result `
    -Name "deepseek-worker-bundle-latest.json"
$result | ConvertTo-Json
Write-Output "DeepSeek worker bundle artifact: $artifact"
