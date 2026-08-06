param(
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [Parameter(Mandatory = $true)][string]$DescriptorBundle,
    [Parameter(Mandatory = $true)][string]$RoutedCatalog,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$MtpSet = "",
    [string]$MtpRoutedCatalog = "",
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
$mtp = if ($MtpSet) { [System.IO.Path]::GetFullPath($MtpSet) } else { "" }
$mtpRouted = if ($MtpRoutedCatalog) {
    [System.IO.Path]::GetFullPath($MtpRoutedCatalog)
} else { "" }
if ([bool]$mtp -ne [bool]$mtpRouted) {
    throw "MtpSet and MtpRoutedCatalog must be published together"
}
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
if ($mtp) {
    foreach ($path in @(
        (Join-Path $mtp "manifest.json"),
        (Join-Path $mtp "dense\dense-set.tsv"),
        (Join-Path $mtp "typed\manifest.json"),
        (Join-Path $mtp "shared\manifest.json"),
        (Join-Path $mtp "routed\catalog.tsv"),
        (Join-Path $mtp "routed\extents.tsv")
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "DeepSeek MTP bundle dependency missing: $path"
        }
    }
    $mtpManifest = Get-Content -LiteralPath (Join-Path $mtp "manifest.json") `
        -Raw | ConvertFrom-Json
    if ($mtpManifest.format -ne "deepseek-mtp-resource-set-v1" -or
        [int]$mtpManifest.layers -ne 1 -or
        [int]$mtpManifest.routed_experts -ne 256) {
        throw "Unsupported DeepSeek MTP resource set"
    }
    foreach ($path in @(
        (Join-Path $mtpRouted "manifest.json"),
        (Join-Path $mtpRouted "catalog.tsv"),
        (Join-Path $mtpRouted "extents.tsv")
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "DeepSeek MTP routed pack dependency missing: $path"
        }
    }
    $mtpPackManifest = Get-Content -LiteralPath `
        (Join-Path $mtpRouted "manifest.json") -Raw | ConvertFrom-Json
    if ($mtpPackManifest.format -ne "deepseek-routed-compact-pack-v1" -or
        [int]$mtpPackManifest.layers -ne 1 -or
        [int]$mtpPackManifest.expert_count -ne 256 -or
        @(Get-ChildItem -LiteralPath $mtpRouted -Filter "experts-*.dsc" `
            -File).Count -ne 1 -or
        @(Get-ChildItem -LiteralPath $mtpRouted `
            -Filter "experts-*.dsc.commit.json" -File).Count -ne 1) {
        throw "Unsupported or incomplete DeepSeek MTP routed pack"
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
$mtpHash = if ($mtp) {
    (Get-FileHash -LiteralPath (Join-Path $mtp "manifest.json") `
        -Algorithm SHA256).Hash.ToLowerInvariant()
} else { "" }
$mtpRoutedHash = if ($mtpRouted) {
    (Get-FileHash -LiteralPath (Join-Path $mtpRouted "catalog.tsv") `
        -Algorithm SHA256).Hash.ToLowerInvariant()
} else { "" }
$bundleVersion = if ($mtp) { 3 } else { 1 }

New-Item -ItemType Directory -Path $partial | Out-Null
try {
    foreach ($name in @("dense", "typed", "shared")) {
        Copy-Item -LiteralPath (Join-Path $descriptors $name) `
            -Destination (Join-Path $partial $name) -Recurse
    }
    if ($mtp) {
        Copy-Item -LiteralPath $mtp -Destination (Join-Path $partial "mtp") `
            -Recurse
    }
    New-Item -ItemType Directory -Path $state -Force | Out-Null
    $runtimeLines = @(
        "deepseek-worker-bundle-v$bundleVersion",
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
    if ($mtp) {
        $runtimeLines += "mtp`tmtp"
        $runtimeLines += "mtp_routed`t$mtpRouted"
    }
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
            version = $bundleVersion
            routed_storage = if ($packed) { "compact-pack" } else { "source-extents" }
            mtp_routed_storage = if ($mtpRouted) { "compact-pack" } else { $null }
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
            multi_token_prediction_layers = if ($mtp) { 1 } else { 0 }
        }
        masses = [ordered]@{
            routed_payload_bytes = 147169738752
            routed_shards = if ($packed) { 43 } else { 0 }
            mtp_source_bytes = if ($mtp) { [long]$mtpManifest.source_bytes } else { 0 }
        }
        indexes = [ordered]@{
            dense_sha256 = $denseHash
            experts_sha256 = $expertsHash
            mtp_sha256 = if ($mtp) { $mtpHash } else { $null }
            mtp_routed_sha256 = if ($mtpRouted) { $mtpRoutedHash } else { $null }
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
    mtp_resource_set = $mtp
    mtp_routed_catalog = $mtpRouted
    model_sha256 = $modelHash
    status = "published"
}
$artifact = Write-JsonArtifact -Value $result `
    -Name "deepseek-worker-bundle-latest.json"
$result | ConvertTo-Json
Write-Output "DeepSeek worker bundle artifact: $artifact"
