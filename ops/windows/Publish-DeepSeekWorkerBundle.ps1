param(
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [Parameter(Mandatory = $true)][string]$DescriptorBundle,
    [Parameter(Mandatory = $true)][string]$RoutedCatalog,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$MtpSet = "",
    [string]$MtpRoutedCatalog = "",
    [string]$StateDirectory = "",
    [ValidateSet("materialized", "external")]
    [string]$SourceExtentStorage = "materialized",
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

function Publish-MaterializedSourceExtents {
    param(
        [Parameter(Mandatory = $true)][string]$BundleRoot,
        [Parameter(Mandatory = $true)][string]$SourceRoot
    )

    $sourcePrefix = [System.IO.Path]::GetFullPath($SourceRoot)
    $sourcePrefix = $sourcePrefix.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar) +
        [System.IO.Path]::DirectorySeparatorChar
    $payloadRoot = Join-Path $BundleRoot "checkpoint"
    New-Item -ItemType Directory -Path $payloadRoot | Out-Null
    $payloadName = "source-extents.bin"
    $payloadPath = Join-Path $payloadRoot $payloadName
    $descriptorRoots = @(
        "dense",
        "typed",
        "shared",
        "mtp\dense",
        "mtp\typed",
        "mtp\shared"
    )
    $descriptorFiles = @($descriptorRoots | ForEach-Object {
        $root = Join-Path $BundleRoot $_
        if (Test-Path -LiteralPath $root -PathType Container) {
            Get-ChildItem -LiteralPath $root -Filter "extents.tsv" `
                -File -Recurse
        }
    } | Sort-Object FullName)
    if ($descriptorFiles.Count -eq 0) {
        throw "DeepSeek bundle contains no materializable source extents"
    }

    $buffer = New-Object byte[] (4MB)
    $sourceStreams = @{}
    $sourceFiles = New-Object 'System.Collections.Generic.HashSet[string]'
    $packedExtents = @{}
    $descriptorCount = 0
    $extentCount = 0
    $uniqueExtentCount = 0
    $logicalBytes = [int64]0
    $output = $null
    try {
        $output = [System.IO.FileStream]::new(
            $payloadPath,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None,
            4MB,
            [System.IO.FileOptions]::SequentialScan)
        foreach ($descriptor in $descriptorFiles) {
            $lines = @([System.IO.File]::ReadAllLines($descriptor.FullName))
            if ($lines.Count -eq 0 -or
                $lines[0] -ne "deepseek-compact-extents-v1") {
                continue
            }
            $rewritten = New-Object 'System.Collections.Generic.List[string]'
            $rewritten.Add($lines[0])
            for ($lineIndex = 1; $lineIndex -lt $lines.Count; $lineIndex += 1) {
                $line = $lines[$lineIndex]
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                $fields = @($line -split "`t")
                if ($fields.Count -ne 4) {
                    throw "Invalid DeepSeek extent row: $($descriptor.FullName)"
                }
                $destinationOffset = [int64]$fields[0]
                $bytes = [int64]$fields[1]
                $sourceOffset = [int64]$fields[2]
                $relativeSource = [string]$fields[3]
                if ($destinationOffset -lt 0 -or $bytes -le 0 -or
                    $sourceOffset -lt 0 -or
                    [System.IO.Path]::IsPathRooted($relativeSource)) {
                    throw "Unsafe DeepSeek extent row: $($descriptor.FullName)"
                }
                $sourcePath = [System.IO.Path]::GetFullPath(
                    (Join-Path $SourceRoot $relativeSource))
                if (-not $sourcePath.StartsWith(
                        $sourcePrefix,
                        [System.StringComparison]::OrdinalIgnoreCase) -or
                    -not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
                    throw "DeepSeek source extent is unavailable: $relativeSource"
                }
                $sourceLength = [int64](Get-Item -LiteralPath $sourcePath).Length
                if ($sourceOffset -gt $sourceLength -or
                    $bytes -gt $sourceLength - $sourceOffset) {
                    throw "DeepSeek source extent exceeds its shard: $relativeSource"
                }
                [void]$sourceFiles.Add($relativeSource.Replace('\', '/'))
                $key = "$sourcePath`n$sourceOffset`n$bytes"
                if ($packedExtents.ContainsKey($key)) {
                    $packedOffset = [int64]$packedExtents[$key]
                } else {
                    if (-not $sourceStreams.ContainsKey($sourcePath)) {
                        $sourceStreams[$sourcePath] = [System.IO.FileStream]::new(
                            $sourcePath,
                            [System.IO.FileMode]::Open,
                            [System.IO.FileAccess]::Read,
                            [System.IO.FileShare]::Read,
                            4MB,
                            [System.IO.FileOptions]::RandomAccess)
                    }
                    $sourceStream = $sourceStreams[$sourcePath]
                    [void]$sourceStream.Seek(
                        $sourceOffset, [System.IO.SeekOrigin]::Begin)
                    $packedOffset = [int64]$output.Position
                    $remaining = $bytes
                    while ($remaining -gt 0) {
                        $take = [int][Math]::Min([int64]$buffer.Length, $remaining)
                        $read = 0
                        while ($read -lt $take) {
                            $received = $sourceStream.Read(
                                $buffer, $read, $take - $read)
                            if ($received -le 0) {
                                throw "Short read while materializing $relativeSource"
                            }
                            $read += $received
                        }
                        $output.Write($buffer, 0, $take)
                        $remaining -= $take
                    }
                    $packedExtents[$key] = $packedOffset
                    $uniqueExtentCount += 1
                }
                $rewritten.Add(
                    "$destinationOffset`t$bytes`t$packedOffset`t$payloadName")
                $extentCount += 1
                $logicalBytes += $bytes
            }
            Write-Utf8NoBom -Path $descriptor.FullName `
                -Value (($rewritten -join "`n") + "`n")
            $descriptorCount += 1
        }
        $output.Flush($true)
    }
    finally {
        if ($null -ne $output) { $output.Dispose() }
        foreach ($stream in $sourceStreams.Values) { $stream.Dispose() }
    }
    if ($descriptorCount -eq 0 -or $extentCount -eq 0 -or
        -not (Test-Path -LiteralPath $payloadPath -PathType Leaf)) {
        throw "DeepSeek source extent materialization produced no payload"
    }
    $payloadBytes = [int64](Get-Item -LiteralPath $payloadPath).Length
    if ($payloadBytes -le 0 -or $payloadBytes -gt $logicalBytes) {
        throw "DeepSeek materialized payload accounting is invalid"
    }
    [PSCustomObject]@{
        checkpoint = "checkpoint"
        descriptor_count = $descriptorCount
        extent_count = $extentCount
        unique_extent_count = $uniqueExtentCount
        source_file_count = $sourceFiles.Count
        logical_bytes = $logicalBytes
        payload_bytes = $payloadBytes
        payload_path = "checkpoint/$payloadName"
        payload_sha256 = (Get-FileHash -LiteralPath $payloadPath `
            -Algorithm SHA256).Hash.ToLowerInvariant()
    }
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
    (Join-Path $source "config.json"),
    (Join-Path $source "tokenizer.json"),
    (Join-Path $source "tokenizer_config.json"),
    (Join-Path $source "encoding\encoding_dsv4.py"),
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
$config = Get-Content -LiteralPath (Join-Path $source "config.json") -Raw |
    ConvertFrom-Json
$layers = [int]$config.num_hidden_layers
$hiddenSize = [int]$config.hidden_size
$vocabSize = [int]$config.vocab_size
$maximumContext = [int64]$config.max_position_embeddings
$expertsPerLayer = [int]$config.n_routed_experts
$activeExperts = [int]$config.num_experts_per_tok
$sharedExperts = [int]$config.n_shared_experts
$expertIntermediate = [int]$config.moe_intermediate_size
$hashLayers = [int]$config.num_hash_layers
$mtpLayers = [int]$config.num_nextn_predict_layers
$compressionRatios = @($config.compress_ratios | ForEach-Object { [int]$_ })
if ($layers -le 0 -or $hiddenSize -le 0 -or $vocabSize -le 0 -or
    $maximumContext -le 0 -or $expertsPerLayer -le 0 -or
    $activeExperts -le 0 -or $activeExperts -gt $expertsPerLayer -or
    $sharedExperts -lt 0 -or $expertIntermediate -le 0 -or
    $hashLayers -lt 0 -or $hashLayers -gt $layers -or
    $compressionRatios.Count -lt $layers) {
    throw "DeepSeek source config has invalid or incomplete VM geometry"
}
foreach ($ratio in $compressionRatios[0..($layers - 1)]) {
    if ($ratio -notin @(0, 4, 128)) {
        throw "DeepSeek source config declares an unsupported compression ratio"
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
        [int]$mtpManifest.layers -ne $mtpLayers -or
        [int]$mtpManifest.routed_experts -ne $expertsPerLayer) {
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
        [int]$mtpPackManifest.layers -ne $mtpLayers -or
        [int]$mtpPackManifest.expert_count -ne
            ($mtpLayers * $expertsPerLayer) -or
        @(Get-ChildItem -LiteralPath $mtpRouted -Filter "experts-*.dsc" `
            -File).Count -ne $mtpLayers -or
        @(Get-ChildItem -LiteralPath $mtpRouted `
            -Filter "experts-*.dsc.commit.json" -File).Count -ne $mtpLayers) {
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
    $routedManifestPath = Join-Path $routed "manifest.json"
    if (-not (Test-Path -LiteralPath $routedManifestPath `
            -PathType Leaf) -or
        @(Get-ChildItem -LiteralPath $routed -Filter "experts-*.dsc" `
            -File).Count -ne $layers -or
        @(Get-ChildItem -LiteralPath $routed `
            -Filter "experts-*.dsc.commit.json" -File).Count -ne $layers) {
        throw "DeepSeek compact pack is not atomically complete"
    }
    $routedManifest = Get-Content -LiteralPath $routedManifestPath -Raw |
        ConvertFrom-Json
    if ([int]$routedManifest.layers -ne $layers -or
        [int]$routedManifest.experts_per_layer -ne $expertsPerLayer -or
        [int64]$routedManifest.expert_count -ne
            ([int64]$layers * $expertsPerLayer)) {
        throw "DeepSeek compact pack disagrees with source config"
    }
}

$modelHash = (Get-FileHash -LiteralPath `
    (Join-Path $source "model.safetensors.index.json") `
    -Algorithm SHA256).Hash.ToLowerInvariant()
$mainNamespace = [Convert]::ToUInt64($modelHash.Substring(0, 15), 16)
if ($mainNamespace -eq 0) { $mainNamespace = 1 }
$mtpNamespace = $mainNamespace + 1
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
    $tokenizerRoot = Join-Path $partial "tokenizer"
    $tokenizerEncoding = Join-Path $tokenizerRoot "encoding"
    New-Item -ItemType Directory -Path $tokenizerEncoding -Force | Out-Null
    foreach ($name in @("config.json", "tokenizer.json", "tokenizer_config.json")) {
        Copy-Item -LiteralPath (Join-Path $source $name) `
            -Destination (Join-Path $tokenizerRoot $name)
    }
    Copy-Item -LiteralPath (Join-Path $source "encoding\encoding_dsv4.py") `
        -Destination (Join-Path $tokenizerEncoding "encoding_dsv4.py")
    $tokenizerEntries = foreach ($relative in @(
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "encoding/encoding_dsv4.py"
    )) {
        $path = Join-Path $tokenizerRoot ($relative -replace '/', '\')
        [ordered]@{
            path = $relative
            bytes = (Get-Item -LiteralPath $path).Length
            sha256 = (Get-FileHash -LiteralPath $path `
                -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $materialized = if ($SourceExtentStorage -eq "materialized") {
        Publish-MaterializedSourceExtents -BundleRoot $partial `
            -SourceRoot $source
    } else { $null }
    $checkpoint = if ($null -ne $materialized) {
        [string]$materialized.checkpoint
    } else { $source }
    New-Item -ItemType Directory -Path $state -Force | Out-Null
    $runtimeLines = @(
        "deepseek-worker-bundle-v$bundleVersion",
        "model_id`t$mainNamespace",
        "model_sha256`t$modelHash",
        "checkpoint`t$checkpoint",
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
    $modelProgramLines = @(
        "expert-runtime-model-v1",
        "model`t3`t$($config.model_type)`t$vocabSize`t$maximumContext`t$hiddenSize",
        "program_input`ttoken_ids`trequest.token_ids`tbatch.token-id.u32.host.v1",
        "program_input`tpositions`trequest.positions`tbatch.position.u32.host.v1",
        "program_output`tnext_token_ids`tresponse.token_ids`tbatch.token-id.u32.host.v1",
        "kernel`tembedding.lookup.int8-row.v1`t1",
        "kernel`tblock.compressed-sparse-attention.hca.v1`t1",
        "kernel`trouter.deepseek.v4.topk.v1`t1",
        "kernel`tmoe.swiglu.routed.v1`t1",
        "kernel`thead.rmsnorm.argmax.int8-row.v1`t1"
    )
    if ($mtp) {
        $modelProgramLines += "kernel`tdecode.speculative.verify.v1`t1"
        $modelProgramLines += "exact_decode`tdecode.speculative.verify.v1`t1`t2"
        $modelProgramLines += "exact_decode_parameter`ttarget_component_index`t0"
        $modelProgramLines += "exact_decode_parameter`tdraft_component_index`t1"
    }
    $modelProgramLines += (
        "component`tdecoder`t0`t$layers`t$expertsPerLayer`t$activeExperts" +
        "`t$sharedExperts`t$hiddenSize`t$expertIntermediate" +
        "`tmoe.swiglu.routed.v1`t1`t2`t2" +
        "`tfp4.e2m1.ue8m0.block32"
    )
    $modelProgramLines += "router`tdecoder`trouter.deepseek.v4.topk.v1`t1"
    $modelProgramLines += "router_parameter`tdecoder`tnormalize`t1"
    $modelProgramLines += "router_parameter`tdecoder`thash_layers`t$hashLayers"
    if ($mtp) {
        $modelProgramLines += (
            "component`tdraft`t1`t$mtpLayers`t$expertsPerLayer`t$activeExperts" +
            "`t$sharedExperts`t$hiddenSize`t$expertIntermediate" +
            "`tmoe.swiglu.routed.v1`t1`t2`t2" +
            "`tfp4.e2m1.ue8m0.block32"
        )
        $modelProgramLines += "router`tdraft`trouter.deepseek.v4.topk.v1`t1"
        $modelProgramLines += "router_parameter`tdraft`tnormalize`t1"
    }
    $operation = 0
    $modelProgramLines += (
        "operation`t$operation`t-" +
        "`tembedding.lookup.int8-row.v1`t1`t-`t0"
    )
    $modelProgramLines += (
        "operation_input`t$operation`ttoken_ids" +
        "`trequest.token_ids`tbatch.token-id.u32.host.v1"
    )
    $modelProgramLines += (
        "operation_output`t$operation`thidden" +
        "`thidden.0`tbatch.hca4.hidden.f32.cuda.v1"
    )
    $operation += 1
    foreach ($layer in 0..($layers - 1)) {
        $ratio = $compressionRatios[$layer]
        $blockHidden = "layer.$layer.after_attention"
        $expertInput = "layer.$layer.expert_input"
        $routeIndices = "layer.$layer.route_indices"
        $routeWeights = "layer.$layer.route_weights"
        $residual = "layer.$layer.residual"
        $modelProgramLines += (
            "layer`t$layer`tblock.compressed-sparse-attention.hca.v1" +
            "`t1`tdecoder`t$layer"
        )
        $modelProgramLines += "layer_parameter`t$layer`tcompression_ratio`t$ratio"
        $modelProgramLines += (
            "operation`t$operation`t$layer" +
            "`tblock.compressed-sparse-attention.hca.v1`t1`t-`t0"
        )
        $modelProgramLines += "operation_parameter`t$operation`tcompression_ratio`t$ratio"
        $modelProgramLines += (
            "operation_input`t$operation`thidden`thidden.$layer" +
            "`tbatch.hca4.hidden.f32.cuda.v1"
        )
        $modelProgramLines += (
            "operation_input`t$operation`tpositions`trequest.positions" +
            "`tbatch.position.u32.host.v1"
        )
        $modelProgramLines += (
            "operation_output`t$operation`thidden`t$blockHidden" +
            "`tbatch.hca4.hidden.f32.cuda.v1"
        )
        $operation += 1
        $modelProgramLines += (
            "operation`t$operation`t$layer" +
            "`trouter.deepseek.v4.topk.v1`t1`tdecoder`t$layer"
        )
        if ($layer + 1 -lt $layers) {
            $modelProgramLines += (
                "operation_parameter`t$operation" +
                "`tprefetch_target_component_layer`t$($layer + 1)"
            )
        }
        $modelProgramLines += (
            "operation_input`t$operation`thidden`t$blockHidden" +
            "`tbatch.hca4.hidden.f32.cuda.v1"
        )
        $modelProgramLines += (
            "operation_input`t$operation`ttoken_ids`trequest.token_ids" +
            "`tbatch.token-id.u32.host.v1"
        )
        $modelProgramLines += (
            "operation_output`t$operation`texpert_input`t$expertInput" +
            "`tbatch.hidden.f32.cuda.v1"
        )
        $modelProgramLines += (
            "operation_output`t$operation`troute_indices`t$routeIndices" +
            "`tbatch.route-index.u32.cuda.v1"
        )
        $modelProgramLines += (
            "operation_output`t$operation`troute_weights`t$routeWeights" +
            "`tbatch.route-weight.f32.cuda.v1"
        )
        $modelProgramLines += (
            "operation_output`t$operation`tresidual`t$residual" +
            "`tbatch.hca4.hidden.f32.cuda.v1"
        )
        $operation += 1
        $modelProgramLines += (
            "operation`t$operation`t$layer" +
            "`tmoe.swiglu.routed.v1`t1`tdecoder`t$layer"
        )
        $modelProgramLines += (
            "operation_input`t$operation`texpert_input`t$expertInput" +
            "`tbatch.hidden.f32.cuda.v1"
        )
        $modelProgramLines += (
            "operation_input`t$operation`troute_indices`t$routeIndices" +
            "`tbatch.route-index.u32.cuda.v1"
        )
        $modelProgramLines += (
            "operation_input`t$operation`troute_weights`t$routeWeights" +
            "`tbatch.route-weight.f32.cuda.v1"
        )
        $modelProgramLines += (
            "operation_input`t$operation`tresidual`t$residual" +
            "`tbatch.hca4.hidden.f32.cuda.v1"
        )
        $modelProgramLines += (
            "operation_output`t$operation`thidden`thidden.$($layer + 1)" +
            "`tbatch.hca4.hidden.f32.cuda.v1"
        )
        $operation += 1
    }
    $modelProgramLines += (
        "operation`t$operation`t-" +
        "`thead.rmsnorm.argmax.int8-row.v1`t1`t-`t0"
    )
    $modelProgramLines += (
        "operation_input`t$operation`thidden`thidden.$layers" +
        "`tbatch.hca4.hidden.f32.cuda.v1"
    )
    $modelProgramLines += (
        "operation_output`t$operation`ttoken_ids`tresponse.token_ids" +
        "`tbatch.token-id.u32.host.v1"
    )
    $prefetchTargets = @($modelProgramLines | ForEach-Object {
        $fields = @($_ -split "`t")
        if ($fields.Count -eq 4 -and
            $fields[0] -eq "operation_parameter" -and
            $fields[2] -eq "prefetch_target_component_layer") {
            [int]$fields[3]
        }
    })
    if ($prefetchTargets.Count -ne $layers - 1) {
        throw "DeepSeek program does not declare every cross-layer prefetch edge"
    }
    for ($index = 0; $index -lt $prefetchTargets.Count; $index += 1) {
        if ($prefetchTargets[$index] -ne $index + 1) {
            throw "DeepSeek program has a non-canonical prefetch target"
        }
    }
    $modelProgramText = ($modelProgramLines -join "`n") + "`n"
    $modelProgramPath = Join-Path $partial "runtime-model.tsv"
    Write-Utf8NoBom -Path $modelProgramPath -Value $modelProgramText
    $modelProgramHash = (Get-FileHash -LiteralPath $modelProgramPath `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    $modelProgramBytes = (Get-Item -LiteralPath $modelProgramPath).Length
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
            source_extent_storage = if ($null -ne $materialized) {
                "materialized-pack"
            } else { "external-snapshot" }
        }
        quantization = [ordered]@{
            dense = "fp8-e4m3-ue8m0-to-sm86-int8-per-row"
            shared = "fp8-e4m3-ue8m0-to-sm86-int8-per-row"
            routed = "fp4-e2m1-ue8m0-block32-direct"
        }
        architecture = [ordered]@{
            model_type = [string]$config.model_type
            layers = $layers
            hidden_size = $hiddenSize
            vocab_size = $vocabSize
            max_position_embeddings = $maximumContext
            expert_intermediate_size = $expertIntermediate
            experts_per_layer = $expertsPerLayer
            active_experts = $activeExperts
            shared_experts_per_layer = $sharedExperts
            namespace_id = $mainNamespace
            mtp_namespace_id = $mtpNamespace
            hash_layers = $hashLayers
            compression_ratios = @($compressionRatios[0..($layers - 1)])
            cross_layer_prefetch_targets = @($prefetchTargets)
            multi_token_prediction_layers = if ($mtp) { $mtpLayers } else { 0 }
        }
        model_program = [ordered]@{
            format = "expert-runtime-model-v1"
            path = "runtime-model.tsv"
            bytes = $modelProgramBytes
            sha256 = $modelProgramHash
        }
        tokenizer = [ordered]@{
            path = "tokenizer"
            files = @($tokenizerEntries)
        }
        masses = [ordered]@{
            routed_payload_bytes = if ($packed) {
                [int64]$routedManifest.pack_bytes
            } else { 0 }
            routed_shards = if ($packed) { [int]$routedManifest.shards } else { 0 }
            mtp_source_bytes = if ($mtp) { [long]$mtpManifest.source_bytes } else { 0 }
            source_extent_logical_bytes = if ($null -ne $materialized) {
                [int64]$materialized.logical_bytes
            } else { 0 }
            source_extent_payload_bytes = if ($null -ne $materialized) {
                [int64]$materialized.payload_bytes
            } else { 0 }
        }
        source_extents = if ($null -ne $materialized) {
            [ordered]@{
                path = [string]$materialized.payload_path
                sha256 = [string]$materialized.payload_sha256
                descriptors = [int]$materialized.descriptor_count
                extents = [int]$materialized.extent_count
                unique_extents = [int]$materialized.unique_extent_count
                source_files = [int]$materialized.source_file_count
            }
        } else { $null }
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
    checkpoint = if ($SourceExtentStorage -eq "materialized") {
        Join-Path $destination "checkpoint"
    } else { $source }
    source_extent_storage = $SourceExtentStorage
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
