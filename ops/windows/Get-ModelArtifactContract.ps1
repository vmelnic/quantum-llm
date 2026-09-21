param(
    [Parameter(Mandatory = $true)][string]$Container
)

. (Join-Path $PSScriptRoot "Common.ps1")

$root = [System.IO.Path]::GetFullPath($Container).TrimEnd('\')
$manifestPath = Join-Path $root "manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Model artifact manifest is missing: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($null -eq $manifest.model_program -or
    $manifest.model_program.format -ne "expert-runtime-model-v1") {
    throw "Model artifact does not declare an Expert Runtime program"
}
$relative = [string]$manifest.model_program.path
if ([string]::IsNullOrWhiteSpace($relative) -or
    [System.IO.Path]::IsPathRooted($relative)) {
    throw "Model artifact program path is not relative"
}
$program = [System.IO.Path]::GetFullPath((Join-Path $root $relative))
if (-not $program.StartsWith(
        $root + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
    -not (Test-Path -LiteralPath $program -PathType Leaf)) {
    throw "Model artifact program path escapes the artifact or is absent"
}
$actualBytes = (Get-Item -LiteralPath $program).Length
$actualHash = (Get-FileHash -LiteralPath $program -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualBytes -ne [int64]$manifest.model_program.bytes -or
    $actualHash -ne [string]$manifest.model_program.sha256) {
    throw "Model artifact program does not match its authenticated manifest"
}
$lines = @(Get-Content -LiteralPath $program)
if ($lines.Count -lt 2 -or $lines[0] -ne "expert-runtime-model-v1") {
    throw "Model artifact program header is invalid"
}
$model = @([string]$lines[1] -split "`t")
[uint32]$schema = 0
[uint32]$maximumContext = 0
if ($model.Count -lt 6 -or $model[0] -ne "model" -or
    -not [uint32]::TryParse($model[1], [ref]$schema) -or $schema -ne 3 -or
    -not [uint32]::TryParse($model[4], [ref]$maximumContext) -or
    $maximumContext -le 1) {
    throw "Model artifact program descriptor is invalid"
}

[uint32]$maximumThinkingTokens = 0
$tokenizerProperty = $manifest.PSObject.Properties["tokenizer"]
$samplingProperty = $null
if ($null -ne $tokenizerProperty -and $null -ne $tokenizerProperty.Value) {
    $samplingProperty = $tokenizerProperty.Value.PSObject.Properties["sampling"]
}
if ($null -ne $samplingProperty -and $null -ne $samplingProperty.Value) {
    $sampling = $samplingProperty.Value
    if ([string]::IsNullOrWhiteSpace(
            [string]$sampling.schema)) {
        throw "Model artifact sampling policy schema is invalid"
    }
    $maximumThinkingProperty = `
        $sampling.PSObject.Properties["maximum_thinking_tokens"]
    if ($null -ne $maximumThinkingProperty) {
        if (-not [uint32]::TryParse(
                [string]$maximumThinkingProperty.Value,
                [ref]$maximumThinkingTokens) -or
            $maximumThinkingTokens -eq 0 -or
            $maximumThinkingTokens -ge $maximumContext) {
            throw "Model artifact maximum-thinking-token policy is invalid"
        }
    }
}

[uint64]$minimumExactKvBytesPerToken = 0
$seenMinimumExactKvBytesPerToken = $false
[uint32]$mtpLayers = 0
$seenMtpLayers = $false
[uint32]$exactDecodeAbi = 0
[uint32]$exactDecodeMaximumEmittedTokens = 0
$exactDecodeCapability = ""
$seenExactDecode = $false
$exactDecodeParameters = @{}
foreach ($line in $lines) {
    $fields = @([string]$line -split "`t")
    if ($fields.Count -eq 3 -and $fields[0] -eq "attribute" -and
        $fields[1] -eq "minimum_exact_kv_bytes_per_token") {
        if ($seenMinimumExactKvBytesPerToken -or
            -not [uint64]::TryParse(
                $fields[2], [ref]$minimumExactKvBytesPerToken) -or
            $minimumExactKvBytesPerToken -eq 0) {
            throw "Model artifact exact-KV resource attribute is invalid"
        }
        $seenMinimumExactKvBytesPerToken = $true
        continue
    }
    if ($fields.Count -eq 3 -and $fields[0] -eq "attribute" -and
        $fields[1] -eq "mtp_layers") {
        if ($seenMtpLayers -or
            -not [uint32]::TryParse($fields[2], [ref]$mtpLayers)) {
            throw "Model artifact MTP layer attribute is invalid"
        }
        $seenMtpLayers = $true
        continue
    }
    if ($fields.Count -eq 4 -and $fields[0] -eq "exact_decode") {
        if ($seenExactDecode -or [string]::IsNullOrWhiteSpace($fields[1]) -or
            -not [uint32]::TryParse($fields[2], [ref]$exactDecodeAbi) -or
            -not [uint32]::TryParse(
                $fields[3], [ref]$exactDecodeMaximumEmittedTokens)) {
            throw "Model artifact exact-decode record is invalid"
        }
        $exactDecodeCapability = $fields[1]
        $seenExactDecode = $true
        continue
    }
    if ($fields.Count -eq 3 -and $fields[0] -eq "exact_decode_parameter") {
        if ([string]::IsNullOrWhiteSpace($fields[1]) -or
            $exactDecodeParameters.ContainsKey($fields[1])) {
            throw "Model artifact exact-decode parameters are invalid"
        }
        [uint32]$parameterValue = 0
        if (-not [uint32]::TryParse($fields[2], [ref]$parameterValue)) {
            throw "Model artifact exact-decode parameter is not an unsigned integer"
        }
        $exactDecodeParameters[$fields[1]] = $parameterValue
    }
}

function Get-ExactDecodeParameter {
    param([Parameter(Mandatory = $true)][string]$Name)
    if ($exactDecodeParameters.ContainsKey($Name)) {
        return [uint32]$exactDecodeParameters[$Name]
    }
    return [uint32]0
}

[PSCustomObject]@{
    schema_version = 1
    program_schema = $schema
    architecture_id = $model[2]
    maximum_context = $maximumContext
    maximum_thinking_tokens = $maximumThinkingTokens
    minimum_exact_kv_bytes_per_token = $minimumExactKvBytesPerToken
    mtp_layers = $mtpLayers
    exact_decode_capability = $exactDecodeCapability
    exact_decode_abi = $exactDecodeAbi
    exact_decode_maximum_emitted_tokens = $exactDecodeMaximumEmittedTokens
    exact_decode_draft_depth = Get-ExactDecodeParameter -Name "draft_depth"
    exact_decode_draft_vocabulary_size = Get-ExactDecodeParameter `
        -Name "draft_vocabulary_size"
    exact_decode_mtp_kv_encoding = Get-ExactDecodeParameter `
        -Name "mtp_kv_encoding"
    program_sha256 = $actualHash
} | ConvertTo-Json
