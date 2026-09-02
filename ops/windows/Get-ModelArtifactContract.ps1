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

[uint64]$minimumExactKvBytesPerToken = 0
$seenMinimumExactKvBytesPerToken = $false
foreach ($line in $lines) {
    $fields = @([string]$line -split "`t")
    if ($fields.Count -ne 3 -or $fields[0] -ne "attribute" -or
        $fields[1] -ne "minimum_exact_kv_bytes_per_token") {
        continue
    }
    if ($seenMinimumExactKvBytesPerToken -or
        -not [uint64]::TryParse(
            $fields[2], [ref]$minimumExactKvBytesPerToken) -or
        $minimumExactKvBytesPerToken -eq 0) {
        throw "Model artifact exact-KV resource attribute is invalid"
    }
    $seenMinimumExactKvBytesPerToken = $true
}

[PSCustomObject]@{
    schema_version = 1
    program_schema = $schema
    architecture_id = $model[2]
    maximum_context = $maximumContext
    minimum_exact_kv_bytes_per_token = $minimumExactKvBytesPerToken
    program_sha256 = $actualHash
} | ConvertTo-Json
