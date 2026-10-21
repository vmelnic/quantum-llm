param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$RegistryFile = ''
)

. (Join-Path $PSScriptRoot 'Common.ps1')
. (Join-Path $PSScriptRoot 'LlamaSpec.ps1')
if (-not $env:MODEL_ROOT) { throw 'MODEL_ROOT is required' }
if (-not $RegistryFile) { $RegistryFile = Join-Path $script:OpsRoot 'llama-models.json' }
$registry = Get-Content -LiteralPath $RegistryFile -Raw | ConvertFrom-Json
if ($registry.schema -ne 'llama-models-v1') { throw 'Invalid llama model registry' }
$matches = @($registry.models | Where-Object { $Model -in $_.aliases })
if ($matches.Count -ne 1) { throw "Unknown or ambiguous llama model: $Model" }
$spec = $matches[0]
if ($spec.artifact -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$' -or
    $spec.file -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$' -or
    $spec.repository -notmatch '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$' -or
    $spec.revision -notmatch '^[0-9a-f]{40}$' -or
    $spec.sha256 -notmatch '^[0-9a-f]{64}$') { throw 'Invalid artifact identity' }
$assets = @($spec)
$projector = Get-LlamaProjector -Spec $spec
if ($null -ne $projector) { $assets += $projector }
$target = Join-Path $env:MODEL_ROOT $spec.artifact
$candidate = "$target.candidate"

function Get-GgufSource($Asset) {
    $cache = 'models--' + ($Asset.repository -replace '/', '--')
    return Join-Path $env:MODEL_ROOT ".hf-cache/hub/$cache/snapshots/$($Asset.revision)/$($Asset.file)"
}

function Test-GgufFile([string]$Path, $Asset) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or
        (Get-Item -LiteralPath $Path).Length -ne [int64]$Asset.bytes) {
        throw "Missing or incomplete GGUF: $Path"
    }
    $stream = [IO.File]::OpenRead($Path)
    try {
        $header = New-Object byte[] 8
        if ($stream.Read($header, 0, 8) -ne 8 -or
            [Text.Encoding]::ASCII.GetString($header, 0, 4) -ne 'GGUF' -or
            [BitConverter]::ToUInt32($header, 4) -ne 3) { throw 'Invalid GGUF v3 header' }
    } finally { $stream.Dispose() }
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $Asset.sha256) { throw "GGUF SHA256 mismatch: $Path" }
}

if (Test-Path -LiteralPath $target) {
    $manifestPath = Join-Path $target 'manifest.json'
    $completion = Get-Content (Join-Path $target 'COMPLETED') -Raw | ConvertFrom-Json
    if ($completion.schema -ne 'llama-completed-v1' -or
        $completion.manifest_sha256 -ne (Get-FileHash $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()) {
        throw 'Published llama manifest failed integrity validation'
    }
    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.schema -ne 'llama-artifact-v1' -or
        ($manifest.model | ConvertTo-Json -Depth 8 -Compress) -ne
        ($spec | ConvertTo-Json -Depth 8 -Compress)) { throw 'Published artifact differs from registry' }
    foreach ($asset in $assets) { Test-GgufFile (Join-Path $target $asset.file) $asset }
    @{status = 'validated'; artifact = $target; model = $spec.model_id} | ConvertTo-Json -Compress
    exit
}
if (Test-Path -LiteralPath $candidate) { throw "Inspect existing candidate before publishing: $candidate" }
foreach ($asset in $assets) { Test-GgufFile (Get-GgufSource $asset) $asset }
try {
    [IO.Directory]::CreateDirectory($candidate) | Out-Null
    foreach ($asset in $assets) {
        New-Item -ItemType HardLink -Path (Join-Path $candidate $asset.file) -Target (Get-GgufSource $asset) | Out-Null
    }
    $manifestPath = Join-Path $candidate 'manifest.json'
    @{schema = 'llama-artifact-v1'; model = $spec; published_utc = [DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    @{schema = 'llama-completed-v1'; manifest_sha256 = (Get-FileHash $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $candidate 'COMPLETED') -Encoding UTF8
    Move-Item -LiteralPath $candidate -Destination $target
} catch {
    if (Test-Path -LiteralPath $candidate) { Remove-Item -LiteralPath $candidate -Recurse -Force }
    throw
}
@{status = 'published'; artifact = $target; model = $spec.model_id} | ConvertTo-Json -Compress
