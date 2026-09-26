param([Parameter(Mandatory = $true)][string]$ModelRoot)

$ErrorActionPreference = 'Stop'
$root = [System.IO.Path]::GetFullPath($ModelRoot)
$revision = 'efbb3b1f70a21d97fd4495240648405f7228554f'
$source = Join-Path $root ".hf-cache/hub/models--ggml-org--Qwen3.8-27B-GGUF/snapshots/$revision"
$target = Join-Path $root 'qwen3.8-27b-gguf-q4-k-m'
$candidate = Join-Path $root 'qwen3.8-27b-gguf-q4-k-m.candidate'
if (Test-Path -LiteralPath $target) { throw 'GGUF artifact already published' }
if (Test-Path -LiteralPath $candidate) { throw 'GGUF candidate already exists; inspect it first' }
$files = [ordered]@{
    'Qwen3.8-27B-Q4_K_M.gguf' = 'c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747'
    'mtp-Qwen3.8-27B-Q4_0.gguf' = 'c5be331f82fb61f5304adfa00ccd60c6743c8ce52255d66fa863446231d49ba4'
}
$sizes = @{
    'Qwen3.8-27B-Q4_K_M.gguf' = [int64]18973870528
    'mtp-Qwen3.8-27B-Q4_0.gguf' = [int64]1680271776
}
# The pinned cache can be complete even when the latest download task belongs
# to another model. Validate the source itself before creating a candidate.
foreach ($name in $files.Keys) {
    $inputPath = Join-Path $source $name
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) { throw "Missing $name" }
    if ((Get-Item -LiteralPath $inputPath).Length -ne $sizes[$name]) {
        throw "Size mismatch: $name"
    }
    $actual = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $files[$name]) { throw "SHA256 mismatch: $name" }
}
try {
    [System.IO.Directory]::CreateDirectory($candidate) | Out-Null
    foreach ($name in $files.Keys) {
        New-Item -ItemType HardLink -Path (Join-Path $candidate $name) `
            -Target (Join-Path $source $name) -ErrorAction Stop | Out-Null
    }
    $manifest = @{
        schema = 'gguf-benchmark-artifact-v1'
        model_id = 'ggml-org/Qwen3.8-27B-GGUF'
        revision = $revision
        files = $files
        published_utc = [DateTime]::UtcNow.ToString('o')
    }
    $manifestPath = Join-Path $candidate 'manifest.json'
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    @{
        schema = 'gguf-benchmark-completed-v1'
        manifest_file_sha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $candidate 'COMPLETED') -Encoding UTF8
    Move-Item -LiteralPath $candidate -Destination $target -ErrorAction Stop
} catch {
    if (Test-Path -LiteralPath $candidate) {
        Remove-Item -LiteralPath $candidate -Recurse -Force
    }
    throw
}
@{artifact = $target; status = 'published'} | ConvertTo-Json -Compress
