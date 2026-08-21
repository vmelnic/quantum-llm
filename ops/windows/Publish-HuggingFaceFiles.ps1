param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [Parameter(Mandatory = $true)][string]$StableName,
    [Parameter(Mandatory = $true)][string]$Files
)

. (Join-Path $PSScriptRoot "Common.ps1")

if ([string]::IsNullOrWhiteSpace($env:MODEL_ROOT)) {
    throw "MODEL_ROOT is required"
}
if ($StableName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "StableName must be one path-safe artifact name"
}
if ($Revision -notmatch '^[0-9a-fA-F]{40,64}$') {
    throw "Publication requires an immutable Hugging Face commit revision"
}

$selectedFiles = @($Files.Split(',') | ForEach-Object { $_.Trim() } |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if ($selectedFiles.Count -eq 0 -or
    ($selectedFiles | Sort-Object -Unique).Count -ne $selectedFiles.Count) {
    throw "Files must contain unique comma-separated repository paths"
}
foreach ($relativePath in $selectedFiles) {
    if ([System.IO.Path]::IsPathRooted($relativePath) -or
        $relativePath.Contains("..") -or
        $relativePath.IndexOfAny([char[]]'*?[]') -ge 0) {
        throw "Invalid repository file path: $relativePath"
    }
}

$snapshot = Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
$snapshotRoot = [System.IO.Path]::GetFullPath($snapshot).TrimEnd('\')
$repositoryRoot = Split-Path (Split-Path $snapshotRoot -Parent) -Parent
$treePath = Join-Path (Join-Path $repositoryRoot "trees") ($Revision + ".json")
if (-not (Test-Path -LiteralPath $treePath -PathType Leaf)) {
    throw "Pinned Hugging Face tree metadata is missing: $treePath"
}
$tree = Get-Content -LiteralPath $treePath -Raw | ConvertFrom-Json

$fileRecords = @()
[int64]$payloadBytes = 0
foreach ($relativePath in $selectedFiles) {
    $metadataProperty = $tree.files.PSObject.Properties[$relativePath]
    if ($null -eq $metadataProperty -or
        $null -eq $metadataProperty.Value.PSObject.Properties["lfs_sha256"] -or
        $null -eq $metadataProperty.Value.PSObject.Properties["lfs_size"]) {
        throw "Pinned LFS metadata is missing for $relativePath"
    }
    $sourcePath = [System.IO.Path]::GetFullPath((Join-Path $snapshotRoot $relativePath))
    if (-not $sourcePath.StartsWith($snapshotRoot + '\',
            [System.StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Selected file escapes the snapshot or is absent: $relativePath"
    }
    $expectedBytes = [int64]$metadataProperty.Value.lfs_size
    $expectedHash = ([string]$metadataProperty.Value.lfs_sha256).ToLowerInvariant()
    $actualBytes = [int64](Get-Item -LiteralPath $sourcePath).Length
    $actualHash = (Get-FileHash -LiteralPath $sourcePath `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualBytes -ne $expectedBytes -or $actualHash -ne $expectedHash) {
        throw "Selected file failed source integrity: $relativePath"
    }
    $payloadBytes += $expectedBytes
    $fileRecords += [ordered]@{
        path = $relativePath.Replace('\', '/')
        bytes = $expectedBytes
        sha256 = $expectedHash
        lfs_blob_id = [string]$metadataProperty.Value.blob_id
        xet_hash = [string]$metadataProperty.Value.xet_hash
    }
}

$modelRoot = [System.IO.Path]::GetFullPath($env:MODEL_ROOT).TrimEnd('\')
New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null
$revisionTag = $Revision.Substring(0, 12).ToLowerInvariant()
$candidate = Join-Path $modelRoot "$StableName.candidate-$revisionTag"
$destination = Join-Path $modelRoot $StableName
if (Test-Path -LiteralPath $destination) {
    throw "Stable artifact already exists; refusing to overwrite: $destination"
}
if (Test-Path -LiteralPath $candidate) {
    throw "Publication candidate already exists; inspect it before retrying: $candidate"
}

New-Item -ItemType Directory -Path $candidate | Out-Null
try {
    foreach ($record in $fileRecords) {
        $sourcePath = Join-Path $snapshotRoot ([string]$record.path)
        $candidatePath = Join-Path $candidate ([string]$record.path)
        New-Item -ItemType Directory -Path (Split-Path $candidatePath -Parent) `
            -Force | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $candidatePath
        $candidateBytes = [int64](Get-Item -LiteralPath $candidatePath).Length
        $candidateHash = (Get-FileHash -LiteralPath $candidatePath `
            -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($candidateBytes -ne [int64]$record.bytes -or
            $candidateHash -ne [string]$record.sha256) {
            throw "Candidate file failed integrity: $($record.path)"
        }
    }

    $manifest = [ordered]@{
        format = "huggingface-exact-files-v1"
        model_id = $ModelId
        revision = $Revision.ToLowerInvariant()
        payload_bytes = $payloadBytes
        files = $fileRecords
    }
    $manifestPath = Join-Path $candidate "manifest.json"
    $manifestJson = ($manifest | ConvertTo-Json -Depth 8) + "`n"
    [System.IO.File]::WriteAllText($manifestPath, $manifestJson,
        [System.Text.UTF8Encoding]::new($false))

    Rename-Item -LiteralPath $candidate -NewName $StableName
    foreach ($record in $fileRecords) {
        $publishedPath = Join-Path $destination ([string]$record.path)
        $publishedBytes = [int64](Get-Item -LiteralPath $publishedPath).Length
        $publishedHash = (Get-FileHash -LiteralPath $publishedPath `
            -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($publishedBytes -ne [int64]$record.bytes -or
            $publishedHash -ne [string]$record.sha256) {
            throw "Published file failed integrity: $($record.path)"
        }
    }
}
catch {
    if ((Test-Path -LiteralPath $destination -PathType Container) -and
        -not (Test-Path -LiteralPath $candidate)) {
        Rename-Item -LiteralPath $destination `
            -NewName (Split-Path $candidate -Leaf)
    }
    throw
}

[PSCustomObject]@{
    status = "published"
    destination = $destination
    model_id = $ModelId
    revision = $Revision.ToLowerInvariant()
    files = $fileRecords.Count
    payload_bytes = $payloadBytes
    manifest_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $destination "manifest.json") -Algorithm SHA256).Hash.ToLowerInvariant()
} | ConvertTo-Json -Depth 4
