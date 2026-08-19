param(
    [Parameter(Mandatory = $true)][string]$Candidate,
    [Parameter(Mandatory = $true)][string]$Destination,
    [ValidateRange(1, 4294967295)][uint32]$ExpectedProgramSchema = 3
)

. (Join-Path $PSScriptRoot "Common.ps1")

function Get-ValidatedProgramIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][uint32]$Schema
    )

    if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
        throw "Model artifact directory is missing: $Root"
    }
    $manifestPath = Join-Path $Root "manifest.json"
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
    $rootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    $programPath = [System.IO.Path]::GetFullPath((Join-Path $rootPath $relative))
    if (-not $programPath.StartsWith($rootPath + '\',
            [System.StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $programPath -PathType Leaf)) {
        throw "Model artifact program path escapes the artifact or is absent"
    }
    $actualBytes = (Get-Item -LiteralPath $programPath).Length
    $actualHash = (Get-FileHash -LiteralPath $programPath `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualBytes -ne [int64]$manifest.model_program.bytes -or
        $actualHash -ne [string]$manifest.model_program.sha256) {
        throw "Model artifact program does not match its authenticated manifest"
    }
    $header = @(Get-Content -LiteralPath $programPath -TotalCount 2)
    if ($header.Count -ne 2 -or $header[0] -ne "expert-runtime-model-v1") {
        throw "Model artifact program header is invalid"
    }
    $model = @($header[1] -split "`t")
    [uint32]$programSchema = 0
    if ($model.Count -lt 3 -or $model[0] -ne "model" -or
        -not [uint32]::TryParse($model[1], [ref]$programSchema) -or
        $programSchema -ne $Schema) {
        throw "Model artifact program schema is not $Schema"
    }
    return [PSCustomObject]@{
        program_schema = $programSchema
        program_bytes = $actualBytes
        program_sha256 = $actualHash
        manifest_sha256 = (Get-FileHash -LiteralPath $manifestPath `
            -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$candidatePath = [System.IO.Path]::GetFullPath($Candidate).TrimEnd('\')
$destinationPath = [System.IO.Path]::GetFullPath($Destination).TrimEnd('\')
if ($candidatePath -eq $destinationPath) {
    throw "Candidate and destination must be different directories"
}
$candidateParent = Split-Path $candidatePath -Parent
$destinationParent = Split-Path $destinationPath -Parent
if (-not $candidateParent.Equals($destinationParent,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Transactional artifact promotion requires one parent directory"
}
$validated = Get-ValidatedProgramIdentity -Root $candidatePath `
    -Schema $ExpectedProgramSchema
$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
$rollbackPath = "$destinationPath.rollback-$stamp"
if (Test-Path -LiteralPath $rollbackPath) {
    throw "Rollback destination already exists: $rollbackPath"
}

$candidateLeaf = Split-Path $candidatePath -Leaf
$destinationLeaf = Split-Path $destinationPath -Leaf
$rollbackLeaf = Split-Path $rollbackPath -Leaf
$hadDestination = Test-Path -LiteralPath $destinationPath -PathType Container
if ($hadDestination) {
    Rename-Item -LiteralPath $destinationPath -NewName $rollbackLeaf
}
try {
    Rename-Item -LiteralPath $candidatePath -NewName $destinationLeaf
    $published = Get-ValidatedProgramIdentity -Root $destinationPath `
        -Schema $ExpectedProgramSchema
    if ($published.program_bytes -ne $validated.program_bytes -or
        $published.program_sha256 -ne $validated.program_sha256 -or
        $published.manifest_sha256 -ne $validated.manifest_sha256) {
        throw "Published artifact identity changed during promotion"
    }
}
catch {
    if ((Test-Path -LiteralPath $destinationPath -PathType Container) -and
        -not (Test-Path -LiteralPath $candidatePath)) {
        Rename-Item -LiteralPath $destinationPath -NewName $candidateLeaf
    }
    if ($hadDestination -and
        (Test-Path -LiteralPath $rollbackPath -PathType Container) -and
        -not (Test-Path -LiteralPath $destinationPath)) {
        Rename-Item -LiteralPath $rollbackPath -NewName $destinationLeaf
    }
    throw
}

[PSCustomObject]@{
    status = "published"
    destination = $destinationPath
    rollback = if ($hadDestination) { $rollbackPath } else { $null }
    replaced_existing = $hadDestination
    program_schema = $published.program_schema
    program_bytes = $published.program_bytes
    program_sha256 = $published.program_sha256
    manifest_sha256 = $published.manifest_sha256
} | ConvertTo-Json
