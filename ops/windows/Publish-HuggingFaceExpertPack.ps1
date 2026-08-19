param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [Parameter(Mandatory = $true)][string]$StableName,
    [Parameter(Mandatory = $true)][string]$Adapter,
    [ValidateSet("int8-symmetric-per-row-v1", "fp4-e2m1-ue8m0-block32-v1")]
    [string]$QuantProfile = "fp4-e2m1-ue8m0-block32-v1",
    [int64]$MaxExpertPackBytes = 4GB,
    [switch]$ReclaimSourceShards
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

$snapshot = Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
$indexPath = Join-Path $snapshot "model.safetensors.index.json"
$index = Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json
$referenced = @($index.weight_map.PSObject.Properties.Value | Sort-Object -Unique)
$missing = @($referenced | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $snapshot ([string]$_)) -PathType Leaf)
})
$incomplete = @(Get-ChildItem -LiteralPath $snapshot -Filter "*.incomplete" `
    -File -ErrorAction SilentlyContinue)
if ($missing.Count -ne 0 -or $incomplete.Count -ne 0) {
    throw "Hugging Face snapshot is incomplete: missing=$($missing.Count), incomplete=$($incomplete.Count)"
}

$modelRoot = [System.IO.Path]::GetFullPath($env:MODEL_ROOT).TrimEnd('\')
New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null
$revisionTag = $Revision.Substring(0, 12).ToLowerInvariant()
$candidate = Join-Path $modelRoot "$StableName.candidate-$revisionTag"
$destination = Join-Path $modelRoot $StableName

& (Join-Path $PSScriptRoot "Invoke-SourceInventory.ps1") `
    -ModelId $ModelId -Revision $Revision -Snapshot $snapshot -Adapter $Adapter

if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
    $arguments = @{
        Action = "Compile"
        Path = $snapshot
        Output = $candidate
        SourceId = $ModelId
        SourceRevision = $Revision
        Adapter = $Adapter
        QuantProfile = $QuantProfile
        MaxExpertPackBytes = $MaxExpertPackBytes
        Resume = $true
    }
    if ($ReclaimSourceShards) { $arguments.ReclaimSourceShards = $true }
    & (Join-Path $PSScriptRoot "Invoke-ExpertPack.ps1") @arguments
}

& (Join-Path $PSScriptRoot "Invoke-ExpertPack.ps1") `
    -Action Validate -Path $candidate
& (Join-Path $PSScriptRoot "Promote-ModelArtifact.ps1") `
    -Candidate $candidate -Destination $destination -ExpectedProgramSchema 3
