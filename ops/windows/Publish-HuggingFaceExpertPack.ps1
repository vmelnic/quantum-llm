param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [Parameter(Mandatory = $true)][string]$StableName,
    [Parameter(Mandatory = $true)][string]$Adapter,
    [string]$ConfigFile = "config.json",
    [string]$IndexFile = "model.safetensors.index.json",
    [ValidateSet("int8-symmetric-per-row-v1", "fp4-e2m1-ue8m0-block32-v1", "fp4-e2m1-ue8m0-block32-mse-v2", "fp4-e2m1-ue8m0-block32-activation-aware-v3", "fp4-e2m1-ue8m0-block32-activation-codes-v4", "nvfp4-e2m1-e4m3fn-block16-w4a4-v1")]
    [string]$QuantProfile = "fp4-e2m1-ue8m0-block32-v1",
    [string]$ActivationCalibration = "",
    [string]$DenseEncodingPolicy = "",
    [ValidateSet("q8", "bf16")]
    [string]$DenseActivationInput = "q8",
    [string]$SamplingProfiles = "",
    [int64]$MaxExpertPackBytes = 4GB,
    [int]$QualitySamplesPerTensor = 8,
    [double]$MaximumRelativeL2 = 0.20,
    [double]$MinimumCosine = 0.98,
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
if ($QualitySamplesPerTensor -lt 3 -or $MaximumRelativeL2 -le 0 -or
    $MinimumCosine -lt -1 -or $MinimumCosine -gt 1) {
    throw "Invalid numerical quality parameters"
}

$snapshot = Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
$indexPath = Join-Path $snapshot $IndexFile
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
    -ModelId $ModelId -Revision $Revision -Snapshot $snapshot -Adapter $Adapter `
    -ConfigFile $ConfigFile -IndexFile $IndexFile

if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
    $arguments = @{
        Action = "Compile"
        Path = $snapshot
        Output = $candidate
        SourceId = $ModelId
        SourceRevision = $Revision
        Adapter = $Adapter
        ConfigFile = $ConfigFile
        IndexFile = $IndexFile
        QuantProfile = $QuantProfile
        DenseActivationInput = $DenseActivationInput
        MaxExpertPackBytes = $MaxExpertPackBytes
        Resume = $true
    }
    if ($ReclaimSourceShards) { $arguments.ReclaimSourceShards = $true }
    if (-not [string]::IsNullOrWhiteSpace($SamplingProfiles)) {
        $arguments.SamplingProfiles = $SamplingProfiles
    }
    if (-not [string]::IsNullOrWhiteSpace($ActivationCalibration)) {
        $arguments.ActivationCalibration = $ActivationCalibration
    }
    if (-not [string]::IsNullOrWhiteSpace($DenseEncodingPolicy)) {
        $arguments.DenseEncodingPolicy = $DenseEncodingPolicy
    }
    & (Join-Path $PSScriptRoot "Invoke-ExpertPack.ps1") @arguments
}

& (Join-Path $PSScriptRoot "Invoke-ExpertPack.ps1") `
    -Action Validate -Path $candidate
if ($QuantProfile -in @(
        "fp4-e2m1-ue8m0-block32-v1",
        "fp4-e2m1-ue8m0-block32-mse-v2",
        "fp4-e2m1-ue8m0-block32-activation-aware-v3",
        "fp4-e2m1-ue8m0-block32-activation-codes-v4"
    )) {
    & (Join-Path $PSScriptRoot "Invoke-Fp4SourceQualityGate.ps1") `
        -ModelId $ModelId -Revision $Revision `
        -ArtifactName ([System.IO.Path]::GetFileName($candidate)) `
        -SamplesPerTensor $QualitySamplesPerTensor `
        -MaximumRelativeL2 $MaximumRelativeL2 `
        -MinimumCosine $MinimumCosine
}
& (Join-Path $PSScriptRoot "Promote-ModelArtifact.ps1") `
    -Candidate $candidate -Destination $destination -ExpectedProgramSchema 3
