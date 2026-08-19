param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [string]$Snapshot,
    [switch]$TensorGroups,
    [string]$Adapter,
    [ValidateSet("deepseek_v4")][string]$Contract,
    [switch]$EstimateRepresentations
)

. (Join-Path $PSScriptRoot "Common.ps1")

$source = if ($Snapshot) {
    [System.IO.Path]::GetFullPath($Snapshot)
} else {
    Resolve-HuggingFaceSnapshot -ModelId $ModelId -Revision $Revision
}
$python = Get-PythonCommand

Push-Location $script:RepoRoot
try {
    $arguments = @("-m", "compiler", "inspect-source", "--source", $source)
    if ($TensorGroups) { $arguments += "--tensor-groups" }
    if ($Adapter) { $arguments += @("--adapter", $Adapter) }
    if ($Contract) { $arguments += @("--contract", $Contract) }
    if ($EstimateRepresentations) { $arguments += "--estimate-representations" }
    & $python.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Source inventory exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
