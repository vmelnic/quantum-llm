param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [string]$Snapshot,
    [switch]$TensorGroups
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
    & $python.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Source inventory exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
