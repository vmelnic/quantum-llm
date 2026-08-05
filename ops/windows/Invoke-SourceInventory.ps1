param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$Revision,
    [string]$Snapshot
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
    & $python.Source -m compiler inspect-source --source $source
    if ($LASTEXITCODE -ne 0) {
        throw "Source inventory exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
