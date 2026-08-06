param(
    [Parameter(Mandatory = $true)][string]$Container,
    [Parameter(Mandatory = $true)][string]$PromptFile,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [string]$Layers = "20,21,22,23",
    [int64]$RamGiB = 48,
    [int64]$VramGiB = 14,
    [int]$ChunkTokens = 4,
    [int64]$MaximumRecords = 0,
    [switch]$ReplaceOutput
)

. (Join-Path $PSScriptRoot "Common.ps1")

$runner = Join-Path $script:RepoRoot `
    "out\build\windows-msvc-release\runtime\Release\expert-qwen3-next-runner.exe"
foreach ($path in @($runner, $Container, $PromptFile)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path is missing: $path" }
}
if ($RamGiB -le 0 -or $VramGiB -le 0 -or $ChunkTokens -le 0 -or
    $MaximumRecords -lt 0) {
    throw "Invalid Neural CPU trace resource limit"
}
if (Test-Path -LiteralPath $OutputDirectory) {
    if (-not $ReplaceOutput) {
        throw "Trace output already exists: $OutputDirectory"
    }
    Remove-Item -LiteralPath $OutputDirectory -Recurse -Force
}

$arguments = @(
    $Container, "--trace-moe", $PromptFile, $OutputDirectory, $Layers,
    [string]$RamGiB, [string]$VramGiB, [string]$ChunkTokens,
    [string]$MaximumRecords
)
& $runner @arguments
if ($LASTEXITCODE -ne 0) { throw "MoE trace failed with exit code $LASTEXITCODE" }
