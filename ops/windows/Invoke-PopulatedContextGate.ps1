param(
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$PromptTokens,
    [Parameter(Mandatory = $true)][string]$Output,
    [ValidateRange(1, 4294967295)][uint32]$PromptLimit,
    [ValidateSet("prefix", "suffix")][string]$PromptWindow = "prefix",
    [ValidateRange(1, 4294967295)][uint32]$MaxTokens = 128,
    [ValidateRange(0.0, 2.0)][double]$Temperature = 1.0,
    [ValidateRange(0.000001, 1.0)][double]$TopP = 0.95,
    [ValidateRange(0, 4294967295)][uint32]$TopK = 20,
    [ValidateRange(0, 4294967295)][uint32]$Seed = 1,
    [ValidateRange(1.0, 86400.0)][double]$TimeoutSeconds = 14400.0,
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [string]$ApiKey = ""
)

. (Join-Path $PSScriptRoot "Common.ps1")

$python = Get-PythonCommand
$gate = Join-Path $script:RepoRoot "ops\python\populated_context_gate.py"
$resolvedTokens = [System.IO.Path]::GetFullPath($PromptTokens)
$resolvedOutput = [System.IO.Path]::GetFullPath($Output)
if (-not (Test-Path -LiteralPath $resolvedTokens -PathType Leaf)) {
    throw "Prompt token file is missing: $resolvedTokens"
}

$arguments = @(
    $gate,
    "--base-url", $BaseUrl,
    "--model", $Model,
    "--prompt-tokens", $resolvedTokens,
    "--prompt-limit", [string]$PromptLimit,
    "--prompt-window", $PromptWindow,
    "--max-tokens", [string]$MaxTokens,
    "--temperature", [string]::Format(
        [System.Globalization.CultureInfo]::InvariantCulture,
        "{0:R}", $Temperature
    ),
    "--top-p", [string]::Format(
        [System.Globalization.CultureInfo]::InvariantCulture,
        "{0:R}", $TopP
    ),
    "--top-k", [string]$TopK,
    "--seed", [string]$Seed,
    "--timeout-seconds", [string]::Format(
        [System.Globalization.CultureInfo]::InvariantCulture,
        "{0:R}", $TimeoutSeconds
    ),
    "--output", $resolvedOutput
)

Push-Location $script:RepoRoot
$previousApiKey = $env:EXPERT_API_KEY
try {
    if (-not [string]::IsNullOrEmpty($ApiKey)) {
        $env:EXPERT_API_KEY = $ApiKey
    }
    & $python.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Populated-context gate failed with exit code $LASTEXITCODE"
    }
}
finally {
    if ($null -eq $previousApiKey) {
        Remove-Item Env:EXPERT_API_KEY -ErrorAction SilentlyContinue
    }
    else {
        $env:EXPERT_API_KEY = $previousApiKey
    }
    Pop-Location
}
