param(
    [string]$BaseUri = "http://127.0.0.1:8080",
    [string]$Model = "qwen3-next-80b-a3b-expert-pack-int8",
    [string]$Prompt = "Explain in two sentences why mixture-of-experts models can exceed local GPU memory.",
    [int]$MaxTokens = 32,
    [int]$Rounds = 2
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories
if ($MaxTokens -lt 1 -or $Rounds -lt 1) { throw "Invalid probe settings" }
$python = Get-PythonCommand
$probe = Join-Path $script:RepoRoot "ops\python\probe_openai_api.py"
$line = & $python.Source $probe --base-url $BaseUri --model $Model `
    --prompt $Prompt --max-tokens $MaxTokens --rounds $Rounds
if ($LASTEXITCODE -ne 0) { throw "API text probe failed with $LASTEXITCODE" }
$result = $line | ConvertFrom-Json -ErrorAction Stop
$path = Write-JsonArtifact -Value $result -Name "p6-text-probe-latest.json"
$result | ConvertTo-Json -Depth 10
Write-Output "P6 text probe artifact: $path"
