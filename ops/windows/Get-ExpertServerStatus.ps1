param(
    [int]$Port = 8080,
    [ValidateRange(0, 3600)][int]$WaitSeconds = 0,
    [string]$ExpectedModel = "",
    [int]$ExpectedContext = 0,
    [int]$ExpectedMaximumNewTokens = 0,
    [double]$ExpectedGenerationTimeoutSeconds = 0
)

$deadline = [DateTime]::UtcNow.AddSeconds($WaitSeconds)
$ready = $false
$lastError = ""
$modelInfo = $null
do {
    try {
        $readyResponse = Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/ready" `
            -TimeoutSec 5 -ErrorAction Stop
        $ready = [bool]$readyResponse.ready
        if (-not $ready) { $lastError = "service reports ready=false" }
        if ($ready) {
            $modelInfo = Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/model-info" `
                -TimeoutSec 10 -ErrorAction Stop
            break
        }
    } catch {
        $lastError = $_.Exception.Message
    }
    if ([DateTime]::UtcNow -ge $deadline) { break }
    Start-Sleep -Milliseconds 500
} while ($true)

$tasks = foreach ($name in @(
    "QuantumLLM-DeepSeekV4Flash",
    "QuantumLLM-P6ExpertServer"
)) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($null -ne $task) {
        [PSCustomObject]@{ name = $name; state = [string]$task.State }
    }
}

$mismatches = [Collections.Generic.List[string]]::new()
if ($ready -and $ExpectedModel -and [string]$modelInfo.model -ne $ExpectedModel) {
    [void]$mismatches.Add("model expected=$ExpectedModel actual=$($modelInfo.model)")
}
if ($ready -and $ExpectedContext -gt 0 -and
    [int]$modelInfo.runtime_config.max_context -ne $ExpectedContext) {
    [void]$mismatches.Add("context expected=$ExpectedContext actual=$($modelInfo.runtime_config.max_context)")
}
if ($ready -and $ExpectedMaximumNewTokens -gt 0 -and
    [int]$modelInfo.runtime_config.maximum_new_tokens -ne $ExpectedMaximumNewTokens) {
    [void]$mismatches.Add("output expected=$ExpectedMaximumNewTokens actual=$($modelInfo.runtime_config.maximum_new_tokens)")
}
if ($ready -and $ExpectedGenerationTimeoutSeconds -gt 0 -and
    [double]$modelInfo.runtime_config.generation_timeout_seconds -ne
        $ExpectedGenerationTimeoutSeconds) {
    [void]$mismatches.Add("generation timeout expected=$ExpectedGenerationTimeoutSeconds actual=$($modelInfo.runtime_config.generation_timeout_seconds)")
}

$result = [PSCustomObject]@{
    ready = $ready
    endpoint = "http://127.0.0.1:${Port}"
    tasks = @($tasks)
    model = if ($null -ne $modelInfo) { [string]$modelInfo.model } else { $null }
    build_id = if ($null -ne $modelInfo) { [string]$modelInfo.build_id } else { $null }
    max_context = if ($null -ne $modelInfo) {
        [int]$modelInfo.runtime_config.max_context
    } else { $null }
    maximum_new_tokens = if ($null -ne $modelInfo) {
        [int]$modelInfo.runtime_config.maximum_new_tokens
    } else { $null }
    generation_timeout_seconds = if ($null -ne $modelInfo) {
        [double]$modelInfo.runtime_config.generation_timeout_seconds
    } else { $null }
    worker_capacity = if ($null -ne $modelInfo) {
        [int]$modelInfo.worker_capacity
    } else { $null }
    active_requests = if ($null -ne $modelInfo) {
        [int]$modelInfo.active_requests
    } else { $null }
    mismatches = @($mismatches)
    last_error = if ($ready) { "" } else { $lastError }
}
$result | ConvertTo-Json -Depth 5

if (($WaitSeconds -gt 0 -and -not $ready) -or $mismatches.Count -gt 0) {
    exit 1
}
