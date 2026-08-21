param(
    [int]$Port = 8080,
    [string]$ApiKey = "",
    [ValidateRange(0, 3600)][int]$WaitSeconds = 0,
    [string]$ExpectedModel = "",
    [string]$ExpectedTaskName = "",
    [int]$ExpectedContext = 0,
    [int]$ExpectedMaximumNewTokens = 0,
    [double]$ExpectedGenerationTimeoutSeconds = 0,
    [int]$ExpectedMaximumBodyMiB = 0
)

$deadline = [DateTime]::UtcNow.AddSeconds($WaitSeconds)
$ready = $false
$lastError = ""
$modelInfo = $null
$expectedTaskName = if ($ExpectedTaskName) {
    $ExpectedTaskName
} elseif ($ExpectedModel) { "QuantumLLM-ExpertVm" } else { "" }
$waitStarted = [DateTime]::Now
$headers = @{}
if ($ApiKey) { $headers.Authorization = "Bearer $ApiKey" }
do {
    try {
        $readyResponse = Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/ready" `
            -Headers $headers -TimeoutSec 5 -ErrorAction Stop
        $ready = [bool]$readyResponse.ready
        if (-not $ready) { $lastError = "service reports ready=false" }
        if ($ready) {
            $modelInfo = Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/model-info" `
                -Headers $headers -TimeoutSec 10 -ErrorAction Stop
            break
        }
    } catch {
        $lastError = $_.Exception.Message
    }
    if (-not $ready -and $expectedTaskName -and
        [DateTime]::Now -ge $waitStarted.AddSeconds(1)) {
        $expectedTask = Get-ScheduledTask -TaskName $expectedTaskName `
            -ErrorAction SilentlyContinue
        if ($null -ne $expectedTask -and
            [string]$expectedTask.State -ne "Running") {
            $expectedInfo = Get-ScheduledTaskInfo -TaskName $expectedTaskName
            if ($expectedInfo.LastRunTime -ge $waitStarted.AddMinutes(-1) -and
                $expectedInfo.LastTaskResult -ne 0) {
                $lastError = "scheduled task $expectedTaskName exited with result $($expectedInfo.LastTaskResult)"
                break
            }
        }
    }
    if ([DateTime]::UtcNow -ge $deadline) { break }
    Start-Sleep -Milliseconds 500
} while ($true)

$tasks = foreach ($name in @("QuantumLLM-ExpertVm")) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($null -ne $task) {
        $taskInfo = Get-ScheduledTaskInfo -TaskName $name
        [PSCustomObject]@{
            name = $name
            state = [string]$task.State
            last_run = $taskInfo.LastRunTime
            last_result = $taskInfo.LastTaskResult
        }
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
if ($ready -and $ExpectedMaximumBodyMiB -gt 0 -and
    [int64]$modelInfo.runtime_config.maximum_body_bytes -ne
        [int64]$ExpectedMaximumBodyMiB * 1MB) {
    [void]$mismatches.Add("maximum body expected=$ExpectedMaximumBodyMiB MiB actual=$($modelInfo.runtime_config.maximum_body_bytes) bytes")
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
    maximum_body_bytes = if ($null -ne $modelInfo) {
        [int64]$modelInfo.runtime_config.maximum_body_bytes
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
