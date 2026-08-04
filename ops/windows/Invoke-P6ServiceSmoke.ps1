param(
    [string]$BaseUri = "http://127.0.0.1:8080",
    [string]$ExpectedModel = "qwen3-next-80b-a3b-expert-pack-int8",
    [string]$ExpectedBuildId = "",
    [int]$Concurrency = 4,
    [int]$NewTokens = 4,
    [int]$TimeoutSeconds = 600
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories
Add-Type -AssemblyName System.Net.Http

if ($Concurrency -lt 2 -or $NewTokens -lt 2 -or $TimeoutSeconds -lt 1) {
    throw "Invalid P6 service smoke settings"
}

$headers = @{}
if ($env:EXPERT_API_KEY) {
    $headers.Authorization = "Bearer $($env:EXPERT_API_KEY)"
}
function Invoke-JsonGet {
    param([string]$Path)
    Invoke-RestMethod -Uri "$BaseUri$Path" -Headers $headers -TimeoutSec $TimeoutSeconds
}

function Get-MetricsText {
    (Invoke-WebRequest -Uri "$BaseUri/metrics" -Headers $headers `
        -UseBasicParsing -TimeoutSec $TimeoutSeconds).Content
}

function Get-MetricValue {
    param([string]$Text, [string]$Name)
    $match = [regex]::Match($Text, "(?m)^$([regex]::Escape($Name))\s+([0-9.eE+-]+)$")
    if (-not $match.Success) { throw "Metric is missing: $Name" }
    return [double]::Parse($match.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
}

function New-CompletionBody {
    param([int]$TokenCount, [bool]$Stream)
    @{
        model = $ExpectedModel
        prompt = @(151644, 872, 374)
        max_tokens = $TokenCount
        stream = $Stream
        temperature = 0
    } | ConvertTo-Json -Compress
}

$health = Invoke-JsonGet -Path "/health"
$ready = Invoke-JsonGet -Path "/ready"
$info = Invoke-JsonGet -Path "/model-info"
$models = Invoke-JsonGet -Path "/v1/models"
if ($health.status -ne "ok" -or -not $ready.ready) { throw "Service is not ready" }
if ($info.model -ne $ExpectedModel -or $models.data[0].id -ne $ExpectedModel) {
    throw "Unexpected deployed model identity"
}
if ($ExpectedBuildId -and $info.build_id -ne $ExpectedBuildId) {
    throw "Unexpected build id: $($info.build_id)"
}
if (-not $info.manifest_content_sha256 -or -not $info.experts_index_sha256) {
    throw "Service does not publish the deployed container identity"
}
if ([int]$info.worker_capacity -lt $Concurrency) {
    throw "Worker capacity $($info.worker_capacity) is below requested concurrency $Concurrency"
}

$before = Get-MetricsText
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
if ($env:EXPERT_API_KEY) {
    $client.DefaultRequestHeaders.Authorization = `
        [System.Net.Http.Headers.AuthenticationHeaderValue]::new("Bearer", $env:EXPERT_API_KEY)
}
try {
    $tasks = @()
    $contents = @()
    foreach ($index in 1..$Concurrency) {
        $content = [System.Net.Http.StringContent]::new(
            (New-CompletionBody -TokenCount $NewTokens -Stream $false),
            [Text.Encoding]::UTF8, "application/json")
        $contents += $content
        $tasks += $client.PostAsync("$BaseUri/v1/completions", $content)
    }
    [Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]$tasks)
    $responses = @()
    foreach ($task in $tasks) {
        $response = $task.GetAwaiter().GetResult()
        $text = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        if (-not $response.IsSuccessStatusCode) {
            throw "Concurrent completion failed with $([int]$response.StatusCode): $text"
        }
        $payload = $text | ConvertFrom-Json
        if ($payload.model -ne $ExpectedModel -or $payload.choices.Count -ne 1) {
            throw "Concurrent completion returned an invalid payload"
        }
        $responses += $payload
        $response.Dispose()
    }
    $concurrentCompletionTokens = [int](($responses | ForEach-Object {
        [int]$_.usage.completion_tokens
    } | Measure-Object -Sum).Sum)
    if ($concurrentCompletionTokens -lt $Concurrency) {
        throw "Concurrent requests returned no generated token"
    }
    foreach ($content in $contents) { $content.Dispose() }

    $streamContent = [System.Net.Http.StringContent]::new(
        (New-CompletionBody -TokenCount $NewTokens -Stream $true),
        [Text.Encoding]::UTF8, "application/json")
    $streamResponse = $client.PostAsync(
        "$BaseUri/v1/completions", $streamContent).GetAwaiter().GetResult()
    $streamText = $streamResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if (-not $streamResponse.IsSuccessStatusCode -or $streamText -notmatch "data: \[DONE\]") {
        throw "Streaming completion did not terminate correctly"
    }
    $streamResponse.Dispose()
    $streamContent.Dispose()

    $cancelBefore = Get-MetricValue -Text (Get-MetricsText) `
        -Name "expert_service_cancelled_total"
    $cancelRequest = [System.Net.Http.HttpRequestMessage]::new(
        [System.Net.Http.HttpMethod]::Post, "$BaseUri/v1/completions")
    $cancelRequest.Headers.ConnectionClose = $true
    $cancelRequest.Content = [System.Net.Http.StringContent]::new(
        (New-CompletionBody -TokenCount 64 -Stream $true),
        [Text.Encoding]::UTF8, "application/json")
    $cancelResponse = $client.SendAsync(
        $cancelRequest, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead
    ).GetAwaiter().GetResult()
    if (-not $cancelResponse.IsSuccessStatusCode) { throw "Cancellation probe was rejected" }
    $cancelResponse.Dispose()
    $cancelRequest.Dispose()

    $deadline = [DateTime]::UtcNow.AddSeconds([Math]::Min($TimeoutSeconds, 120))
    $cancelAfter = $cancelBefore
    while ([DateTime]::UtcNow -lt $deadline -and $cancelAfter -le $cancelBefore) {
        Start-Sleep -Milliseconds 250
        $cancelAfter = Get-MetricValue -Text (Get-MetricsText) `
            -Name "expert_service_cancelled_total"
    }
    if ($cancelAfter -le $cancelBefore) {
        throw "Service did not observe the disconnected streaming client"
    }
}
finally {
    $client.Dispose()
}

$after = Get-MetricsText
$batches = (Get-MetricValue -Text $after -Name "expert_service_decode_batches_total") - `
    (Get-MetricValue -Text $before -Name "expert_service_decode_batches_total")
$rows = (Get-MetricValue -Text $after -Name "expert_service_decode_rows_total") - `
    (Get-MetricValue -Text $before -Name "expert_service_decode_rows_total")
$completed = (Get-MetricValue -Text $after -Name "expert_service_completed_total") - `
    (Get-MetricValue -Text $before -Name "expert_service_completed_total")
if ($completed -lt ($Concurrency + 1) -or $rows -lt $concurrentCompletionTokens -or $batches -le 0) {
    throw "Persistent service counters do not cover the completed smoke requests"
}

$result = [PSCustomObject]@{
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    endpoint = $BaseUri
    model = $info.model
    build_id = $info.build_id
    manifest_content_sha256 = $info.manifest_content_sha256
    experts_index_sha256 = $info.experts_index_sha256
    worker_protocol = $info.worker_protocol
    worker_capacity = $info.worker_capacity
    concurrency = $Concurrency
    new_tokens = $NewTokens
    completed_delta = $completed
    decode_batches_delta = $batches
    decode_rows_delta = $rows
    effective_decode_batch = $rows / $batches
    cancellation_observed = $true
    ttft_p95_seconds = Get-MetricValue -Text $after `
        -Name "expert_service_ttft_seconds_p95"
    inter_token_p95_seconds = Get-MetricValue -Text $after `
        -Name "expert_service_inter_token_seconds_p95"
}
$path = Write-JsonArtifact -Value $result -Name "p6-service-smoke-latest.json"
$result | ConvertTo-Json -Depth 5
Write-Output "P6 service smoke artifact: $path"
