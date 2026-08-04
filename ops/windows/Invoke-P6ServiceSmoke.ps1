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
    $body = @{
        model = $ExpectedModel
        prompt = @(151644, 872, 374)
        max_tokens = $TokenCount
        stream = $Stream
        temperature = 0
    }
    if ($Stream) { $body.stream_options = @{ include_usage = $true } }
    $body | ConvertTo-Json -Compress
}

$health = Invoke-JsonGet -Path "/health"
$ready = Invoke-JsonGet -Path "/ready"
$info = Invoke-JsonGet -Path "/model-info"
$models = Invoke-JsonGet -Path "/v1/models"
$model = Invoke-JsonGet -Path "/v1/models/$ExpectedModel"
if ($health.status -ne "ok" -or -not $ready.ready) { throw "Service is not ready" }
if ($info.model -ne $ExpectedModel -or $models.data[0].id -ne $ExpectedModel -or
    $model.id -ne $ExpectedModel -or $model.object -ne "model") {
    throw "Unexpected deployed model identity"
}
if ($ExpectedBuildId -and $info.build_id -ne $ExpectedBuildId) {
    throw "Unexpected build id: $($info.build_id)"
}
if (-not $info.manifest_content_sha256 -or -not $info.experts_index_sha256) {
    throw "Service does not publish the deployed container identity"
}
if ([int]$info.worker_protocol -lt 3 -or
    $info.worker_kv.dtype -ne "fp16" -or
    [int]$info.worker_kv.page_tokens -lt 1 -or
    [int64]$info.worker_kv.page_bytes -lt 1 -or
    [int]$info.worker_kv.page_capacity -lt $Concurrency) {
    throw "Service does not publish a usable protocol-v3 paged KV contract"
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
    if (-not $streamResponse.IsSuccessStatusCode -or
        $streamText -notmatch "data: \[DONE\]" -or
        $streamText -notmatch '"choices":\[\],"usage":') {
        throw "Streaming completion did not terminate correctly"
    }
    $streamResponse.Dispose()
    $streamContent.Dispose()

    $responsesBody = @{
        model = $ExpectedModel
        instructions = "Answer briefly."
        input = @(@{
            role = "user"
            content = @(@{ type = "input_text"; text = "Say hello." })
        })
        max_output_tokens = $NewTokens
        temperature = 0
    } | ConvertTo-Json -Depth 6 -Compress
    $responsesContent = [System.Net.Http.StringContent]::new(
        $responsesBody, [Text.Encoding]::UTF8, "application/json")
    $responsesHttp = $client.PostAsync(
        "$BaseUri/v1/responses", $responsesContent).GetAwaiter().GetResult()
    $responsesText = $responsesHttp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if (-not $responsesHttp.IsSuccessStatusCode) {
        throw "Responses API failed with $([int]$responsesHttp.StatusCode): $responsesText"
    }
    $responsesPayload = $responsesText | ConvertFrom-Json
    if ($responsesPayload.object -ne "response" -or
        $responsesPayload.status -ne "completed" -or
        $responsesPayload.output[0].content[0].type -ne "output_text" -or
        [int]$responsesPayload.usage.output_tokens -lt 1) {
        throw "Responses API returned an invalid payload"
    }
    $responsesHttp.Dispose()
    $responsesContent.Dispose()

    $responsesStreamContent = [System.Net.Http.StringContent]::new(
        (@{
            model = $ExpectedModel
            input = "Say hello."
            max_output_tokens = $NewTokens
            temperature = 0
            stream = $true
        } | ConvertTo-Json -Compress),
        [Text.Encoding]::UTF8, "application/json")
    $responsesStreamHttp = $client.PostAsync(
        "$BaseUri/v1/responses", $responsesStreamContent).GetAwaiter().GetResult()
    $responsesStreamText = $responsesStreamHttp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if (-not $responsesStreamHttp.IsSuccessStatusCode -or
        $responsesStreamText -notmatch '"type":"response.created"' -or
        $responsesStreamText -notmatch '"type":"response.output_text.delta"' -or
        $responsesStreamText -notmatch '"type":"response.completed"') {
        throw "Responses API stream did not emit its typed lifecycle"
    }
    $responsesStreamHttp.Dispose()
    $responsesStreamContent.Dispose()

    $chatBody = @{
        model = $ExpectedModel
        messages = @(@{
            role = "user"
            content = @(@{ type = "text"; text = "Say hello." })
        })
        max_completion_tokens = $NewTokens
        temperature = 0
    } | ConvertTo-Json -Depth 6 -Compress
    $chatContent = [System.Net.Http.StringContent]::new(
        $chatBody, [Text.Encoding]::UTF8, "application/json")
    $chatHttp = $client.PostAsync(
        "$BaseUri/v1/chat/completions", $chatContent).GetAwaiter().GetResult()
    $chatText = $chatHttp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if (-not $chatHttp.IsSuccessStatusCode) {
        throw "Chat Completions failed with $([int]$chatHttp.StatusCode): $chatText"
    }
    $chatPayload = $chatText | ConvertFrom-Json
    if ($chatPayload.object -ne "chat.completion" -or
        $chatPayload.choices[0].message.role -ne "assistant" -or
        [int]$chatPayload.usage.completion_tokens -lt 1) {
        throw "Chat Completions returned an invalid payload"
    }
    $chatHttp.Dispose()
    $chatContent.Dispose()

    $unsupportedContent = [System.Net.Http.StringContent]::new(
        (@{ model = $ExpectedModel; input = "hello"; temperature = 0.5 } |
            ConvertTo-Json -Compress),
        [Text.Encoding]::UTF8, "application/json")
    $unsupportedHttp = $client.PostAsync(
        "$BaseUri/v1/responses", $unsupportedContent).GetAwaiter().GetResult()
    $unsupportedText = $unsupportedHttp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    $unsupportedPayload = $unsupportedText | ConvertFrom-Json
    if ([int]$unsupportedHttp.StatusCode -ne 400 -or
        $unsupportedPayload.error.param -ne "temperature" -or
        $unsupportedPayload.error.code -ne "unsupported_value") {
        throw "Unsupported capability did not return a precise OpenAI error"
    }
    $unsupportedHttp.Dispose()
    $unsupportedContent.Dispose()

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
$afterInfo = Invoke-JsonGet -Path "/model-info"
$batches = (Get-MetricValue -Text $after -Name "expert_service_decode_batches_total") - `
    (Get-MetricValue -Text $before -Name "expert_service_decode_batches_total")
$rows = (Get-MetricValue -Text $after -Name "expert_service_decode_rows_total") - `
    (Get-MetricValue -Text $before -Name "expert_service_decode_rows_total")
$completed = (Get-MetricValue -Text $after -Name "expert_service_completed_total") - `
    (Get-MetricValue -Text $before -Name "expert_service_completed_total")
if ($completed -lt ($Concurrency + 4) -or $rows -lt $concurrentCompletionTokens -or $batches -le 0) {
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
    kv_dtype = $afterInfo.worker_kv.dtype
    kv_page_tokens = $afterInfo.worker_kv.page_tokens
    kv_page_bytes = $afterInfo.worker_kv.page_bytes
    kv_page_capacity = $afterInfo.worker_kv.page_capacity
    kv_allocated_pages = $afterInfo.worker_kv.allocated_pages
    kv_reserved_pages = $afterInfo.worker_kv.reserved_pages
    concurrency = $Concurrency
    new_tokens = $NewTokens
    completed_delta = $completed
    decode_batches_delta = $batches
    decode_rows_delta = $rows
    effective_decode_batch = $rows / $batches
    cancellation_observed = $true
    responses_api_observed = $true
    responses_stream_observed = $true
    chat_completions_observed = $true
    streaming_usage_observed = $true
    precise_unsupported_error_observed = $true
    ttft_p95_seconds = Get-MetricValue -Text $after `
        -Name "expert_service_ttft_seconds_p95"
    inter_token_p95_seconds = Get-MetricValue -Text $after `
        -Name "expert_service_inter_token_seconds_p95"
}
$path = Write-JsonArtifact -Value $result -Name "p6-service-smoke-latest.json"
$result | ConvertTo-Json -Depth 5
Write-Output "P6 service smoke artifact: $path"
