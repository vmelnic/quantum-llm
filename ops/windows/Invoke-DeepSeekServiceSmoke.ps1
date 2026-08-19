param(
    [string]$BaseUri = "http://127.0.0.1:8080",
    [string]$ExpectedModel = "deepseek-v4-flash",
    [string]$ExpectedBuildId = "",
    [ValidateSet("latency", "balanced", "capacity")]
    [string]$ExpectedPlacementProfile = "balanced",
    [ValidateRange(2, 8)][int]$NewTokens = 2,
    [int]$TimeoutSeconds = 600,
    [switch]$AllowSourceExtents
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

if ($TimeoutSeconds -lt 1) { throw "Invalid DeepSeek service timeout" }
$headers = @{}
if ($env:EXPERT_API_KEY) {
    $headers.Authorization = "Bearer $($env:EXPERT_API_KEY)"
}
function Invoke-JsonGet {
    param([string]$Path)
    Invoke-RestMethod -Uri "$BaseUri$Path" -Headers $headers `
        -TimeoutSec $TimeoutSeconds
}
function Get-MetricsText {
    (Invoke-WebRequest -Uri "$BaseUri/metrics" -Headers $headers `
        -UseBasicParsing -TimeoutSec $TimeoutSeconds).Content
}
function Invoke-JsonPost {
    param([string]$Path, [hashtable]$Payload)
    Invoke-RestMethod -Uri "$BaseUri$Path" -Method Post -Headers $headers `
        -ContentType "application/json" `
        -Body ($Payload | ConvertTo-Json -Depth 8 -Compress) `
        -TimeoutSec $TimeoutSeconds
}
function Get-MetricValue {
    param([string]$Text, [string]$Name)
    $match = [regex]::Match(
        $Text, "(?m)^$([regex]::Escape($Name))\s+([0-9.eE+-]+)$")
    if (-not $match.Success) { throw "Metric is missing: $Name" }
    [double]::Parse($match.Groups[1].Value,
        [Globalization.CultureInfo]::InvariantCulture)
}

$health = Invoke-JsonGet "/health"
$ready = Invoke-JsonGet "/ready"
$info = Invoke-JsonGet "/model-info"
$models = Invoke-JsonGet "/v1/models"
if ($health.status -ne "ok" -or -not $ready.ready -or
    $info.model -ne $ExpectedModel -or $models.data[0].id -ne $ExpectedModel) {
    throw "DeepSeek service identity or readiness check failed"
}
if ($ExpectedBuildId -and $info.build_id -ne $ExpectedBuildId) {
    throw "Unexpected DeepSeek build id: $($info.build_id)"
}
if ($info.source.model_id -ne "deepseek-ai/DeepSeek-V4-Flash" -or
    -not $info.source.checkpoint_index_sha256 -or
    -not $info.manifest_content_sha256 -or
    -not $info.experts_index_sha256) {
    throw "DeepSeek bundle identity is incomplete"
}
if (-not $AllowSourceExtents -and
    $info.format.routed_storage -ne "compact-pack") {
    throw "Production DeepSeek gate requires the durable compact pack"
}
if ([int]$info.worker_protocol -lt 4 -or
    $info.worker_prefill.mode -ne "causal_sequential" -or
    [int]$info.worker_prefill.chunk_tokens -ne 1 -or
    $info.worker_kv.dtype -ne "bf16" -or
    $info.worker_kv.allocation -ne "preallocated" -or
    [int]$info.worker_kv.page_tokens -lt 1 -or
    [int64]$info.worker_kv.page_bytes -lt 1 -or
    [int]$info.worker_kv.page_capacity -lt 1) {
    throw "DeepSeek worker execution contract is invalid"
}
if ($info.worker_placement.profile -ne $ExpectedPlacementProfile -or
    $info.runtime_config.placement_profile -ne $ExpectedPlacementProfile -or
    [int64]$info.worker_placement.ram_cache_bytes -lt 1 -or
    [int64]$info.worker_placement.vram_cache_bytes -lt 1) {
    throw "DeepSeek placement contract is invalid"
}

$before = Get-MetricsText
$completionBody = @{
    model = $ExpectedModel
    prompt = "Hi"
    max_tokens = $NewTokens
    stream = $false
    temperature = 0
}
$started = [DateTime]::UtcNow
$response = Invoke-JsonPost "/v1/completions" $completionBody
$elapsed = ([DateTime]::UtcNow - $started).TotalSeconds
if ($response.object -ne "text_completion" -or
    $response.model -ne $ExpectedModel -or
    $response.choices.Count -ne 1 -or
    [int]$response.usage.prompt_tokens -lt 1 -or
    [int]$response.usage.completion_tokens -lt 1 -or
    [string]::IsNullOrEmpty([string]$response.choices[0].text)) {
    throw "DeepSeek completion response is invalid"
}
$chat = Invoke-JsonPost "/v1/chat/completions" @{
    model = $ExpectedModel
    messages = @(@{ role = "user"; content = "Hi" })
    max_completion_tokens = $NewTokens
    stream = $false
    temperature = 0
}
if ($chat.object -ne "chat.completion" -or
    $chat.model -ne $ExpectedModel -or $chat.choices.Count -ne 1 -or
    [string]::IsNullOrEmpty([string]$chat.choices[0].message.content) -or
    [int]$chat.usage.completion_tokens -lt 1) {
    throw "DeepSeek chat completion response is invalid"
}
$responses = Invoke-JsonPost "/v1/responses" @{
    model = $ExpectedModel
    input = "Hi"
    max_output_tokens = $NewTokens
    stream = $false
    temperature = 0
}
if ($responses.object -ne "response" -or
    $responses.status -ne "completed" -or
    $responses.model -ne $ExpectedModel -or
    [string]::IsNullOrEmpty([string]$responses.output[0].content[0].text) -or
    [int]$responses.usage.output_tokens -lt 1) {
    throw "DeepSeek Responses API result is invalid"
}

$after = Get-MetricsText
$afterInfo = Invoke-JsonGet "/model-info"
$completed = (Get-MetricValue $after "expert_service_completed_total") -
    (Get-MetricValue $before "expert_service_completed_total")
$generated = (Get-MetricValue $after "expert_service_generated_tokens_total") -
    (Get-MetricValue $before "expert_service_generated_tokens_total")
$rows = (Get-MetricValue $after "expert_service_decode_rows_total") -
    (Get-MetricValue $before "expert_service_decode_rows_total")
$expectedRows = [int]$response.usage.completion_tokens +
    [int]$chat.usage.completion_tokens +
    [int]$responses.usage.output_tokens
# Session retention intentionally keeps worker state between turns; every
# reserved/allocated KV page must be accounted to a retained session.
$retainedSessions = [int]$afterInfo.worker_sessions.retained
$retainedPages = [int]$afterInfo.worker_sessions.reserved_pages
$pagesPerRequest = [int][Math]::Ceiling(
    [double]$afterInfo.runtime_config.max_context /
    [double]$afterInfo.worker_kv.page_tokens)
if ($completed -ne 3 -or $generated -ne $expectedRows -or
    $rows -lt 3 -or $rows -gt $generated -or
    [int]$afterInfo.active_requests -ne 0 -or
    [int]$afterInfo.worker_kv.reserved_pages -ne $retainedPages -or
    [int]$afterInfo.worker_kv.allocated_pages -ne $retainedPages -or
    $retainedPages -gt ($retainedSessions * $pagesPerRequest)) {
    throw "DeepSeek request resources or counters did not settle"
}

$result = [PSCustomObject]@{
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    endpoint = $BaseUri
    model = $info.model
    build_id = $info.build_id
    routed_storage = $info.format.routed_storage
    checkpoint_index_sha256 = $info.source.checkpoint_index_sha256
    manifest_content_sha256 = $info.manifest_content_sha256
    experts_index_sha256 = $info.experts_index_sha256
    worker_protocol = $info.worker_protocol
    prefill_mode = $info.worker_prefill.mode
    kv_dtype = $info.worker_kv.dtype
    kv_allocation = $info.worker_kv.allocation
    placement_profile = $info.worker_placement.profile
    prompt_tokens = [int]$response.usage.prompt_tokens
    completion_tokens = [int]$response.usage.completion_tokens
    elapsed_seconds = $elapsed
    output_text = [string]$response.choices[0].text
    chat_output_text = [string]$chat.choices[0].message.content
    responses_output_text = [string]$responses.output[0].content[0].text
    endpoint_requests = 3
    completed_delta = $completed
    generated_tokens_delta = $generated
    decode_rows_delta = $rows
    retained_sessions = $retainedSessions
    retained_reserved_pages = $retainedPages
    context_settled = $true
    ttft_p95_seconds = Get-MetricValue $after `
        "expert_service_ttft_seconds_p95"
    inter_token_p95_seconds = Get-MetricValue $after `
        "expert_service_inter_token_seconds_p95"
}
$artifact = Write-JsonArtifact -Value $result `
    -Name "deepseek-service-smoke-latest.json"
$result | ConvertTo-Json -Depth 5
Write-Output "DeepSeek service smoke artifact: $artifact"
