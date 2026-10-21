function Get-LlamaProjector {
    param([Parameter(Mandatory = $true)]$Spec)
    $property = $Spec.PSObject.Properties['projector']
    if ($null -eq $property) { return $null }
    $projector = $property.Value
    if ($null -eq $projector -or $projector -isnot [PSCustomObject]) { throw 'Invalid projector policy' }
    $required = @('repository', 'revision', 'file', 'bytes', 'sha256', 'offload')
    if (@(Compare-Object $required @($projector.PSObject.Properties.Name)).Count -ne 0) {
        throw 'Invalid projector policy fields'
    }
    if ($projector.repository -notmatch '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$' -or
        $projector.revision -notmatch '^[0-9a-f]{40}$' -or
        $projector.file -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$' -or
        $projector.sha256 -notmatch '^[0-9a-f]{64}$' -or
        ($projector.bytes -isnot [int] -and $projector.bytes -isnot [long]) -or
        $projector.bytes -lt 1 -or $projector.offload -isnot [bool]) {
        throw 'Invalid projector identity or placement'
    }
    if ($projector.file -eq $Spec.file) { throw 'Projector and model filenames must differ' }
    return $projector
}

function Get-LlamaParallelSlots {
    param([Parameter(Mandatory = $true)]$Spec)
    $property = $Spec.PSObject.Properties['parallel']
    if ($null -eq $property) { return 1 }
    $slots = $property.Value
    if (($slots -isnot [int] -and $slots -isnot [long]) -or $slots -lt 1 -or $slots -gt 2) {
        throw 'Parallel slots must be an integer between 1 and 2'
    }
    if ([long]$Spec.context % $slots -ne 0) { throw 'Total context must be divisible by parallel slots' }
    return [int]$slots
}

function Get-LlamaServerArguments {
    param(
        [Parameter(Mandatory = $true)]$Spec,
        [Parameter(Mandatory = $true)][string]$Weights,
        [string]$HostAddress = '0.0.0.0',
        [int]$Port = 8081,
        [int]$TimeoutSeconds = 14400
    )

    $slots = Get-LlamaParallelSlots -Spec $Spec
    $fit = 'off'
    $gpuLayers = 'all'
    $fitProperty = $Spec.PSObject.Properties['automatic_fit']
    if ($null -ne $fitProperty) {
        if ($fitProperty.Value -isnot [bool]) { throw 'Automatic fit must be a boolean' }
        if ($fitProperty.Value) { $fit = 'on'; $gpuLayers = 'auto' }
    }
    $reasoningEffort = 'xhigh'
    $effortProperty = $Spec.PSObject.Properties['reasoning_effort']
    if ($null -ne $effortProperty) {
        if ($effortProperty.Value -isnot [string] -or
            $effortProperty.Value -notin @('low', 'medium', 'high', 'xhigh')) {
            throw 'Invalid reasoning effort policy'
        }
        $reasoningEffort = $effortProperty.Value
    }
    $arguments = @('--model', $Weights, '--alias', $Spec.model_id,
        '--host', $HostAddress, '--port', [string]$Port,
        '--ctx-size', [string]$Spec.context, '--parallel', [string]$slots,
        '--gpu-layers', $gpuLayers, '--split-mode', 'none', '--main-gpu', '0', '--fit', $fit,
        '--cache-type-k', $Spec.cache_type_k, '--cache-type-v', $Spec.cache_type_v,
        '--flash-attn', 'on', '--batch-size', '512', '--ubatch-size', '128',
        '--threads', '6', '--threads-batch', '6', '--jinja', '--cache-prompt',
        '--ctx-checkpoints', '32', '--checkpoint-min-step', '8192', '--cache-ram', '8192',
        '--reasoning', 'on', '--reasoning-effort', $reasoningEffort, '--reasoning-preserve',
        '--no-context-shift', '--no-warmup', '--predict', [string]$Spec.max_output,
        '--timeout', [string]$TimeoutSeconds, '--no-webui')

    $property = $Spec.PSObject.Properties['speculative']
    if ($null -ne $property) {
        $draft = $property.Value
        $required = @('type', 'max_tokens', 'cache_type_k', 'cache_type_v')
        if ($null -eq $draft -or $draft -isnot [PSCustomObject]) { throw 'Invalid speculative policy' }
        $keys = @($draft.PSObject.Properties.Name)
        if (@(Compare-Object $required $keys).Count -ne 0) { throw 'Invalid speculative policy fields' }
        if ($draft.type -ne 'draft-mtp') { throw 'Unsupported speculative policy' }
        if (($draft.max_tokens -isnot [int] -and $draft.max_tokens -isnot [long]) -or
            $draft.max_tokens -lt 1 -or $draft.max_tokens -ge 128) {
            throw 'Draft limit must be an integer between 1 and 127 for the configured microbatch'
        }
        $cacheTypes = @('f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl')
        if ($draft.cache_type_k -notin $cacheTypes -or $draft.cache_type_v -notin $cacheTypes) {
            throw 'Invalid draft KV cache encoding'
        }
        # Integrated MTP reuses target weights; do not supply a second model.
        $arguments += @('--spec-type', $draft.type, '--spec-draft-n-max', [string]$draft.max_tokens,
            '--spec-draft-type-k', $draft.cache_type_k, '--spec-draft-type-v', $draft.cache_type_v)
    }
    $projector = Get-LlamaProjector -Spec $Spec
    if ($null -ne $projector) {
        $path = Join-Path (Split-Path $Weights -Parent) $projector.file
        $arguments += @('--mmproj', $path)
        if (-not $projector.offload) { $arguments += '--no-mmproj-offload' }
    }
    return $arguments
}
