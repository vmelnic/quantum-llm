. (Join-Path $PSScriptRoot "Common.ps1")

$artifact = Join-Path (Join-Path $script:RepoRoot "artifacts") "p6-download-process.json"
if (-not (Test-Path $artifact)) { throw "No P6 download process artifact" }
$state = Get-Content $artifact -Raw | ConvertFrom-Json
$process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
if ($null -eq $process -or $process.ProcessName -notin @("python", "hf")) {
    # hf.exe is a launcher; Xet work may continue in its Python child after
    # the launcher PID disappears. Recover the tracked PID from the command
    # line instead of declaring a live resumable download dead.
    $escapedModel = [WildcardPattern]::Escape([string]$state.model_id)
    $worker = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -in @("python.exe", "hf.exe") -and
            $_.CommandLine -like "*download*$escapedModel*" -or
            ($_.Name -in @("python.exe", "hf.exe") -and
             $_.CommandLine -like "*$escapedModel*--max-workers*")
        } | Select-Object -First 1
    if ($null -ne $worker) {
        $state.pid = [int]$worker.ProcessId
        $state | ConvertTo-Json | Set-Content $artifact -Encoding UTF8
        $process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
    }
}
$cacheName = "models--" + ($state.model_id -replace "/", "--")
$root = Join-Path "C:\Users\vladi\.cache\huggingface\hub" $cacheName
$bytes = if (Test-Path $root) {
    [int64]((Get-ChildItem $root -File -Recurse -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum)
} else { [int64]0 }
$shards = @(Get-ChildItem (Join-Path $root "snapshots") -Filter "model-*.safetensors" `
    -File -Recurse -ErrorAction SilentlyContinue)
$incomplete = @(Get-ChildItem (Join-Path $root "blobs") -Filter "*.incomplete" `
    -File -ErrorAction SilentlyContinue)
[PSCustomObject]@{
    model_id = $state.model_id
    pid = $state.pid
    running = $null -ne $process
    completed = ($null -eq $process) -and $shards.Count -eq 41 -and $incomplete.Count -eq 0
    complete_shards = $shards.Count
    expected_shards = 41
    incomplete_transfers = $incomplete.Count
    cached_bytes = $bytes
    stderr_tail = @(Get-Content $state.stderr -Tail 5 -ErrorAction SilentlyContinue)
} | ConvertTo-Json -Depth 4
