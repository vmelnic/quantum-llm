param(
    [string]$ModelId = "Qwen/Qwen3-Next-80B-A3B-Instruct",
    [int]$MaxWorkers = 4
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

$hf = "C:\Users\vladi\.hf-cli\venv\Scripts\hf.exe"
if (-not (Test-Path $hf -PathType Leaf)) { $hf = (Get-Command hf.exe -ErrorAction Stop).Source }
$artifact = Join-Path (Join-Path $script:RepoRoot "artifacts") "p6-download-process.json"
$stdout = Join-Path (Join-Path $script:RepoRoot "logs") "p6-download.stdout.log"
$stderr = Join-Path (Join-Path $script:RepoRoot "logs") "p6-download.stderr.log"
if (Test-Path $artifact) {
    $previous = Get-Content $artifact -Raw | ConvertFrom-Json
    $running = Get-Process -Id ([int]$previous.pid) -ErrorAction SilentlyContinue
    if ($null -ne $running) { throw "P6 download is already running as PID $($previous.pid)" }
}
# Windows PowerShell 5.1 flattens ArgumentList itself and does not reliably
# preserve an item containing spaces. Reject those here because Hugging Face
# repository ids cannot contain whitespace, then pass one deterministic string.
if ($ModelId -match "\s") { throw "ModelId cannot contain whitespace: $ModelId" }
$escapedModel = [WildcardPattern]::Escape($ModelId)
$existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -in @("python.exe", "hf.exe") -and
        $_.CommandLine -like "*download*$escapedModel*"
    } | Select-Object -First 1
if ($null -ne $existing) {
    throw "P6 download is already running as PID $($existing.ProcessId)"
}
$arguments = "download $ModelId --max-workers $MaxWorkers"
# Keep a cmd.exe parent alive around the Python launcher. Directly starting the
# hf.exe console shim can exit before its Python/Xet descendants are observable
# when this script itself is invoked through non-interactive SSH.
$cmdArguments = "/d /c $hf $arguments"
$process = Start-Process -FilePath $env:ComSpec -ArgumentList $cmdArguments `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -PassThru -WindowStyle Hidden
Start-Sleep -Milliseconds 500
$process.Refresh()
if ($process.HasExited -and $process.ExitCode -ne 0) {
    $tail = @(Get-Content $stderr -Tail 20 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
    throw "P6 download exited immediately with code $($process.ExitCode):$([Environment]::NewLine)$tail"
}
[PSCustomObject]@{
    schema_version = 1
    model_id = $ModelId
    pid = $process.Id
    started_utc = [DateTime]::UtcNow.ToString("o")
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json | Set-Content $artifact -Encoding UTF8
Get-Content $artifact -Raw
