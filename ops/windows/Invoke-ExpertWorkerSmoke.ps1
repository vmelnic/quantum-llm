param(
    [string]$Container = "C:\Users\vladi\quantum-llm\work\models\olmoe-expert-pack-int8",
    [string]$Configuration = "Release"
)

. (Join-Path $PSScriptRoot "Common.ps1")

$worker = Join-Path $script:RepoRoot "out\build\windows-msvc-release\runtime\$Configuration\expert-olmoe-runner.exe"
if (-not (Test-Path $worker -PathType Leaf)) {
    throw "Worker executable not found: $worker"
}

$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $worker
$start.Arguments = "`"$Container`" --worker 64"
$start.UseShellExecute = $false
$start.RedirectStandardInput = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$start.CreateNoWindow = $true
$process = [Diagnostics.Process]::new()
$process.StartInfo = $start
if (-not $process.Start()) { throw "Failed to start CUDA worker" }

try {
    $ready = $process.StandardOutput.ReadLine() | ConvertFrom-Json
    if ($ready.type -ne "ready") { throw "Worker did not become ready" }
    $process.StandardInput.WriteLine("BEGIN`t1`t510,5347,273,6181,310")
    $process.StandardInput.Flush()
    $begun = $process.StandardOutput.ReadLine() | ConvertFrom-Json
    $process.StandardInput.WriteLine("NEXT`t1`t0")
    $process.StandardInput.Flush()
    $first = $process.StandardOutput.ReadLine() | ConvertFrom-Json
    $process.StandardInput.WriteLine("NEXT`t1`t1")
    $process.StandardInput.Flush()
    $second = $process.StandardOutput.ReadLine() | ConvertFrom-Json
    $process.StandardInput.WriteLine("SHUTDOWN")
    $process.StandardInput.Flush()
    $shutdown = $process.StandardOutput.ReadLine() | ConvertFrom-Json
    if ($begun.type -ne "begun" -or $first.token -ne 7785 -or
        $second.token -ne 15 -or $shutdown.type -ne "shutdown") {
        throw "Worker protocol returned an unexpected response"
    }
    if (-not $process.WaitForExit(10000) -or $process.ExitCode -ne 0) {
        throw "Worker did not shut down cleanly"
    }
    [PSCustomObject]@{
        status = "pass"
        first_tokens = @($first.token, $second.token)
        protocol = $ready.protocol
    } | ConvertTo-Json -Depth 4
}
finally {
    if (-not $process.HasExited) { $process.Kill($true) }
    $process.Dispose()
}
