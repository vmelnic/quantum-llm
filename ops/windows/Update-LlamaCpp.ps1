param([string]$Version = 'latest')

. (Join-Path $PSScriptRoot 'Common.ps1')
$ProgressPreference = 'SilentlyContinue'
$headers = @{ 'User-Agent' = 'quantum-llm-ops'; Accept = 'application/vnd.github+json' }
$api = 'https://api.github.com/repos/ggml-org/llama.cpp'
if ($Version -eq 'latest') {
    $release = Invoke-RestMethod "$api/releases/latest" -Headers $headers
    if ($release.prerelease -or $release.draft) { throw 'Expected a stable release' }
    $Version = [string]$release.tag_name
}
if ($Version -notmatch '^v[0-9]+\.[0-9]+\.[0-9]+$') { throw 'Expected a stable version tag' }
$commit = [string](Invoke-RestMethod "$api/commits/$Version" -Headers $headers).sha
if ($commit -notmatch '^[0-9a-f]{40}$') { throw 'Invalid upstream commit' }
$installPath = Join-Path $script:RepoRoot 'artifacts/llama-install.json'
if (Test-Path -LiteralPath $installPath -PathType Leaf) {
    $current = Get-Content -LiteralPath $installPath -Raw | ConvertFrom-Json
    if ($current.schema -ne 'llama-install-v1') { throw 'Invalid llama.cpp installation record' }
    if ($current.version -eq $Version) {
        if ($current.commit -ne $commit -or
            (Get-FileHash -LiteralPath $current.binary -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            $current.binary_sha256) { throw 'Installed release integrity mismatch; refusing in-place rebuild' }
        @{status = 'already-installed'; version = $Version; commit = $commit; binary = $current.binary} |
            ConvertTo-Json -Compress
        exit
    }
}
$root = Join-Path $script:RepoRoot "out/llama-$Version"
$source = Join-Path $root 'source'
$build = Join-Path $root 'build'
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    [IO.Directory]::CreateDirectory($root) | Out-Null
    & git clone --branch $Version --depth 1 https://github.com/ggml-org/llama.cpp.git $source
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp clone failed' }
}
$actual = (& git -C $source rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actual -ne $commit) { throw 'Upstream source commit mismatch' }
$dirty = @(& git -C $source status --porcelain)
if ($LASTEXITCODE -ne 0 -or $dirty.Count -gt 0) { throw 'Refusing to build modified upstream source' }
$cudaRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit/CUDA/v13.4'
$nvcc = Join-Path $cudaRoot 'bin/nvcc.exe'
if (-not (Test-Path -LiteralPath $nvcc -PathType Leaf)) { throw 'CUDA 13.4 compiler is missing' }
& cmake --fresh -S $source -B $build -T "cuda=$cudaRoot" `
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 `
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
if ($LASTEXITCODE -ne 0) { throw 'llama.cpp CMake configure failed' }
& cmake --build $build --config Release --target llama-server --clean-first --parallel 2
if ($LASTEXITCODE -ne 0) { throw 'llama.cpp clean CUDA build failed' }
$binary = Join-Path $build 'bin/Release/llama-server.exe'
if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) { throw 'Missing llama-server binary' }
function Read-BinaryOutput([string]$Argument) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $binary
    $info.Arguments = $Argument
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.CreateNoWindow = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    try {
        $null = $process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(60000)) { $process.Kill(); throw "CLI inspection timed out: $Argument" }
        $output = $stdout.Result + "`n" + $stderr.Result
        if ($process.ExitCode -ne 0) { throw "CLI inspection failed: $output" }
        return $output
    } finally { $process.Dispose() }
}
$reportedVersion = Read-BinaryOutput '--version'
if ($reportedVersion -notmatch [regex]::Escape($Version.TrimStart('v')) -or
    $reportedVersion -notmatch [regex]::Escape($commit.Substring(0, 7))) {
    throw "Built binary version mismatch: $reportedVersion"
}
# Validate CLI compatibility without loading weights or contacting the active server.
$help = Read-BinaryOutput '--help'
$required = @('--ctx-size', '--parallel', '--fit', '--cache-type-k', '--cache-type-v',
    '--ctx-checkpoints', '--checkpoint-min-step', '--reasoning-effort', '--reasoning-preserve',
    '--spec-type', '--spec-draft-n-max', '--spec-draft-type-k', '--spec-draft-type-v')
foreach ($flag in $required) {
    if ($help -notmatch [regex]::Escape($flag)) { throw "New binary lacks launcher option $flag" }
}
function Write-AtomicJson([string]$Path, $Value) {
    $temporary = $Path + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    try {
        $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
        if (Test-Path -LiteralPath $Path) { [IO.File]::Replace($temporary, $Path, [NullString]::Value) }
        else { [IO.File]::Move($temporary, $Path) }
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
    }
}
Initialize-ExperimentDirectories
$install = @{ schema = 'llama-install-v1'; version = $Version; commit = $commit;
    binary = $binary; binary_sha256 = (Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant();
    cuda_architecture = 86; installed_utc = [DateTime]::UtcNow.ToString('o') }
Write-AtomicJson (Join-Path $script:RepoRoot 'artifacts/llama-install.json') $install
$launchPath = Join-Path $script:RepoRoot 'artifacts/llama-server-launch.json'
if (Test-Path -LiteralPath $launchPath -PathType Leaf) {
    $launch = Get-Content -LiteralPath $launchPath -Raw | ConvertFrom-Json
    if ($launch.schema -ne 'llama-launch-v1') { throw 'Unknown prepared launch schema' }
    $launch.binary = $binary
    Write-AtomicJson $launchPath $launch
}
$install.status = 'installed'
$install.restart_required = $true
$install | ConvertTo-Json -Depth 4 -Compress
