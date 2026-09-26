param([string]$BuildRoot = '')

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
if (-not $BuildRoot) { $BuildRoot = Join-Path $repo 'out/llama-v0.5.0' }
$source = Join-Path $BuildRoot 'source'
$build = Join-Path $BuildRoot 'build'
$commit = '7fe450e19305b828c199d602c23a8337aaa1f03b'
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    [System.IO.Directory]::CreateDirectory($BuildRoot) | Out-Null
    & git clone --branch v0.5.0 --depth 1 https://github.com/ggml-org/llama.cpp.git $source
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp clone failed' }
}
$actual = (& git -C $source rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actual -ne $commit) {
    throw "llama.cpp source mismatch: expected $commit, found $actual"
}
$cudaRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit/CUDA/v13.4'
$nvcc = Join-Path $cudaRoot 'bin/nvcc.exe'
if (-not (Test-Path -LiteralPath $nvcc -PathType Leaf)) {
    throw "CUDA 13.4 compiler is missing: $nvcc"
}
$nvccVersion = (& $nvcc --version) -join ' '
if ($LASTEXITCODE -ne 0 -or $nvccVersion -notmatch 'release 13\.4') {
    throw "Expected CUDA 13.4 compiler at $nvcc; found: $nvccVersion"
}
$integration = Join-Path $cudaRoot 'extras/visual_studio_integration/MSBuildExtensions'
if (-not (Test-Path -LiteralPath $integration -PathType Container)) {
    throw "CUDA Visual Studio integration is missing: $integration"
}
& cmake --fresh -S $source -B $build -T "cuda=$cudaRoot" `
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 `
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
if ($LASTEXITCODE -ne 0) { throw 'llama.cpp CMake configure failed' }
& cmake --build $build --config Release --target llama-server --clean-first --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'llama.cpp CUDA build failed' }
$binary = Join-Path $build 'bin/Release/llama-server.exe'
if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) { throw "Missing $binary" }
@{ binary = $binary; commit = $commit; status = 'built' } | ConvertTo-Json -Compress
