$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
$prefix = Join-Path $repo 'out/benchmarks/pi256k-tools'
[System.IO.Directory]::CreateDirectory($prefix) | Out-Null
if (-not (Get-Command node -ErrorAction SilentlyContinue) -or
    -not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw 'Node and npm are required on the benchmark host'
}
& npm install --prefix $prefix --no-save --no-audit --no-fund `
    '@earendil-works/pi-coding-agent@0.87.1'
if ($LASTEXITCODE -ne 0) { throw 'Pinned Pi installation failed' }
$cli = Join-Path $prefix 'node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js'
if (-not (Test-Path -LiteralPath $cli -PathType Leaf)) { throw 'Pinned Pi CLI is missing' }
$version = (& node $cli --version).Trim()
if ($LASTEXITCODE -ne 0 -or $version -notmatch '0\.87\.1') {
    throw "Pi version mismatch: $version"
}
@{ cli = $cli; version = $version; status = 'ready' } | ConvertTo-Json -Compress
