param(
    [string]$FlywireRoot = (Join-Path $PSScriptRoot '../../Flywire-Brain-Model'),
    [ValidateSet('gpu', 'cpu')][string]$NeuralBackend = 'gpu',
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$serverRoot = Split-Path $PSScriptRoot -Parent
$FlywireRoot = (Resolve-Path -LiteralPath $FlywireRoot).Path
$pythonPath = Join-Path $FlywireRoot '.venv-body312/Scripts/python.exe'
$profilePath = Join-Path $FlywireRoot 'hybrid_fly/eon_config.json'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python environment missing: $pythonPath"
}
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $serverRoot 'src'
try {
    Write-Host "Loading Flywire. When serving starts, open http://127.0.0.1:$Port"
    Write-Host 'Keep this terminal open. Stop with Ctrl+C.'
    & $pythonPath -m flygym_stream.cli serve --local-windows `
        --flywire-config $profilePath --neural-backend $NeuralBackend `
        --backend cpu --bind 127.0.0.1 --port $Port `
        --width 640 --height 480 --fps 15 --speed 0.1 `
        --flywire-output (Join-Path $FlywireRoot 'Outputs/server-local')
    if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $oldPythonPath
}
