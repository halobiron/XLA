param(
    [int]$Port = 8765,
    [string]$Device = 'cpu'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$logDirectory = Join-Path $projectRoot 'outputs\deployment'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python environment not found: $python"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:PYTHONUNBUFFERED = '1'
# Keep concurrent HTTP clients responsive while CPU inference is running.
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'

$server = Start-Process -FilePath $python -ArgumentList @(
    '-u', '-m', 'tacsx.serve', '--host', '127.0.0.1',
    '--port', $Port, '--device', $Device
) -WorkingDirectory $projectRoot -WindowStyle Hidden -Wait -PassThru `
    -RedirectStandardOutput (Join-Path $logDirectory 'server.log') `
    -RedirectStandardError (Join-Path $logDirectory 'server-error.log')

exit $server.ExitCode
