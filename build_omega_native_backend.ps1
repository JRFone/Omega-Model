$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = (Get-Command python.exe -ErrorAction Stop).Source }
Push-Location $Project
try {
    & $Python build_native_backend.py --clean
    if ($LASTEXITCODE -ne 0) { throw "Native backend build failed." }
    & $Python omega_cli.py native-status
} finally { Pop-Location }
