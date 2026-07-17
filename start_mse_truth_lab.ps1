$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python.exe -ErrorAction Stop).Source
}
Start-Process -FilePath $Python -ArgumentList @((Join-Path $Project "omega_desktop.py"), "--mode", "truthmse") -WorkingDirectory $Project
