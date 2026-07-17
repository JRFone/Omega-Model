$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\pythonw.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = (Get-Command pythonw.exe -ErrorAction Stop).Source }
Start-Process -FilePath $Python -ArgumentList @('"' + (Join-Path $Project "omega_desktop.py") + '"', '--mode', 'charts') -WorkingDirectory $Project
