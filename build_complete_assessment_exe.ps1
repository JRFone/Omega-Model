$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = (Get-Command python.exe -ErrorAction Stop).Source }
& $Python -m pip install --upgrade pyinstaller
& $Python -m PyInstaller --noconfirm --clean --windowed --name "Omega FISH Complete Assessment" --paths $Project (Join-Path $Project "omega_complete_app.py")
Write-Host "Executable created under dist\Omega FISH Complete Assessment."
