$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$PortableExe = Join-Path $Project "dist\Omega FISH Model\Omega FISH Model.exe"
$InstalledExe = Join-Path $env:LOCALAPPDATA "Programs\Omega FISH Model\Omega FISH Model.exe"

if (Test-Path -LiteralPath $PortableExe) {
    Start-Process -FilePath $PortableExe -WorkingDirectory (Split-Path -Parent $PortableExe)
    exit
}
if (Test-Path -LiteralPath $InstalledExe) {
    Start-Process -FilePath $InstalledExe -WorkingDirectory (Split-Path -Parent $InstalledExe)
    exit
}

$Candidates = @(
    (Join-Path $Project ".venv\Scripts\pythonw.exe"),
    (Join-Path $Project ".venv\Scripts\python.exe")
)
foreach ($Candidate in $Candidates) {
    if (Test-Path -LiteralPath $Candidate) {
        Start-Process -FilePath $Candidate -ArgumentList @('"' + (Join-Path $Project "omega_desktop.py") + '"') -WorkingDirectory $Project
        exit
    }
}
$PythonW = Get-Command pythonw.exe -ErrorAction SilentlyContinue
if ($PythonW) {
    Start-Process -FilePath $PythonW.Source -ArgumentList @('"' + (Join-Path $Project "omega_desktop.py") + '"') -WorkingDirectory $Project
    exit
}
$Python = Get-Command python.exe -ErrorAction SilentlyContinue
if ($Python) {
    Start-Process -FilePath $Python.Source -ArgumentList @('"' + (Join-Path $Project "omega_desktop.py") + '"') -WorkingDirectory $Project
    exit
}
throw "Omega FISH is not installed. Run SETUP_OMEGA_FISH.bat first."
