$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Project

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $Python = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $Python) {
        $Py = Get-Command py.exe -ErrorAction SilentlyContinue
        if (-not $Py) { throw "Python 3 was not found." }
        & $Py.Source -3 -m venv .venv
    } else {
        & $Python.Source -m venv .venv
    }
}

$VenvPython = Join-Path $Project ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Project "requirements_build.txt")
& $VenvPython -m pip install pyinstaller

Write-Host "Running integrated assessment tests before building..."
& $VenvPython -m unittest tests.test_age_structured -v
if ($LASTEXITCODE -ne 0) { throw "Integrated assessment tests failed. Executable was not built." }

& $VenvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name "Omega FISH Integrated Assessment" `
    --collect-all numpy `
    --collect-all pandas `
    --collect-submodules stock_model `
    --hidden-import tkinter `
    integrated_assessment_app.py

Write-Host ""
Write-Host "Build complete. Executable:"
Write-Host (Join-Path $Project "dist\Omega FISH Integrated Assessment\Omega FISH Integrated Assessment.exe")
