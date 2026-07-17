$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$App = Join-Path $Project "quant_lab_app.py"

function Find-Python {
    $Candidates = @(
        (Join-Path $Project ".venv\Scripts\pythonw.exe"),
        (Join-Path $Project ".venv\Scripts\python.exe")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate) {
            return @{ File = $Candidate; Args = @() }
        }
    }
    $PythonW = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($PythonW) { return @{ File = $PythonW.Source; Args = @() } }
    $Python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($Python) { return @{ File = $Python.Source; Args = @() } }
    $Py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($Py) { return @{ File = $Py.Source; Args = @("-3") } }
    throw "Python 3 was not found. Run SETUP_OMEGA_FISH.bat first."
}

if (-not (Test-Path -LiteralPath $App)) {
    throw "Quant Lab launcher not found: $App"
}

$Python = Find-Python
$Args = @()
$Args += $Python.Args
$Args += "`"$App`""
Start-Process -FilePath $Python.File -ArgumentList $Args -WorkingDirectory $Project
