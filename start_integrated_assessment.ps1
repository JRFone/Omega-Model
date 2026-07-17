$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$App = Join-Path $Project "integrated_assessment_app.py"

function Find-Python {
    $Candidates = @(
        (Join-Path $Project ".venv\Scripts\pythonw.exe"),
        (Join-Path $Project ".venv\Scripts\python.exe")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate) { return $Candidate }
    }
    $PythonW = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($PythonW) { return $PythonW.Source }
    $Python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($Python) { return $Python.Source }
    throw "Python 3 was not found. Run SETUP_OMEGA_FISH.bat first."
}

if (-not (Test-Path -LiteralPath $App)) { throw "Integrated Assessment Lab was not found: $App" }
$Python = Find-Python
Start-Process -FilePath $Python -ArgumentList @("`"$App`"") -WorkingDirectory $Project
