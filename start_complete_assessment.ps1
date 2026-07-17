$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$App = Join-Path $Project "omega_complete_app.py"
function Find-Python {
    $Candidates = @((Join-Path $Project ".venv\Scripts\pythonw.exe"), (Join-Path $Project ".venv\Scripts\python.exe"))
    foreach ($Candidate in $Candidates) { if (Test-Path -LiteralPath $Candidate) { return $Candidate } }
    $PythonW = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($PythonW) { return $PythonW.Source }
    $Python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($Python) { return $Python.Source }
    $Py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($Py) { Start-Process -FilePath $Py.Source -ArgumentList @("-3", "`"$App`"") -WorkingDirectory $Project; exit }
    throw "Python 3 was not found. Run SETUP_OMEGA_FISH.bat first."
}
$Executable = Find-Python
Start-Process -FilePath $Executable -ArgumentList @("`"$App`"") -WorkingDirectory $Project
