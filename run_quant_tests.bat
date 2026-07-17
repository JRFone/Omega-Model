@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
if not defined PYTHON_CMD where python.exe >nul 2>nul && set "PYTHON_CMD=python.exe"
if not defined PYTHON_CMD where py.exe >nul 2>nul && set "PYTHON_CMD=py.exe -3"

if not defined PYTHON_CMD (
  echo Python 3 was not found. Run SETUP_OMEGA_FISH.bat first.
  pause
  exit /b 1
)

set "PYTHONPATH=%CD%"
%PYTHON_CMD% -m compileall omega_desktop.py omega_self_check.py quant_lab_app.py integrated_assessment_app.py omega_complete_app.py stock_model tests
if errorlevel 1 goto :failed
%PYTHON_CMD% -m unittest discover -s tests -v
if errorlevel 1 goto :failed
%PYTHON_CMD% omega_self_check.py --quick
if errorlevel 1 goto :failed

echo.
echo Omega FISH complete validation passed.
pause
exit /b 0

:failed
echo.
echo Omega FISH validation failed. Review the error output above.
pause
exit /b 1
