@echo off
setlocal
cd /d "%~dp0"
if not exist "logs" mkdir "logs"
if exist ".venv\Scripts\pythonw.exe" (
  start "Omega FISH Model" ".venv\Scripts\pythonw.exe" "omega_desktop.py"
  exit /b 0
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "omega_desktop.py" >> "logs\startup.log" 2>&1
  if errorlevel 1 pause
  exit /b %errorlevel%
)
where py >nul 2>&1
if not errorlevel 1 (
  py -3 "omega_desktop.py" >> "logs\startup.log" 2>&1
  if errorlevel 1 pause
  exit /b %errorlevel%
)
echo Omega could not find its Python environment. Run SETUP_OMEGA_FISH.bat first.
pause
exit /b 1
