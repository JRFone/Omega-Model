@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" omega_desktop.py --mode noaa
) else (
  py -3 omega_desktop.py --mode noaa
)
if errorlevel 1 pause
