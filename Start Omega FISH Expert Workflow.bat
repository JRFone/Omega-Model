@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" omega_desktop.py --mode expert
) else (
  py -3 omega_desktop.py --mode expert
)
if errorlevel 1 pause
