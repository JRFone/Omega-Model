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
%PYTHON_CMD% omega_cli.py complete-demo --output reports\complete_release_11_demo
pause
