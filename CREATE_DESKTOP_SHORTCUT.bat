@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "tools\create_desktop_shortcut.py"
) else (
  py -3 "tools\create_desktop_shortcut.py"
)
if errorlevel 1 (
  echo.
  echo The desktop shortcut could not be created.
  pause
  exit /b 1
)
echo.
echo Omega FISH Model shortcut created on the Windows Desktop.
pause
