@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_integrated_assessment_exe.ps1"
echo.
pause
