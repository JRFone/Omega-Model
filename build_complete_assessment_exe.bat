@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_complete_assessment_exe.ps1"
pause
