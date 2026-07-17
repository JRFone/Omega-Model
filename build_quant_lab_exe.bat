@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_quant_lab_exe.ps1"
pause
