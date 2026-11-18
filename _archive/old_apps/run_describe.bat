@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first. & pause & exit /b 1)
.venv\Scripts\python describe.py %*
pause
