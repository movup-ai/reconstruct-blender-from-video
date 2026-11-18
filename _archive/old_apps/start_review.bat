@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first. & pause & exit /b 1)
echo Key-point review app: http://127.0.0.1:8765   (close this window to stop it)
start "" http://127.0.0.1:8765
.venv\Scripts\python review_app.py
pause
