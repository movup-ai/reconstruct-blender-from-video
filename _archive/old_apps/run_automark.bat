@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first. & pause & exit /b 1)
set "DEV=cpu"
.venv\Scripts\python -c "import onnxruntime as o,sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)" >nul 2>nul
if not errorlevel 1 set "DEV=cuda"
echo Device: %DEV%
if "%~1"=="" (
  echo No video given - processing everything in "F:\Blender task\Source"
  .venv\Scripts\python automark.py "F:\Blender task\Source" --out "%~dp0output" --device %DEV%
) else (
  .venv\Scripts\python automark.py %* --out "%~dp0output" --device %DEV%
)
echo.
echo Finished. For each clip open  output\...\check.html  in Chrome or Edge,
echo fix the marks, press Save, and keep checked.json next to check.html.
pause
