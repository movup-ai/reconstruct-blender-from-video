@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first. & pause & exit /b 1)
set "DEV=cpu"
.venv\Scripts\python -c "import onnxruntime as o,sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)" >nul 2>nul
if not errorlevel 1 set "DEV=cuda"
echo Fit only (no Blender): writes preview.jpg per clip into the work folder so you can check before rendering.
.venv\Scripts\python pipeline\run_pipeline.py %* --stage fit --device %DEV%
pause
