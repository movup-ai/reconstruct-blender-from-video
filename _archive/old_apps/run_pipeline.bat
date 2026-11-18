@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first. & pause & exit /b 1)
set "DEV=cpu"
.venv\Scripts\python -c "import onnxruntime as o,sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)" >nul 2>nul
if not errorlevel 1 set "DEV=cuda"
echo Device: %DEV%
echo Video -^> marks -^> camera -^> 3D people -^> grey set -^> Blender render -^> videos + zip
echo.
.venv\Scripts\python pipeline\run_pipeline.py %* --device %DEV%
echo.
echo Finished. Submissions + zips: see submit_root in config.json (default F:\Blender task\Submit_auto)
echo Previews, logs and report_*.csv: work_root (default F:\Blender task\Work_auto)
pause
