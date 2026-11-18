@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  Setting up the automark agent (one time)
echo ============================================================
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto havepy
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto havepy
echo.
echo Python was not found.
echo Install Python 3.10 or newer from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during the install, then run setup.bat again.
pause
exit /b 1
:havepy
echo Using: %PY%
%PY% -m venv .venv
if errorlevel 1 (echo Could not create the Python environment. & pause & exit /b 1)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install rtmlib onnxruntime opencv-python numpy scipy requests imageio-ffmpeg
if errorlevel 1 (echo Package install failed - check the internet connection and try again. & pause & exit /b 1)
where nvidia-smi >nul 2>nul
if errorlevel 1 goto nogpu
echo NVIDIA graphics card found - installing the GPU version of onnxruntime (large download)...
.venv\Scripts\python -m pip uninstall -y onnxruntime
.venv\Scripts\python -m pip install "onnxruntime-gpu[cuda,cudnn]"
if errorlevel 1 (
  echo GPU version failed - going back to the CPU version.
  .venv\Scripts\python -m pip install onnxruntime
)
:nogpu
echo Installing OpenVINO (about 3x faster pose marking on Intel CPUs)...
.venv\Scripts\python -m pip install openvino
.venv\Scripts\python pipeline\get_model.py
set "OLL="
where ollama >nul 2>nul
if not errorlevel 1 set "OLL=ollama"
if not defined OLL if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLL=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLL (
  echo.
  echo Local AI for "AI edit": install Ollama ^(free^) from https://ollama.com and run setup.bat again,
  echo or press "Download local model" in the app's Settings after installing it.
  goto afteroll
)
set "OMODEL=qwen3-vl:4b"
where nvidia-smi >nul 2>nul
if not errorlevel 1 set "OMODEL=qwen3-vl:8b"
echo Downloading the local AI model %OMODEL% for AI edit (one time, a few GB)...
"%OLL%" pull %OMODEL%
.venv\Scripts\python -c "import json,os;p='config.json';c=json.load(open(p,encoding='utf-8')) if os.path.exists(p) else {};c['ollama_model']='%OMODEL%';c['ai_provider']='local';json.dump(c,open(p,'w',encoding='utf-8'),ensure_ascii=False,indent=2)"
:afteroll
if errorlevel 1 (echo Depth model download failed - run setup.bat again later. The pipeline still works with a flat grey ground.)
echo.
echo ============================================================
echo  Setup finished.
echo  Start the app:  double-click start_app.bat  (opens in your browser)
echo  Command-line alternative: drag clips onto run_pipeline.bat
echo ============================================================
pause
