@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
echo ============================================================
echo  Blender Agent - setup (safe to run again)
echo ============================================================
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
  echo Python 3.10+ was not found. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH", then run Setup.bat again.
  goto end_fail
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment...
  %PY% -m venv .venv || goto end_fail
)
rem An NVIDIA card (its driver installs nvcuda.dll) gets the GPU build; any other PC gets the CPU build.
set "REQ=requirements-cpu.txt"
set "DEV=cpu"
if exist "%SystemRoot%\System32\nvcuda.dll" (set "REQ=requirements.txt" & set "DEV=cuda")
if "%DEV%"=="cpu" echo No NVIDIA card found - installing the CPU build (the models run on the processor, slower).
echo Installing packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r %REQ% || goto end_fail
".venv\Scripts\python.exe" -m pip install rtmlib==0.0.16 --no-deps || goto end_fail
if "%DEV%"=="cuda" ".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0,'pipeline'); from common import gpu_dlls; o=gpu_dlls(); sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)" || echo   !! NVIDIA (CUDA) runtime not available - update the NVIDIA driver, then run Setup.bat again.
echo Depth model...
".venv\Scripts\python.exe" pipeline\get_model.py || echo   (depth model download failed - run Setup.bat again later)
echo Pose models (first download ~400 MB)...
".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0,'pipeline'); from common import gpu_dlls, pick_device; d=pick_device(); d=='cuda' and gpu_dlls(); from rtmlib import Body; Body(mode='performance', backend='onnxruntime', device=d); print('  pose models ready (%%s)' %% d)" || echo   (pose models will download on first use)
echo 3D pose model (MotionBERT)...
".venv\Scripts\python.exe" -c "import torch" 2>nul
if errorlevel 1 if "%DEV%"=="cpu" ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || echo   (PyTorch not installed - people are fitted without the 3D pose step)
if not exist "third_party\MotionBERT-main\lib\model\DSTformer.py" (
  if not exist third_party mkdir third_party
  curl -sS -L --retry 5 -o "third_party\mb.zip" https://github.com/Walter0807/MotionBERT/archive/refs/heads/main.zip && tar -xf "third_party\mb.zip" -C third_party && del "third_party\mb.zip"
)
if not exist "third_party\MotionBERT-main\lib\model\DSTformer.py" echo   (MotionBERT code missing - people are fitted without the 3D pose step)
if not exist "models\motionbert\FT_MB_release_MB_ft_h36m.bin" echo   (MotionBERT weights missing: models\motionbert\FT_MB_release_MB_ft_h36m.bin from huggingface.co/walterzhu/MotionBERT)
echo Lens model (MoGe-2)...
".venv\Scripts\python.exe" -c "import utils3d_moge" 2>nul || ".venv\Scripts\python.exe" -m pip install huggingface-hub "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" || echo   (utils3d not installed - every clip uses the fixed lens of config.json)
if not exist "third_party\MoGe\moge\model\v2.py" (
  if not exist third_party mkdir third_party
  curl -sS -L --retry 5 -o "third_party\moge.zip" https://github.com/microsoft/MoGe/archive/refs/heads/main.zip && tar -xf "third_party\moge.zip" -C third_party && del "third_party\moge.zip" && ren "third_party\MoGe-main" MoGe
)
if not exist "models\moge2\model.pt" echo   (MoGe-2 weights missing: models\moge2\model.pt - every clip uses the fixed lens of config.json)
if not exist "assets\sample_character.json" (
  echo NOTE: assets\sample_character.json is missing - characters fall back to box mannequins.
)
".venv\Scripts\python.exe" -m agent.selftest
echo.
echo Optional AI suggestions: paste your DeepSeek API key in the app (start page, "AI suggestions").
echo ============================================================
echo  Setup finished.  Start the app with "Start_Blender_Agent.bat"
echo ============================================================
if /i not "%~1"=="/auto" pause
exit /b 0
:end_fail
echo Setup failed - see the messages above.
if /i not "%~1"=="/auto" pause
exit /b 1
