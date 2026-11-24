@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
echo ============================================================
echo  Blender Agent - setup (safe to run again)
echo ============================================================
rem GPU only: every model runs on an NVIDIA card (its driver installs nvcuda.dll). There is no processor build.
if not exist "%SystemRoot%\System32\nvcuda.dll" (
  echo No NVIDIA card or driver was found. Blender Agent needs an NVIDIA card: install the NVIDIA driver,
  echo then run Setup.bat again.
  goto end_fail
)
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
set "VPY=.venv\Scripts\python.exe"
set "TORCH_VER=2.8.0+cu126"
set "TV_VER=0.23.0+cu126"
set "TRT_VER=10.14.1.48"
"%VPY%" -m pip install --upgrade pip >nul

rem The processor build of onnxruntime shares its folder with onnxruntime-gpu: take both out, the GPU one is installed below.
"%VPY%" -m pip show onnxruntime >nul 2>nul
if not errorlevel 1 (
  echo Removing the processor build of onnxruntime...
  "%VPY%" -m pip uninstall -y onnxruntime onnxruntime-gpu >nul
)

echo PyTorch %TORCH_VER% (NVIDIA build)...
"%VPY%" -c "import sys, torch, torchvision; sys.exit(0 if torch.__version__ == '%TORCH_VER%' and torchvision.__version__ == '%TV_VER%' else 1)" 2>nul
if not errorlevel 1 goto torch_done
rem the wheels saved in downloads\torch are used when they fit this Python; otherwise they are downloaded (about 2.8 GB)
"%VPY%" -m pip install --no-deps --no-index --find-links downloads\torch "torch==%TORCH_VER%" "torchvision==%TV_VER%" 2>nul
if not errorlevel 1 goto torch_done
"%VPY%" -m pip install --no-deps "torch==%TORCH_VER%" "torchvision==%TV_VER%" --index-url https://download.pytorch.org/whl/cu126 || goto end_fail
:torch_done

echo TensorRT %TRT_VER% runtime...
if exist ".venv\Lib\site-packages\tensorrt_libs\nvinfer_10.dll" goto trt_done
"%VPY%" -m pip install --no-deps --no-index --find-links downloads\trt "tensorrt-cu12-libs==%TRT_VER%" 2>nul
if not errorlevel 1 goto trt_done
"%VPY%" -m pip install --no-deps "tensorrt-cu12-libs==%TRT_VER%" || echo   !! TensorRT not installed - the pose models use plain CUDA, slower.
:trt_done

echo Installing packages...
"%VPY%" -m pip install -r requirements.txt || goto end_fail
"%VPY%" -m pip install rtmlib==0.0.16 --no-deps || goto end_fail
"%VPY%" -c "import utils3d_moge" 2>nul || "%VPY%" -m pip install "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" || echo   !! utils3d not installed - every clip uses the fixed lens of config.json

echo Checking the NVIDIA card...
"%VPY%" -c "import sys; sys.path.insert(0,'pipeline'); from common import gpu_dlls; o=gpu_dlls(); import torch; ok='CUDAExecutionProvider' in o.get_available_providers() and torch.cuda.is_available(); print('  %%s, onnxruntime %%s' %% (torch.cuda.get_device_name(0) if ok else 'NOT AVAILABLE', o.get_available_providers()[:2])); sys.exit(0 if ok else 1)"
if errorlevel 1 (
  echo The NVIDIA [CUDA] runtime is not available - update the NVIDIA driver, then run Setup.bat again.
  goto end_fail
)

echo Depth model...
"%VPY%" pipeline\get_model.py || echo   (depth model download failed - run Setup.bat again later)
echo 3D pose model (MotionBERT)...
if not exist "third_party\MotionBERT-main\lib\model\DSTformer.py" (
  if not exist third_party mkdir third_party
  curl -sS -L --retry 5 -o "third_party\mb.zip" https://github.com/Walter0807/MotionBERT/archive/refs/heads/main.zip && tar -xf "third_party\mb.zip" -C third_party && del "third_party\mb.zip"
)
if not exist "third_party\MotionBERT-main\lib\model\DSTformer.py" echo   !! MotionBERT code missing - people are fitted without the 3D pose step
if not exist "models\motionbert\FT_MB_release_MB_ft_h36m.bin" echo   !! MotionBERT weights missing: models\motionbert\FT_MB_release_MB_ft_h36m.bin from huggingface.co/walterzhu/MotionBERT
echo Lens and set models (MoGe-2, SAM 2.1, CLIP)...
if not exist "third_party\MoGe\moge\model\v2.py" (
  if not exist third_party mkdir third_party
  curl -sS -L --retry 5 -o "third_party\moge.zip" https://github.com/microsoft/MoGe/archive/refs/heads/main.zip && tar -xf "third_party\moge.zip" -C third_party && del "third_party\moge.zip" && ren "third_party\MoGe-main" MoGe
)
if not exist "third_party\sam2\sam2\build_sam.py" (
  if not exist third_party mkdir third_party
  curl -sS -L --retry 5 -o "third_party\sam2.zip" https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip && tar -xf "third_party\sam2.zip" -C third_party && del "third_party\sam2.zip" && ren "third_party\sam2-main" sam2
)
if not exist "models\moge2\model.pt" echo   !! MoGe-2 weights missing: models\moge2\model.pt - every clip uses the fixed lens of config.json
if not exist "models\sam2\sam2.1_hiera_large.pt" echo   !! SAM 2.1 weights missing: models\sam2\sam2.1_hiera_large.pt - no automatic set
if not exist "models\clip\open_clip_model.safetensors" echo   !! CLIP weights missing: models\clip\open_clip_model.safetensors - set pieces are named from shape and colour only
echo Carried-object model (Grounding DINO, about 0.9 GB the first time)...
if not exist "models\grounding_dino\model.safetensors" (
  "%VPY%" -c "from huggingface_hub import snapshot_download as s; s('IDEA-Research/grounding-dino-base', local_dir=r'models\grounding_dino', allow_patterns=['*.json','*.safetensors','*.txt'])" || echo   !! Grounding DINO not downloaded - bags, phones and backpacks people carry are not built
)
echo Set segmentation model (Mask2Former ADE20K, about 0.8 GB the first time)...
if not exist "models\mask2former_ade\model.safetensors" (
  "%VPY%" -c "from huggingface_hub import snapshot_download as s; s('facebook/mask2former-swin-large-ade-semantic', local_dir=r'models\mask2former_ade', allow_patterns=['*.json','*.safetensors'])" || echo   !! Mask2Former not downloaded - set pieces are named by CLIP and plants / curtains use library shapes
)
echo Pose models (first download ~600 MB; the first TensorRT engine build takes about 8 minutes)...
"%VPY%" -c "from rtmlib import Body; from rtmlib.tools.file import download_checkpoint as d; m=Body.MODE['performance']; d(m['det']); d(m['pose'])" || echo   !! pose models not downloaded - they are downloaded on first use
rem the person detector, batched: the picture and its 4 tiles in one GPU call (same boxes, about 3x faster)
"%VPY%" pipeline\det_batch_model.py || echo   !! batched person detector not made - the detector takes one picture a call (slower)
"%VPY%" -c "import automark; m=automark.Marker(); print('  pose models ready (%%s)' %% m.engine)" || echo   !! pose models not ready - they are downloaded and built on first use
if not exist "assets\sample_character.json" (
  echo NOTE: assets\sample_character.json is missing - characters fall back to box mannequins.
)
"%VPY%" -m agent.selftest
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
