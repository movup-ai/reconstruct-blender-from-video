@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Blender Agent - keep this window open
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (
  echo First start: setting up ^(one time, needs internet^)...
  call "%~dp0Setup.bat" /auto
  if not exist ".venv\Scripts\python.exe" (echo Setup did not finish - see the messages above. & pause & exit /b 1)
)
:run
".venv\Scripts\python.exe" -m agent %*
set RC=%ERRORLEVEL%
if "%RC%"=="0" exit /b 0
if "%RC%"=="3" exit /b 0
echo.
echo Blender Agent stopped unexpectedly (code %RC%) - restarting in 5 seconds. Close this window to stop.
echo Details: logs\agent.log
timeout /t 5 /nobreak >nul
goto run
