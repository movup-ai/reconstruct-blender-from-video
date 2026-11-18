@echo off
rem Drag a version folder (or a .zip plus comparison.mp4) onto this file to run the self-check.
rem Requires Python 3.9+ from python.org; prefers the "py" launcher over the Microsoft Store stub.
setlocal
set "PYEXE="
where py >nul 2>nul && set "PYEXE=py -3"
if not defined PYEXE (
    where python >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
    echo Python 3.9+ not found. Install it from https://www.python.org/downloads/
    pause
    exit /b 2
)
%PYEXE% "%~dp0check_package.py" %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
