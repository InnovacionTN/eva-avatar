@echo off
REM ============================================================
REM  Eva talks - live full-body conversational avatar
REM  Double-click to run, or launch from a terminal.
REM ============================================================

REM Always run from this script's own folder (the project root)
cd /d "%~dp0"

set "PY=third_party\FLP-win\FasterLivePortrait-windows\venv\python.exe"

if not exist "%PY%" (
    echo [ERROR] FasterLivePortrait python not found:
    echo         %PY%
    echo Make sure you are running this from the project root.
    pause
    exit /b 1
)

echo Starting Eva...  (talk to her in Spanish; press Q in the window to quit)
echo Gesture keys ^(focus the window first^): n=nod  m=shake  t=tilt  l=lean-in  g=shrug
echo.

"%PY%" src\eva_platica.py --source assets\eva_body.png --preview --no-virtual-cam --vision %*

REM Keep the window open if Eva exits with an error, so you can read it
if errorlevel 1 (
    echo.
    echo [Eva exited with an error - see the messages above]
    pause
)
