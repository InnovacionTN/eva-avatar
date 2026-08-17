@echo off
REM ============================================================
REM  Eva talks - version vertical 9:16 para totem / pantalla
REM  compartida en vertical. Igual que run_eva.bat pero con la
REM  fuente recortada a 9:16 (Eva centrada) y mas resolucion.
REM ============================================================

cd /d "%~dp0"

set "PY=third_party\FLP-win\FasterLivePortrait-windows\venv\python.exe"

if not exist "%PY%" (
    echo [ERROR] FasterLivePortrait python not found:
    echo         %PY%
    echo Make sure you are running this from the project root.
    pause
    exit /b 1
)

echo Starting Eva (totem 9:16)...  (talk to her in Spanish; press Q in the window to quit)
echo Arrastra la ventana al totem HDMI y presiona F = pantalla completa
echo Gesture keys ^(focus the window first^): n=nod  m=shake  t=tilt  l=lean-in  g=shrug
echo.

"%PY%" src\eva_platica.py --source assets\eva_body_glow_916.png --source-max-dim 1600 --out-width 864 --out-height 1536 --preview --no-virtual-cam --vision %*

if errorlevel 1 (
    echo.
    echo [Eva exited with an error - see the messages above]
    pause
)
