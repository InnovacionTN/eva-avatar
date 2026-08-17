@echo off
REM ============================================================
REM  Eva talks - LED film P8 (0.96 x 2.30 m = 120 x 288 leds)
REM  El controlador del LED escala TODO el escritorio 1920x1080
REM  al panel vertical, asi que Eva se manda pre-distorsionada:
REM  lienzo 960x2304 (1:2.4, el aspecto real del panel) estirado
REM  a pantalla completa con --stretch. En el LED se ve correcta.
REM
REM  Uso: corre este .bat, arrastra la ventana al monitor del LED
REM  y presiona F (pantalla completa). Q para salir.
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

echo Camaras detectadas:
"%PY%" -c "from pygrabber.dshow_graph import FilterGraph; devs = FilterGraph().get_input_devices(); print('\n'.join(f'  [{i}] {n}' for i, n in enumerate(devs)))" 2>nul
echo.

set "CAM=0"
set /p "CAM=Elige la camara para la vision de Eva (numero o nombre, Enter = 0): "

echo.
echo Starting Eva (LED film P8)...  Arrastra la ventana al monitor del LED y presiona F
echo Gesture keys ^(focus the window first^): n=nod  m=shake  t=tilt  l=lean-in  g=shrug
echo.

"%PY%" src\eva_platica.py --source assets\eva_body_led.png --source-max-dim 1280 --out-width 960 --out-height 2304 --stretch --preview --no-virtual-cam --vision --vision-camera "%CAM%" --mic-device umc %*

if errorlevel 1 (
    echo.
    echo [Eva exited with an error - see the messages above]
    pause
)
