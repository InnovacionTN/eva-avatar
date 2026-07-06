@echo off
REM ============================================================
REM  Eva talks - KIOSK version (vertical / portrait screen)
REM  Starts the web server tuned for a 9:16 kiosk display
REM  (Android tablet/totem on the same Wi-Fi/LAN as this PC).
REM  On the kiosk, open:  http://<IP-de-esta-PC>:8080
REM ============================================================

cd /d "%~dp0"

set "PY=third_party\FLP-win\FasterLivePortrait-windows\venv\python.exe"
set "PORT=8080"

if not exist "%PY%" (
    echo [ERROR] FasterLivePortrait python not found: %PY%
    pause & exit /b 1
)

echo Direcciones de esta PC (abre una de estas en el navegador del kiosko):
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike '169.254.*' -and $_.IPAddress -ne '127.0.0.1'}).IPAddress | ForEach-Object { 'http://' + $_ + ':%PORT%' }"') do echo    %%i
echo.
echo Cargando motores GPU (~30s)...  Ctrl+C para detener.
echo.

"%PY%" src\eva_web.py --port %PORT% --source assets\eva_body_vertical.png --source-max-dim 1600 --height 1280 %*

if errorlevel 1 (
    echo.
    echo [Eva salio con un error - revisa los mensajes de arriba]
    pause
)
