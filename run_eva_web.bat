@echo off
REM ============================================================
REM  Eva talks - WEB version (browser + tunnel)
REM  Starts the GPU server, then opens a public tunnel URL that
REM  anyone can open in their browser to talk to Eva.
REM ============================================================

cd /d "%~dp0"

set "PY=third_party\FLP-win\FasterLivePortrait-windows\venv\python.exe"
set "CLOUDFLARED=C:\Program Files (x86)\cloudflared\cloudflared.exe"
set "PORT=8080"

if not exist "%PY%" (
    echo [ERROR] FasterLivePortrait python not found: %PY%
    pause & exit /b 1
)

echo Starting Eva web server in a separate window (loading GPU engines ~30s)...
start "Eva web server" "%PY%" src\eva_web.py --port %PORT% %*

echo.
echo Opening public tunnel - your shareable URL appears below as
echo    https://something.trycloudflare.com
echo Share that link. Keep BOTH windows open. Press Ctrl+C here to stop the tunnel.
echo.

if exist "%CLOUDFLARED%" (
    "%CLOUDFLARED%" tunnel --url http://localhost:%PORT%
) else (
    echo [WARN] cloudflared not found at %CLOUDFLARED%
    echo Server is still running locally at http://localhost:%PORT%
    echo Install cloudflared, or run your own tunnel to port %PORT%.
    pause
)
