@echo off
setlocal EnableExtensions

REM ============================================================================
REM  H3 Video Gen — Windows launcher
REM  Creates .venv, installs deps when needed, ensures .env, starts UI,
REM  opens http://127.0.0.1:7860 when the server is ready.
REM ============================================================================

cd /d "%~dp0"
title H3 Video Gen

set "HOST=127.0.0.1"
set "PORT=7860"
set "URL=http://%HOST%:%PORT%"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "MARKER=%~dp0.venv\.deps_installed"

echo.
echo  ========================================
echo   H3 Video Gen
echo  ========================================
echo.

REM ---- Find system Python ----
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python 3 was not found on PATH.
  echo         Install Python 3.11+ and enable "Add python.exe to PATH".
  goto :fail
)

REM ---- Virtual environment ----
if not exist "%VENV_PY%" (
  echo [1/4] Creating virtual environment .venv ...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create .venv
    goto :fail
  )
  if exist "%MARKER%" del "%MARKER%" >nul 2>&1
) else (
  echo [1/4] Virtual environment already present.
)

if not exist "%VENV_PY%" (
  echo [ERROR] Expected "%VENV_PY%" after venv create.
  goto :fail
)

REM ---- Dependencies ----
if not exist "%~dp0requirements.txt" (
  echo [ERROR] requirements.txt not found in %CD%
  goto :fail
)

if not exist "%MARKER%" (
  echo [2/4] Installing Python packages ^(first run may take a few minutes^)...
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 goto :fail
  "%VENV_PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] pip install failed.
    goto :fail
  )
  >"%MARKER%" echo installed
) else (
  echo [2/4] Dependencies already installed.
  echo       Delete .venv\.deps_installed to force a reinstall.
)

REM ---- .env ----
if not exist "%~dp0.env" (
  if exist "%~dp0.env.example" (
    echo [3/4] Creating .env from .env.example ...
    copy /Y "%~dp0.env.example" "%~dp0.env" >nul
    echo.
    echo  [WARNING] Set GEMINI_API_KEY in .env for full director/critic features.
    echo.
  ) else (
    echo [3/4] No .env or .env.example — continuing with defaults.
  )
) else (
  echo [3/4] .env found.
)

REM ---- Port already in use? open browser and exit ----
netstat -ano 2>nul | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo [note] Port %PORT% is already listening — opening browser to existing server.
  start "" "%URL%"
  echo.
  echo  Close any other H3 Video Gen window if you meant to restart the server.
  echo.
  pause
  exit /b 0
)

echo [4/4] Starting server on %URL%
echo       Press Ctrl+C in this window to stop.
echo.

REM Wait for /api/health then open the browser (minimized helper window)
start "H3VG-open-browser" /MIN powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$u='%URL%/api/health'; $ok=$false; for($i=0;$i -lt 90;$i++){ try { $r=Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -lt 500){ $ok=$true; break } } catch {} ; Start-Sleep -Seconds 1 }; Start-Process '%URL%'"

"%VENV_PY%" run.py serve
set "EC=%ERRORLEVEL%"

echo.
if not "%EC%"=="0" (
  echo [ERROR] Server exited with code %EC%.
  goto :fail
)
exit /b 0

:fail
echo.
echo  Setup/launch failed. See messages above.
echo.
pause
exit /b 1
