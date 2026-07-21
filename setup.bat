@echo off
REM Creates a private Python environment for this project and installs what it
REM needs. Double-click this file, or run it from a Command Prompt.
REM
REM Everything lives in .venv inside this folder. Delete .venv and run this
REM again to start over; nothing else on your PC is touched.

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo   Faceless Studio - Windows setup
echo   ------------------------------

REM The ML stack lags new Python releases. Prefer a version the ecosystem has
REM settled on rather than the newest one installed.
set PYEXE=
for %%v in (3.12 3.11 3.13) do (
  if "!PYEXE!"=="" (
    py -%%v -c "import sys" >nul 2>&1
    if !errorlevel! equ 0 (
      set PYEXE=py -%%v
      echo   Using Python %%v
    )
  )
)

if "!PYEXE!"=="" (
  python -c "import sys" >nul 2>&1
  if !errorlevel! neq 0 (
    echo.
    echo   Python is not installed, or not on your PATH.
    echo.
    echo   Install Python 3.12 from:  https://www.python.org/downloads/
    echo   IMPORTANT: tick "Add python.exe to PATH" on the first screen.
    echo.
    pause
    exit /b 1
  )
  set PYEXE=python
  echo   Warning: using your default Python - 3.12 was not found.
  echo   If the voice engine misbehaves, install Python 3.12, delete the
  echo   .venv folder, and run this again.
)

REM Rebuild the venv if it was made by a different Python version.
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys;raise SystemExit(0 if sys.version_info[:2] in ((3,11),(3,12),(3,13)) else 1)" >nul 2>&1
  if !errorlevel! neq 0 (
    echo   Rebuilding .venv - it was built with an unsupported Python.
    rmdir /s /q .venv
  )
)

if not exist .venv\Scripts\python.exe (
  echo   Creating .venv ...
  !PYEXE! -m venv .venv
  if !errorlevel! neq 0 (
    echo   Could not create the environment. Is Python installed correctly?
    pause
    exit /b 1
  )
)
echo   .venv ready

REM All the real work happens in Python - same script macOS and Linux use.
.venv\Scripts\python.exe tools\install_deps.py
set RESULT=!errorlevel!

echo.
if !RESULT! neq 0 (
  echo   Setup finished with warnings. Read the messages above.
) else (
  echo   Setup complete. Next: check ffmpeg is installed, then run Start.bat
)
echo.
pause
exit /b !RESULT!
