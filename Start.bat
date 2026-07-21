@echo off
REM Double-click this to open the Faceless Studio control panel.
REM Leave this window open while you work. Close it when you are done.

cd /d "%~dp0"

REM Windows consoles default to a legacy codepage that cannot draw box
REM characters. 65001 is UTF-8. Harmless if it fails.
chcp 65001 >nul 2>&1


if not exist .venv\Scripts\python.exe (
  echo.
  echo   Setup has not been run yet.
  echo   Double-click setup.bat first, then come back here.
  echo.
  pause
  exit /b 1
)

echo.
echo   Starting Faceless Studio...
echo   Your browser will open in a moment.
echo   Keep this window open. Press Ctrl+C here when you are finished.
echo.

.venv\Scripts\python.exe studio.py

echo.
echo   Studio stopped.
pause
