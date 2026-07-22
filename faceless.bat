@echo off
REM The `faceless` command for Windows — the twin of the ./faceless script macOS
REM uses. No install step, no activating the venv: it finds the project's own
REM Python and runs the same code, so it works the moment setup has run.
REM
REM Run it from this folder as:   .\faceless start
REM (PowerShell needs the .\ ; plain cmd.exe does not.) After setup.bat, a bare
REM `faceless start` also works whenever the .venv is active, because setup drops
REM a copy of this launcher onto your PATH.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   Python isn't set up for this project yet. Run once:
  echo.
  echo     setup.bat
  echo.
  exit /b 1
)

REM cli.py maps the friendly verbs (start -^> studio, check -^> doctor, ...) and
REM hands off to make_video.py.
".venv\Scripts\python.exe" cli.py %*
