@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONPATH=%CD%\src"
if not exist "runtime\python.exe" (
  echo Required private Python runtime is missing. No downloads will be attempted.
  pause
  exit /b 1
)
"runtime\python.exe" "scripts\launch.py"
if errorlevel 1 pause
