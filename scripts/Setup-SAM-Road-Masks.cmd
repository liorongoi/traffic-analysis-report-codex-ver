@echo off
setlocal
cd /d "%~dp0.."
if not exist "runtime\python.exe" (
  echo RoadLoom's private Python runtime was not found. Install RoadLoom first.
  pause
  exit /b 1
)
echo This optional setup downloads SAM 3.1 and installs its CPU runtime into RoadLoom.
echo It does not change the RoadLoom installer. Close the app before continuing.
"runtime\python.exe" "scripts\setup-sam3.1.py" %*
if errorlevel 1 pause
