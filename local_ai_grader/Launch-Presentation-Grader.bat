@echo off
setlocal
cd /d "%~dp0"
"%~dp0python\python.exe" "%~dp0launcher.py"
if errorlevel 1 (
  echo.
  echo Presentation Grader could not start. See the log files in %%LOCALAPPDATA%%\PresentationGrader\logs.
  pause
)
endlocal
