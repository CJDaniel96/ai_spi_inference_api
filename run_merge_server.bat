@echo off
REM ---------------------------------------------------------------------------
REM Auto-restart wrapper for the merge server (new modular entry point).
REM Relaunches `python -m app.main` if it exits/crashes, so a single failure
REM does not leave the service down. Close this window (or Ctrl+C -> Y) to stop.
REM For a proper production setup, run this under a service manager instead
REM (e.g. NSSM: `nssm install SPI_Merge <PYTHON_EXE> -m app.main`), which gives
REM auto-start on boot, crash recovery, and graceful stop.
REM ---------------------------------------------------------------------------
setlocal enableextensions
cd /d "%~dp0"

call "%~dp0resolve_python.bat"
if errorlevel 1 exit /b 1
if not defined AI_CONFIG_PATH set "AI_CONFIG_PATH=%~dp0config\ai_server.json"

:loop
echo [%date% %time%] Starting merge server (python -m app.main) ...
"%PYTHON_EXE%" -m app.main
echo [%date% %time%] Merge server exited (code %errorlevel%). Restarting in 5s ...
timeout /t 5 >nul 2>nul
goto loop

endlocal
