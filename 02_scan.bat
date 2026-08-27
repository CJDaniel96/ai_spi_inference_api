@echo off
setlocal enableextensions

REM Change to this script's directory so module imports resolve
cd /d "%~dp0"

if /i not "%SPI_ENABLE_LEGACY_SCANNER%"=="1" (
    echo Legacy scanner is disabled to prevent duplicate processing.
    echo Set SPI_ENABLE_LEGACY_SCANNER=1 only when the durable Ingest Worker is stopped.
    exit /b 2
)

call "%~dp0resolve_python.bat"
if errorlevel 1 exit /b 1
if not defined AI_CONFIG_PATH set "AI_CONFIG_PATH=%~dp0config\ai_server.json"

REM Scan Jobs
start "SPI_02_Scan" "%PYTHON_EXE%" scan_jobs.py

endlocal
