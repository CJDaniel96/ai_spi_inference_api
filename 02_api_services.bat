@echo off
setlocal enableextensions

REM Change to this script's directory so module imports resolve
cd /d "%~dp0"

call "%~dp0resolve_python.bat"
if errorlevel 1 exit /b 1
if not defined AI_CONFIG_PATH set "AI_CONFIG_PATH=%~dp0config\ai_server.json"

REM Launch Merge Server (5050) via the auto-restart wrapper (relaunches on crash).
REM Legacy fallback (deprecated): "%PYTHON_EXE%" ai_server_fastapi.py
start "SPI_02_main_5050" cmd /k call "%~dp0run_merge_server.bat"

REM Small delay between windows (optional)
timeout /t 1 >nul 2>nul

REM Launch Anomaly Server (8000)
start "SPI_02_Anomaly_8000" "%PYTHON_EXE%" patchcore_api_trt.py --host 0.0.0.0 --port 8000

@REM timeout /t 1 >nul 2>nul

REM Launch Paste Detection Server (8001)
@REM start "SPI_02_Paste_8001" "%PYTHON_EXE%" paste_detection_server.py --host 0.0.0.0 --port 8001

timeout /t 1 >nul 2>nul

REM Launch Distance Detection Server (8002)
start "SPI_02_Distance_8002" "%PYTHON_EXE%" distance_detection_api_trt.py --host 0.0.0.0 --port 8002

timeout /t 1 >nul 2>nul

REM The legacy scan_jobs.py is intentionally NOT started here.  The durable
REM Ingest Worker is the only scanner in three-stage production mode.

endlocal
