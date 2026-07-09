@echo off
setlocal enableextensions

REM Change to this script's directory so module imports resolve
cd /d "%~dp0"

REM Use Python from the specific Conda env path for 'spi_env'
set "PYTHON_EXE=C:\Users\Admin\.conda\envs\py310_cu117_j15\python.exe"

REM Launch Merge Server (5050) — new modular entry point (app/).
REM Legacy fallback (deprecated): "%PYTHON_EXE%" ai_server_fastapi.py
start "SPI_02_main_5050" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" -m app.main"

REM Small delay between windows (optional)
timeout /t 1 >nul 2>nul

REM Launch Anomaly Server (8000)
start "SPI_02_Anomaly_8000" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" patchcore_api_trt.py --host 0.0.0.0 --port 8000"

@REM timeout /t 1 >nul 2>nul

REM Launch Paste Detection Server (8001)
@REM start "SPI_02_Paste_8001" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" paste_detection_server.py --host 0.0.0.0 --port 8001"

timeout /t 1 >nul 2>nul

REM Launch Distance Detection Server (8002)
start "SPI_02_Distance_8002" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" distance_detection_api_trt.py --host 0.0.0.0 --port 8002"

timeout /t 1 >nul 2>nul

REM Scan Jobs
start "SPI_02_Scan" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" scan_jobs.py"

endlocal
