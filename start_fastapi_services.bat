@echo off
setlocal enableextensions

REM Change to this script's directory so module imports resolve
cd /d "%~dp0"

REM Use Python from the specific Conda env path for 'trt'
set "PYTHON_EXE=C:\Users\Admin\.conda\envs\trt\python.exe"

REM Launch Merge Server (5050)
start "Merge Server 5050" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" ai_server_fastapi.py"

REM Small delay between windows (optional)
timeout /t 1 >nul 2>nul

REM Launch Patchcore Server (8000)
start "Patchcore Server 8000" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" patchcore_server.py --host 0.0.0.0 --port 8000"

timeout /t 1 >nul 2>nul

REM Launch Paste Detection Server (8001)
start "Paste Server 8001" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" paste_detection_server.py --host 0.0.0.0 --port 8001"

timeout /t 1 >nul 2>nul

REM Launch Distance Detection Server (8002)
start "Distance Server 8002" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" distance_detection_server.py --host 0.0.0.0 --port 8002"

endlocal
