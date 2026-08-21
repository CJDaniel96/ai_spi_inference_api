@echo off
setlocal enableextensions

REM Change to this script's directory so module imports resolve
cd /d "%~dp0"

REM Use Python from the specific Conda env path for 'spi_env'
set "PYTHON_EXE=C:\Users\Admin\.conda\envs\py312_cu128_j15\python.exe"

REM Scan Jobs
start "SPI_02_Scan" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" scan_jobs.py"

endlocal
