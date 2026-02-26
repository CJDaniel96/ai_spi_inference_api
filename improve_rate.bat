@echo off
setlocal enableextensions

REM Change to this script's directory so module imports resolve
cd /d "%~dp0"

REM Use Python from the specific Conda env path for 'spi_env'
set "PYTHON_EXE=C:\Users\Admin\.conda\envs\py310_cu117_j15\python.exe"

echo ========================================
echo   SPI+AI Analytics Mode Selection
echo ========================================
echo 1. Improve Rate
echo 2. Defect Analysis
echo ========================================
set /p choice="Select mode (1 or 2, default is 1): "

set "MODE=improve_rate"
if "%choice%"=="1" set "MODE=improve_rate"
if "%choice%"=="2" set "MODE=defect_analysis"

echo Starting %MODE%...

REM Launch Analytics Script
start "improve rate - %MODE%" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" test\cal_improve_rate.py --mode %MODE%"

endlocal
