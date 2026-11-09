@echo off
setlocal enableextensions
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: Install uv before running setup.bat.
    exit /b 1
)
uv sync --python 3.12 --extra inference --extra cuda
if errorlevel 1 exit /b 1
echo Setup complete. Start the controller with start_spi_ai.bat.
exit /b 0
