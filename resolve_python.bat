@echo off
REM Resolve one Python 3.12 runtime for every AIPC launcher.
REM Set PYTHON_EXE before calling a launcher to override auto-detection.

if defined PYTHON_EXE (
    if exist "%PYTHON_EXE%" goto validate_python
    echo ERROR: configured PYTHON_EXE does not exist: "%PYTHON_EXE%"
    exit /b 1
)

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    goto validate_python
)

if exist "%USERPROFILE%\.conda\envs\py312_cu128_j15\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\.conda\envs\py312_cu128_j15\python.exe"
    goto validate_python
)

for /f "delims=" %%I in ('where python 2^>nul') do (
    if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
)

if not defined PYTHON_EXE (
    echo ERROR: Python was not found. Run setup.bat or set PYTHON_EXE.
    exit /b 1
)

:validate_python
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.12 is required: "%PYTHON_EXE%"
    exit /b 1
)
exit /b 0
