@echo off
REM Resolve one Python 3.12 runtime for every AIPC launcher.
REM Set PYTHON_EXE before calling a launcher to override auto-detection.
REM This file deliberately avoids GOTO labels because LF-only Git checkouts can
REM make label lookup unreliable in some Windows CMD/terminal combinations.

REM Accept both: set PYTHON_EXE=C:\path\python.exe
REM and:         set PYTHON_EXE="C:\path\python.exe"
if defined PYTHON_EXE set "PYTHON_EXE=%PYTHON_EXE:"=%"

if defined PYTHON_EXE (
    if not exist "%PYTHON_EXE%" (
        echo ERROR: configured PYTHON_EXE does not exist: "%PYTHON_EXE%"
        echo Clear it with: set PYTHON_EXE=
        exit /b 1
    )
)

if not defined PYTHON_EXE if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not defined PYTHON_EXE if exist "%USERPROFILE%\.conda\envs\py312_cu128_j15\python.exe" set "PYTHON_EXE=%USERPROFILE%\.conda\envs\py312_cu128_j15\python.exe"

if not defined PYTHON_EXE (
    for /f "delims=" %%I in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
)

if not defined PYTHON_EXE (
    echo ERROR: Python was not found. Run setup.bat or set PYTHON_EXE.
    exit /b 1
)

"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.12 is required: "%PYTHON_EXE%"
    "%PYTHON_EXE%" --version
    exit /b 1
)

echo Using Python: "%PYTHON_EXE%"
exit /b 0
