@echo off
setlocal enableextensions
cd /d "%~dp0"
call "%~dp0resolve_python.bat"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -m app.tools.spi_statistics %*
set "STATISTICS_EXIT=%ERRORLEVEL%"
if not "%STATISTICS_EXIT%"=="0" pause
exit /b %STATISTICS_EXIT%
