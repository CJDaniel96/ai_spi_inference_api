@echo off
setlocal enableextensions
cd /d "%~dp0"
call "%~dp0resolve_python.bat"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -m app.db_export %*
set "EXPORT_EXIT=%ERRORLEVEL%"
if not "%EXPORT_EXIT%"=="0" pause
exit /b %EXPORT_EXIT%
