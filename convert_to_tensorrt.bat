@echo off
setlocal enableextensions
cd /d "%~dp0"

call "%~dp0resolve_python.bat"
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" "%~dp0convert_to_tensorrt.py" %*
set "CONVERTER_EXIT=%ERRORLEVEL%"
exit /b %CONVERTER_EXIT%
