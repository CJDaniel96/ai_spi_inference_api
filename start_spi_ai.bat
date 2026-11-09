@echo off
setlocal enableextensions
title SPI AI Controller
cd /d "%~dp0"
call "%~dp0resolve_python.bat"
if errorlevel 1 exit /b 1
if not defined AI_CONFIG_PATH set "AI_CONFIG_PATH=%~dp0config\ai_server.json"
"%PYTHON_EXE%" -m app.main %*
set "SPI_EXIT=%ERRORLEVEL%"
if not "%SPI_EXIT%"=="0" pause
exit /b %SPI_EXIT%
