@echo off
setlocal enableextensions
cd /d "%~dp0"
call "%~dp0resolve_python.bat"
if errorlevel 1 exit /b 1
if not defined AI_CONFIG_PATH set "AI_CONFIG_PATH=%~dp0config\ai_server.json"
"%PYTHON_EXE%" -m app.pipeline ingest
set "PIPELINE_EXIT=%ERRORLEVEL%"
if not "%PIPELINE_EXIT%"=="0" pause
exit /b %PIPELINE_EXIT%
