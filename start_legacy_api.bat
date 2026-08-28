@echo off
setlocal enableextensions
cd /d "%~dp0"

REM Optional synchronous compatibility API on port 5050.  The durable 01/02/03
REM pipeline does not require this process.
if not defined AI_CONFIG_PATH set "AI_CONFIG_PATH=%~dp0config\ai_server.json"
call "%~dp0run_merge_server.bat"
set "LEGACY_API_EXIT=%ERRORLEVEL%"
exit /b %LEGACY_API_EXIT%
