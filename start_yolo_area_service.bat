@echo off
setlocal enableextensions
cd /d "%~dp0"
call "%~dp0resolve_python.bat"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -m app.area_service --config "%~dp0config\yolo_area.json" %*
endlocal
