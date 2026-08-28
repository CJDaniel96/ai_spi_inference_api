@echo off
setlocal enableextensions

REM Start only the model services required by Stage 02.  The optional legacy
REM 5050 API and legacy scanner have separate, explicitly named launchers.
cd /d "%~dp0"

call "%~dp0resolve_python.bat"
if errorlevel 1 exit /b 1
if not defined AI_CONFIG_PATH set "AI_CONFIG_PATH=%~dp0config\ai_server.json"
if not defined SPI_MODEL_DEVICE set "SPI_MODEL_DEVICE=cuda:0"
if not defined PATCHCORE_MODEL_PATH set "PATCHCORE_MODEL_PATH=models\patchcore\model_fp16.engine"
if not defined DISTANCE_CENTER_MODEL_PATH set "DISTANCE_CENTER_MODEL_PATH=models\distance\center.engine"
if not defined DISTANCE_PAD_MODEL_PATH set "DISTANCE_PAD_MODEL_PATH=models\distance\pad.engine"

REM Required PatchCore anomaly model (8000)
REM PATCHCORE_MODEL_ARGS can carry trusted .pt options such as --trust-pickle.
start "SPI_Model_Anomaly_8000" "%PYTHON_EXE%" patchcore_api_trt.py --host 0.0.0.0 --port 8000 --model-path "%PATCHCORE_MODEL_PATH%" --device "%SPI_MODEL_DEVICE%" %PATCHCORE_MODEL_ARGS%

@REM timeout /t 1 >nul 2>nul

REM Optional paste detection model (8001)
@REM start "SPI_Model_Paste_8001" "%PYTHON_EXE%" paste_detection_server.py --host 0.0.0.0 --port 8001 --yolo-model "%PASTE_MODEL_PATH%" --device "%SPI_MODEL_DEVICE%"

timeout /t 1 >nul 2>nul

REM Required distance detection model (8002)
start "SPI_Model_Distance_8002" "%PYTHON_EXE%" distance_detection_api_trt.py --host 0.0.0.0 --port 8002 --center-model "%DISTANCE_CENTER_MODEL_PATH%" --pad-model "%DISTANCE_PAD_MODEL_PATH%" --device "%SPI_MODEL_DEVICE%"

timeout /t 1 >nul 2>nul

REM In production, verify both /health payloads report status=healthy before
REM starting Stage 02.  HTTP 200 alone may mean an engine is still initializing.

endlocal
