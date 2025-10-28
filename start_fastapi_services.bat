@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Change to the directory of this script so relative imports work
pushd %~dp0

REM Activate Conda environment 'trt'
call conda activate trt
if errorlevel 1 (
  echo Failed to activate Conda environment 'trt'. Ensure conda is initialized in CMD and the env exists.
  goto :end
)

REM Start the merge server (FastAPI) on port 5050
start "ai-merge-5050" cmd /c "uvicorn ai_server_fastapi:app --host 0.0.0.0 --port 5050"

REM Start three inference services (stubs) on ports 8000, 8001, 8002
REM Replace 'inference_stub:app' with your real services if available
start "inference-8000" cmd /c "uvicorn inference_stub:app --host 0.0.0.0 --port 8000"
start "inference-8001" cmd /c "uvicorn inference_stub:app --host 0.0.0.0 --port 8001"
start "inference-8002" cmd /c "uvicorn inference_stub:app --host 0.0.0.0 --port 8002"

echo Launched 4 FastAPI services in separate windows.
echo - Merge:     http://127.0.0.1:5050/process
echo - Inference: http://127.0.0.1:8000/inference, :8001, :8002

:end
popd
endlocal

