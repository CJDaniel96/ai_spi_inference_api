@echo off
setlocal

set ENV_NAME=spi_prd
set PYTHON_VERSION=3.9

echo Checking for conda...
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda is not found in your PATH. Please install Anaconda/Miniconda and add it to your PATH.
    exit /b 1
)

echo Creating Conda environment '%ENV_NAME%' with Python %PYTHON_VERSION%...
conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y

echo Activating environment and installing dependencies...
call conda run -n %ENV_NAME% pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
call conda run -n %ENV_NAME% pip install -r requirements.txt

echo.
echo Setup complete! To activate the environment, run: conda activate %ENV_NAME%
endlocal